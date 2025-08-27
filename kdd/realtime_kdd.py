import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict, deque
import joblib
import tensorflow as tf
import argparse
import warnings
warnings.filterwarnings("ignore")

# ====================================
# 1. Flow tracker
# ====================================
flows = defaultdict(lambda: deque(maxlen=1000))

# ====================================
# 2. Load models based on mode
# ====================================
def load_models(mode):
    models = {}
    if mode == "autoencoder":
        models["ae"] = tf.keras.models.load_model("dae_model.h5")
        preproc = joblib.load("preprocessing.pkl")
        models["scalers"] = preproc["scalers"]
        models["expected_cols"] = preproc["columns"]
        models["threshold"] = 0.01
    elif mode == "rf":
        models["rf"] = joblib.load("random_forest.pkl")
        models["encoder_model"] = tf.keras.models.load_model("encoder_model.h5")
        models["scaler"] = joblib.load("scaler.pkl")
        models["encoders"] = joblib.load("onehot_encoder.pkl")
        models["categorical_cols"] = ["protocol_type", "service", "flag"]
        models["attack_map"] = ['normal','dos','probe','r2l','u2r']
    elif mode == "knn":
        models["scaler"] = joblib.load("scaler.pkl")
        models["knn"] = joblib.load("knn_model.pkl")
        models["label_encoder"] = joblib.load("label_encoder.pkl")
        models["enc"] = joblib.load("onehot_encoder.pkl")
        models["categorical_cols"] = ["protocol_type", "service", "flag"]
    return models

# ====================================
# 3. Feature extraction
# ====================================
def compute_flow_features(flow_packets):
    if not flow_packets:
        return None
    pkt0 = flow_packets[0]
    proto = "tcp" if pkt0.haslayer(TCP) else "udp" if pkt0.haslayer(UDP) else "icmp"
    features = {
        "duration": 0,
        "protocol_type": proto,
        "service": "http",
        "flag": "SF",
        "src_bytes": sum(len(pkt) for pkt in flow_packets),
        "dst_bytes": sum(len(pkt) for pkt in flow_packets),
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 0,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": len(flow_packets),
        "srv_count": len(flow_packets),
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": len(flow_packets),
        "dst_host_srv_count": len(flow_packets),
        "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    }
    return features

# ====================================
# 4. Prediction callbacks
# ====================================
def predict_autoencoder(features, models, flow_key):
    df = pd.DataFrame([features])
    # scale numeric
    for col, scaler in models["scalers"].items():
        if col in df.columns:
            df[col] = scaler.transform(df[col].to_numpy().reshape(-1, 1))
    missing_cols = [col for col in models["expected_cols"] if col not in df.columns]
    if missing_cols:
        df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1)
    df = df[models["expected_cols"]]
    X = df.values.astype(np.float32)

    reconstruction = models["ae"].predict(X, verbose=0)
    loss = np.mean((X - reconstruction) ** 2)
    if loss > models["threshold"]:
        print(f"[AE ALERT] Flow {flow_key} anomaly detected (loss={loss:.5f})")
    else:
        print(f"[AE] Flow {flow_key} normal (loss={loss:.5f})")

def predict_rf(features, models, flow_key):
    df = pd.DataFrame([features])
    df_cat = df[models["categorical_cols"]]
    df_num = df.drop(columns=models["categorical_cols"])
    X_cat = models["encoders"].transform(df_cat)
    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()
    X_all = np.hstack([df_num.values, X_cat])
    X_scaled = models["scaler"].transform(X_all)
    # Pad to 128 features
    if X_scaled.shape[1] < 128:
        pad = np.zeros((X_scaled.shape[0], 128 - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, pad])
    encoded = models["encoder_model"].predict(X_scaled, verbose=0)
    y_pred_enc = models["rf"].predict(encoded)
    y_pred = models["attack_map"][y_pred_enc[0]]
    print(f"[RF] Flow {flow_key} => {y_pred}")

def predict_knn(features, models, flow_key):
    df = pd.DataFrame([features])
    cat_enc = models["enc"].transform(df[models["categorical_cols"]])
    df = df.drop(models["categorical_cols"], axis=1)
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(cat_enc)], axis=1)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_scaled = models["scaler"].transform(df)
    y_pred_enc = models["knn"].predict(X_scaled)
    y_pred = models["label_encoder"].inverse_transform(y_pred_enc)
    print(f"[KNN] Flow {flow_key} => {y_pred[0]}")

# ====================================
# 5. Main packet callback
# ====================================
def predict_packet(pkt, mode, models):
    if not pkt.haslayer(IP):
        return
    flow_key = (pkt[IP].src, pkt[IP].dst, pkt[IP].proto)
    flows[flow_key].append(pkt)
    features = compute_flow_features(list(flows[flow_key]))
    if not features:
        return

    if mode == "autoencoder":
        predict_autoencoder(features, models, flow_key)
    elif mode == "rf":
        predict_rf(features, models, flow_key)
    elif mode == "knn":
        predict_knn(features, models, flow_key)

# ====================================
# 6. Run sniffing
# ====================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", type=str, default="Wi-Fi", help="Network interface")
    parser.add_argument("--mode", type=str, choices=["autoencoder","rf","knn"], required=True)
    args = parser.parse_args()

    models = load_models(args.mode)
    print(f"[INFO] Starting IDS in {args.mode.upper()} mode on interface {args.iface}...")
    sniff(iface=args.iface, prn=lambda pkt: predict_packet(pkt, args.mode, models), store=False)
