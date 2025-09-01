import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict, deque
import joblib
from tensorflow.keras.models import load_model
import warnings
from colorama import Fore, Style, init
import argparse

warnings.filterwarnings("ignore")
init(autoreset=True)

LABEL_COLORS = {
    "normal": Fore.GREEN,
    "dos": Fore.RED,
    "probe": Fore.YELLOW,
    "r2l": Fore.MAGENTA,
    "u2r": Fore.CYAN,
    "unknown": Fore.WHITE
}

# =======================
# 1. Argument parser
# =======================
parser = argparse.ArgumentParser(description="Real-time IDS")
parser.add_argument("--iface", type=str, required=True, help="Network interface to sniff")
parser.add_argument("--model", type=str, required=True, choices=["rf", "knn", "autoencoder", "hybrid", "dt"],
                    help="Choose model: rf, knn, autoencoder, hybrid, dt")

args = parser.parse_args()
IFACE = args.iface
MODEL_CHOICE = args.model.lower()

# =======================
# 2. Flow tracker
# =======================
flows = defaultdict(lambda: deque(maxlen=1000))  # last 1000 packets per flow

# =======================
# 3. Load models & preprocessing
# =======================
if MODEL_CHOICE == "rf":
    rf_model = joblib.load("rf_model_rfe.pkl")
    rfe_selector = joblib.load("rfe_selector.pkl")
    le_protocol = joblib.load("le_protocol.pkl")
    le_service  = joblib.load("le_service.pkl")
    le_flag     = joblib.load("le_flag.pkl")
    categorical_cols = ["protocol_type", "service", "flag"]
    attack_map = ['normal', 'dos', 'probe', 'r2l', 'u2r']

elif MODEL_CHOICE == "knn":
    scaler = joblib.load("scaler.pkl")
    knn = joblib.load("knn_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    enc = joblib.load("onehot_encoder.pkl")
    categorical_cols = ["protocol_type", "service", "flag"]

elif MODEL_CHOICE == "autoencoder":
    autoencoder = load_model("dae_model.h5")
    preproc = joblib.load("preprocessing.pkl")
    scalers = preproc["scalers"]
    expected_cols = preproc["columns"]
    RECONSTRUCTION_THRESHOLD = 0.01

elif MODEL_CHOICE == "hybrid":
    rf_model = joblib.load("random_forest.pkl")
    encoder_model = load_model("encoder_model.h5")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("onehot_encoder.pkl")
    categorical_cols = ["protocol_type", "service", "flag"]
    attack_map = ['probe', 'dos', 'normal', 'r2l', 'u2r']

elif MODEL_CHOICE == "dt":
    dt_rfe = joblib.load("dt_selected_features_model.pkl")  # Modèle Decision Tree avec RFE
    rfe_selector = joblib.load("rfe_selector.pkl")          # Sélecteur RFE
    le_protocol = joblib.load("le_protocol.pkl")
    le_service  = joblib.load("le_service.pkl")
    le_flag     = joblib.load("le_flag.pkl")
    categorical_cols = ["protocol_type", "service", "flag"]
    attack_map = ['normal', 'dos', 'probe', 'r2l', 'u2r']


else:
    raise ValueError("Unsupported model choice!")

# =======================
# 4. Common feature computation
# =======================
def compute_flow_features(flow_packets):
    if not flow_packets:
        return None
    pkt0 = flow_packets[0]
    proto = "tcp" if pkt0.haslayer(TCP) else "udp" if pkt0.haslayer(UDP) else "icmp"
    src_bytes = sum(len(pkt[IP].payload) for pkt in flow_packets if pkt.haslayer(IP))
    dst_bytes = src_bytes
    features = {
        "duration": 0,
        "protocol_type": proto,
        "service": "http",
        "flag": "SF",
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
        "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
        "is_host_login": 0, "is_guest_login": 0,
        "count": len(flow_packets), "srv_count": len(flow_packets),
        "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0, "same_srv_rate": 1.0, "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0, "dst_host_count": len(flow_packets),
        "dst_host_srv_count": len(flow_packets), "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    }
    return features

# =======================
# 5. Preprocessing helpers
# =======================
def preprocess_rf(df):
    df['protocol_type'] = le_protocol.transform(df['protocol_type'])
    df['service']       = le_service.transform(df['service'])
    df['flag']          = le_flag.transform(df['flag'])
    return rfe_selector.transform(df)

def preprocess_knn(df):
    cat_enc = enc.transform(df[categorical_cols])
    if hasattr(cat_enc, "toarray"):
        cat_enc = cat_enc.toarray()
    df = df.drop(categorical_cols, axis=1).reset_index(drop=True)
    df_cat = pd.DataFrame(cat_enc, columns=[str(c) for c in range(cat_enc.shape[1])])
    df_all = pd.concat([df, df_cat], axis=1)
    
    # Forcer toutes les colonnes en float et noms en string
    df_all = df_all.astype(float)
    df_all.columns = df_all.columns.astype(str)
    
    # Scale
    return scaler.transform(df_all)


def preprocess_autoencoder(features):
    df = pd.DataFrame([features])
    for col, scaler_obj in scalers.items():
        if col in df.columns:
            df[col] = scaler_obj.transform(df[col].to_numpy().reshape(-1,1))
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        df_missing = pd.DataFrame(0, index=df.index, columns=missing_cols)
        df = pd.concat([df, df_missing], axis=1)
    df = df[expected_cols]
    return df.values.astype(np.float32)

def preprocess_hybrid(df):
    df_cat = df[categorical_cols]
    df_num = df.drop(columns=categorical_cols)
    X_cat = encoders.transform(df_cat)
    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()
    X_all = np.hstack([df_num.values, X_cat])
    X_scaled = scaler.transform(X_all)
    expected_dim = 128
    if X_scaled.shape[1] < expected_dim:
        padding = np.zeros((X_scaled.shape[0], expected_dim - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, padding])
    return X_scaled.astype(np.float32)

def preprocess_dt(df):
    df['protocol_type'] = le_protocol.transform(df['protocol_type'])
    df['service']       = le_service.transform(df['service'])
    df['flag']          = le_flag.transform(df['flag'])
    return rfe_selector.transform(df)


# =======================
# 6. Packet prediction
# =======================
def predict_packet(pkt):
    try:
        if not pkt.haslayer(IP):
            return
        flow_key = (pkt[IP].src, pkt[IP].dst, pkt[IP].proto)
        flows[flow_key].append(pkt)
        features = compute_flow_features(list(flows[flow_key]))
        if not features:
            return
        df = pd.DataFrame([features])

        if MODEL_CHOICE == "rf":
            X = preprocess_rf(df)
            y_pred_enc = rf_model.predict(X)
            label = attack_map[y_pred_enc[0]]
        elif MODEL_CHOICE == "knn":
            X = preprocess_knn(df)
            y_pred_enc = knn.predict(X)
            y_pred = label_encoder.inverse_transform(y_pred_enc)
            label = y_pred[0]
        elif MODEL_CHOICE == "autoencoder":
            X = preprocess_autoencoder(features)
            recon = autoencoder.predict(X, verbose=0)
            loss = np.mean((X - recon)**2)
            if loss > RECONSTRUCTION_THRESHOLD:
                label = "anomaly"
            else:
                label = "normal"
        elif MODEL_CHOICE == "hybrid":
            X = preprocess_hybrid(df)
            encoded = encoder_model.predict(X, verbose=0)
            y_pred_enc = rf_model.predict(encoded)
            label = attack_map[y_pred_enc[0]]
        elif MODEL_CHOICE == "dt":
            X = preprocess_dt(df)
            y_pred_enc = dt_rfe.predict(X)
            label = attack_map[y_pred_enc[0]]

        else:
            label = "unknown"

        color = LABEL_COLORS.get(label.lower(), Fore.WHITE)
        print(f"[{MODEL_CHOICE.upper()}] Flow {flow_key} => {color}{label}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}[ERROR] {e}{Style.RESET_ALL}")

# =======================
# 7. Start sniffing
# =======================
print(f"[INFO] Starting real-time IDS on iface={IFACE} using model={MODEL_CHOICE.upper()}")
sniff(iface=IFACE, prn=predict_packet, store=False)
