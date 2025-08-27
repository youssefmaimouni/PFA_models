import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict, deque
import joblib
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
# =====================
# 1. Load model + preprocessing
# =====================
autoencoder = tf.keras.models.load_model("dae_model.h5")
preproc = joblib.load("preprocessing.pkl")
scalers = preproc["scalers"]        # dict of column_name: scaler
expected_cols = preproc["columns"]  # list of columns expected by the model

# Threshold (tune for your dataset)
RECONSTRUCTION_THRESHOLD = 0.01  

# =====================
# 2. Flow tracker
# =====================
flows = defaultdict(lambda: deque(maxlen=1000))  # store last 1000 packets per flow

# =====================
# 3. Feature computation
# =====================
def compute_flow_features(flow_packets):
    """Compute simplified KDD-like features for a flow."""
    if not flow_packets:
        return {}
    
    pkt0 = flow_packets[0]
    
    features = {
        "duration": 0,
        "protocol_type": "tcp" if pkt0.haslayer(TCP) else "udp" if pkt0.haslayer(UDP) else "icmp",
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

# =====================
# 4. Preprocessing for model (optimized)
# =====================
def preprocess_input(features_dict):
    df = pd.DataFrame([features_dict])

    # Scale numeric features
    for col, scaler in scalers.items():
        if col in df.columns:
            df[col] = scaler.transform(df[col].to_numpy().reshape(-1, 1))

    # Find missing columns
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        df_missing = pd.DataFrame(0, index=df.index, columns=missing_cols)
        df = pd.concat([df, df_missing], axis=1)

    # Ensure correct order
    df = df[expected_cols]

    return df.values.astype(np.float32)

# =====================
# 5. Prediction callback
# =====================
def predict_packet(pkt):
    try:
        if not pkt.haslayer(IP):
            return

        flow_key = (pkt[IP].src, pkt[IP].dst, pkt[IP].proto)
        flows[flow_key].append(pkt)

        # Extract features
        features = compute_flow_features(list(flows[flow_key]))
        X = preprocess_input(features)

        # Autoencoder reconstruction
        reconstruction = autoencoder.predict(X, verbose=0)
        loss = np.mean((X - reconstruction) ** 2)

        # Anomaly detection
        if loss > RECONSTRUCTION_THRESHOLD:
            print(f"[ALERT] Flow {flow_key} anomaly detected (loss={loss:.5f})")
        else:
            print(f"Flow {flow_key} normal (loss={loss:.5f})")
    except Exception as e:
        print(f"[ERROR] Packet processing failed: {e}")

# =====================
# 6. Start sniffing
# =====================
print("Starting real-time IDS... Press Ctrl+C to stop.")
sniff(iface="Wi-Fi", prn=predict_packet, store=False)
