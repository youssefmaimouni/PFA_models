import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
import joblib
from collections import defaultdict, deque
import time

# =====================
# 1. Load model and preprocessing
# =====================
scaler = joblib.load("scaler.pkl")
knn = joblib.load("knn_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
enc = joblib.load("onehot_encoder.pkl")  # one-hot encoder
categorical_cols = ["protocol_type", "service", "flag"]

# =====================
# 2. Flow tracker
# =====================
flows = defaultdict(lambda: deque(maxlen=1000))  # store last 1000 packets per flow

# =====================
# 3. Feature computation
# =====================
def compute_flow_features(flow_packets):
    """Compute simplified KDD-like features for a flow."""
    features = {
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "src_bytes": 0,
        "dst_bytes": 0,
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
    # Sum packet lengths for src_bytes/dst_bytes
    features["src_bytes"] = sum(len(pkt) for pkt in flow_packets)
    features["dst_bytes"] = sum(len(pkt) for pkt in flow_packets)
    # Protocol
    pkt0 = flow_packets[0]
    if pkt0.haslayer(TCP):
        features["protocol_type"] = "tcp"
    elif pkt0.haslayer(UDP):
        features["protocol_type"] = "udp"
    else:
        features["protocol_type"] = "icmp"
    return features

# =====================
# 4. Prediction callback
# =====================
def predict_packet(pkt):
    if not pkt.haslayer(IP):
        return
    flow_key = (pkt[IP].src, pkt[IP].dst, pkt[IP].proto)
    flows[flow_key].append(pkt)
    
    # Compute features on last N packets
    features = compute_flow_features(list(flows[flow_key]))
    df = pd.DataFrame([features])

    # One-hot encode categorical features
    cat_enc = enc.transform(df[categorical_cols])
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(cat_enc)], axis=1)

    # Ensure numeric
    df.columns = df.columns.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    y_pred_enc = knn.predict(df_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    print(f"Flow {flow_key} predicted attack category: {y_pred[0]}")

# =====================
# 5. Start sniffing
# =====================
sniff(iface="Wi-Fi", prn=predict_packet, store=False)
