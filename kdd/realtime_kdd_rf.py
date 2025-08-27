import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict, deque
import joblib
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings("ignore")

# =====================
# 1. Load models
# =====================
rf_model = joblib.load("random_forest.pkl")
encoder_model = load_model("encoder_model.h5")

scaler = joblib.load("scaler.pkl")  # MinMaxScaler
encoders = joblib.load("onehot_encoder.pkl")  # dict: {'protocol_type': LabelEncoder(), ...}

categorical_cols = ["protocol_type", "service", "flag"]

feature_columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

# =====================
# 2. Flow tracker
# =====================
flows = defaultdict(lambda: deque(maxlen=1000))

# =====================
# 3. Compute flow features
# =====================
def compute_flow_features(flow_packets):
    if not flow_packets:
        return None

    pkt0 = flow_packets[0]
    proto = "tcp" if pkt0.haslayer(TCP) else "udp" if pkt0.haslayer(UDP) else "icmp"
    
    src_bytes = sum(len(pkt[IP].payload) for pkt in flow_packets if pkt.haslayer(IP))
    dst_bytes = src_bytes  # placeholder, can be improved if we track src/dst separately

    features = {
        "duration": 0,
        "protocol_type": proto,
        "service": "http",
        "flag": "SF",
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
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
# 4. Preprocess features
# =====================
def preprocess_features(df):
    df_cat = df[categorical_cols]
    df_num = df.drop(columns=categorical_cols)

    # One-hot encode categorical columns
    X_cat = encoders.transform(df_cat)
    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()

    # Combine numerical and categorical
    X_all = np.hstack([df_num.values, X_cat])

    # Scale first (125 features expected by scaler)
    X_scaled = scaler.transform(X_all)

    # Pad to 128 features for NDAE
    expected_dim = 128
    if X_scaled.shape[1] < expected_dim:
        padding = np.zeros((X_scaled.shape[0], expected_dim - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, padding])

    return X_scaled.astype(np.float32)

# =====================
# 5. Predict callback
# =====================
attack_map = ['normal','dos','probe','r2l','u2r']

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
        X_scaled = preprocess_features(df)
        encoded = encoder_model.predict(X_scaled, verbose=0)
        y_pred_enc = rf_model.predict(encoded)
        y_pred = attack_map[y_pred_enc[0]]

        print(f"[PREDICTION] Flow {flow_key} => {y_pred}")

    except Exception as e:
        print(f"[ERROR] {e}")

# =====================
# 6. Start sniffing
# =====================
print("[INFO] Starting real-time KDD packet sniffing...")
sniff(iface="Wi-Fi", prn=predict_packet, store=False)
