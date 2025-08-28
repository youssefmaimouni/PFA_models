import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict, deque
import joblib
import warnings
from colorama import Fore, Style, init  # ✅ add colorama

warnings.filterwarnings("ignore")

# =====================
# 1. Init colorama
# =====================
init(autoreset=True)
LABEL_COLORS = {
    "normal": Fore.GREEN,
    "dos": Fore.RED,
    "probe": Fore.YELLOW,
    "r2l": Fore.MAGENTA,
    "u2r": Fore.CYAN,
    "unknown": Fore.WHITE
}

# =====================
# 2. Load Random Forest & Preprocessing
# =====================
rf_model = joblib.load("rf_model_rfe.pkl")
rfe_selector = joblib.load("rfe_selector.pkl")
le_protocol = joblib.load("le_protocol.pkl")
le_service  = joblib.load("le_service.pkl")
le_flag     = joblib.load("le_flag.pkl")

categorical_cols = ["protocol_type", "service", "flag"]
attack_map = ['normal', 'dos', 'probe', 'r2l', 'u2r']

# =====================
# 3. Flow tracker
# =====================
flows = defaultdict(lambda: deque(maxlen=1000))

# =====================
# 4. Compute flow features (simplified for live capture)
# =====================
def compute_flow_features(flow_packets):
    if not flow_packets:
        return None

    pkt0 = flow_packets[0]
    proto = "tcp" if pkt0.haslayer(TCP) else "udp" if pkt0.haslayer(UDP) else "icmp"
    
    src_bytes = sum(len(pkt[IP].payload) for pkt in flow_packets if pkt.haslayer(IP))
    dst_bytes = src_bytes  # placeholder

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
# 5. Preprocess features for RF
# =====================
def preprocess_rf(df):
    df['protocol_type'] = le_protocol.transform(df['protocol_type'])
    df['service']       = le_service.transform(df['service'])
    df['flag']          = le_flag.transform(df['flag'])

    X_rfe = rfe_selector.transform(df)
    return X_rfe

# =====================
# 6. Predict callback
# =====================
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
        X_rfe = preprocess_rf(df)
        y_pred_enc = rf_model.predict(X_rfe)
        label = attack_map[y_pred_enc[0]]

        color = LABEL_COLORS.get(label.lower(), Fore.WHITE)
        print(f"Flow {flow_key} predicted attack category: {color}{label}{Style.RESET_ALL}")

    except Exception as e:
        print(f"[ERROR] {e}")

# =====================
# 7. Start sniffing
# =====================
print("[INFO] Starting real-time KDD packet sniffing...")
sniff(iface="Wi-Fi", prn=predict_packet, store=False)
