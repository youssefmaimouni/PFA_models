# realtime_ids.py
# Sniffer -> Flows -> Features -> Preprocess (encoder/scaler) -> Predict (local or Azure)

import argparse
import time
import threading
import queue
from dataclasses import dataclass
import os
import sys
import signal

import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP
from joblib import load as joblib_load
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
except Exception:
    tf = None

import requests
import warnings
warnings.filterwarnings("ignore")

from colorama import Fore, Style, init
init(autoreset=True)

# Define label -> color mapping
LABEL_COLORS = {
    "Normal": Fore.GREEN,
    "Reconnaissance": Fore.YELLOW,
    "Backdoor": Fore.RED,
    "DoS": Fore.RED,
    "Exploits": Fore.MAGENTA,
    "Analysis": Fore.CYAN,
    "Fuzzers": Fore.BLUE,
    "Worms": Fore.LIGHTRED_EX,
    "Shellcode": Fore.LIGHTMAGENTA_EX,
    "Generic": Fore.LIGHTCYAN_EX,
    "Unknown": Fore.WHITE
}

# -----------------------
# Config
# -----------------------
BATCH_MAX_FLOWS   = 128
BATCH_MAX_SECONDS = 2.0
FLOW_IDLE_TIMEOUT = 10.0
MAX_FLOW_AGE      = 60.0

MODE_LOCAL_SKLEARN = "local_sklearn"
MODE_LOCAL_KERAS   = "local_keras"
MODE_AZURE         = "azure"

def now(): return time.time()

# =======================
# Flow aggregation
# =======================
@dataclass(frozen=True)
class FlowKey:
    src: str
    sport: int
    dst: str
    dport: int
    proto: str  # "TCP" / "UDP" / "OTHER"

class FlowState:
    __slots__ = (
        "first_ts","last_ts","pkt_cnt","byte_cnt","lens","iat",
        "tcp_flags","src","dst","sport","dport","proto"
    )

    def __init__(self, key: FlowKey, ts: float, pkt_len: int, flags: int|None):
        self.first_ts = ts
        self.last_ts  = ts
        self.pkt_cnt  = 1
        self.byte_cnt = pkt_len
        self.lens     = [pkt_len]
        self.iat      = []            # inter-arrival times
        self.tcp_flags = {"syn":0,"ack":0,"fin":0,"rst":0,"psh":0,"urg":0}
        if flags is not None:
            self._accum_tcp_flags(flags)
        self.src, self.dst = key.src, key.dst
        self.sport, self.dport = key.sport, key.dport
        self.proto = key.proto

    def _accum_tcp_flags(self, flags: int):
        if flags & 0x02: self.tcp_flags["syn"] += 1
        if flags & 0x10: self.tcp_flags["ack"] += 1
        if flags & 0x01: self.tcp_flags["fin"] += 1
        if flags & 0x04: self.tcp_flags["rst"] += 1
        if flags & 0x08: self.tcp_flags["psh"] += 1
        if flags & 0x20: self.tcp_flags["urg"] += 1

    def add_packet(self, ts: float, pkt_len: int, flags: int|None):
        self.iat.append(max(ts - self.last_ts, 0.0))
        self.last_ts = ts
        self.pkt_cnt += 1
        self.byte_cnt += pkt_len
        self.lens.append(pkt_len)
        if flags is not None:
            self._accum_tcp_flags(flags)

    def is_idle(self, ts: float, idle_timeout: float) -> bool:
        return (ts - self.last_ts) >= idle_timeout

    def age(self, ts: float) -> float:
        return ts - self.first_ts

    # ---- IMPORTANT ----
    # Emit EXACT UNSW-NB15 FEATURE NAMES (inputs only; no 'attack_cat')
    def to_feature_row(self):
        duration = max(self.last_ts - self.first_ts, 0.0001)
        mean_len = float(np.mean(self.lens)) if self.lens else 0.0

        # ---- State ----
        state = "CON" if self.pkt_cnt > 10 else "FIN"
        allowed_states = ["CON", "FIN"]
        if state not in allowed_states:
            state = "FIN"

        # ---- Service ----
        if self.dport in [80, 8080]:
            service = "http"
        elif self.dport == 443:
            service = "http"  # map https -> http (matches training)
        elif self.dport == 53:
            service = "dns"
        elif self.dport == 21:
            service = "ftp"
        else:
            service = "http"  # unknown services mapped to http

        # ---- Protocol ----
        proto_str = self.proto.lower() if self.proto else "other"
        if proto_str not in ("tcp", "udp", "other"):
            proto_str = "other"

        row = {
            "proto": proto_str,
            "state": state,
            "service": service,
            "dur": duration,
            "sbytes": self.byte_cnt,
            "dbytes": self.byte_cnt,
            "sttl": 64,
            "dttl": 64,
            "sloss": 0,
            "dloss": 0,
            "Sload": self.byte_cnt / duration,
            "Dload": self.byte_cnt / duration,
            "Spkts": self.pkt_cnt,
            "Dpkts": self.pkt_cnt,
            "swin": 255,
            "dwin": 255,
            "stcpb": sum(self.lens),
            "dtcpb": sum(self.lens),
            "smeansz": int(round(mean_len)) if mean_len else 0,
            "dmeansz": int(round(mean_len)) if mean_len else 0,
            "trans_depth": 1,
            "res_bdy_len": 0,
            "Sjit": float(np.std(self.iat)) if len(self.iat) > 1 else 0.0,
            "Djit": float(np.std(self.iat)) if len(self.iat) > 1 else 0.0,
            "Stime": int(self.first_ts),
            "Ltime": int(self.last_ts),
            "Sintpkt": float(np.mean(self.iat)) if self.iat else 0.0,
            "Dintpkt": float(np.mean(self.iat)) if self.iat else 0.0,
            "tcprtt": 0.0,
            "synack": float(self.tcp_flags["syn"]),
            "ackdat": float(self.tcp_flags["ack"]),
            # ... keep the rest as before ...
            "src": self.src,
            "dst": self.dst,
            "sport": self.sport,
            "dport": self.dport,
        }
        return row


class FlowTable:
    def __init__(self):
        self._flows: dict[FlowKey, FlowState] = {}
        self._lock = threading.Lock()

    def upsert(self, key: FlowKey, ts: float, pkt_len: int, flags: int|None):
        with self._lock:
            fs = self._flows.get(key)
            if fs is None:
                self._flows[key] = FlowState(key, ts, pkt_len, flags)
            else:
                fs.add_packet(ts, pkt_len, flags)

    def collect_inactive(self, ts: float, idle_timeout=FLOW_IDLE_TIMEOUT, max_age=MAX_FLOW_AGE):
        frozen = []
        with self._lock:
            to_del = []
            for k, fs in self._flows.items():
                if fs.is_idle(ts, idle_timeout) or fs.age(ts) >= max_age:
                    frozen.append(fs)
                    to_del.append(k)
            for k in to_del:
                del self._flows[k]
        return frozen

    def flush_all(self):
        with self._lock:
            frozen = list(self._flows.values())
            self._flows.clear()
        return frozen

def scapy_to_flowkey(pkt):
    if IP not in pkt:
        return None
    ip = pkt[IP]
    proto = "OTHER"
    sport = 0
    dport = 0
    flags = None
    if TCP in pkt:
        proto = "TCP"
        sport = int(pkt[TCP].sport)
        dport = int(pkt[TCP].dport)
        flags = int(pkt[TCP].flags)
    elif UDP in pkt:
        proto = "UDP"
        sport = int(pkt[UDP].sport)
        dport = int(pkt[UDP].dport)
    key = FlowKey(ip.src, sport, ip.dst, dport, proto)
    pkt_len = len(pkt)
    ts = float(pkt.time)
    return key, (ts, pkt_len, flags)

# =======================
# Preprocessor
# =======================
class Preprocessor:
    """
    Loads:
      - encoder.pkl: the ColumnTransformer you saved (OneHot on ['proto','service','state'], remainder='passthrough')
      - scaler.pkl:  StandardScaler fitted on the numeric columns
    It scales numerics (same order as during fit) and then calls encoder.transform(df)
    """
    # Exact UNSW-NB15 input schema (43 inputs)
    CAT_COLS = ["proto", "service", "state"]
    ALL_INPUT_COLS = [
        "proto","state","dur","sbytes","dbytes","sttl","dttl","sloss","dloss","service",
        "Sload","Dload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb","smeansz","dmeansz",
        "trans_depth","res_bdy_len","Sjit","Djit","Stime","Ltime","Sintpkt","Dintpkt",
        "tcprtt","synack","ackdat","is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd",
        "is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ ltm",
        "ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm"
    ]

    def __init__(self, encoder_path: str, scaler_path: str):
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Missing encoder at {encoder_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler at {scaler_path}")

        self.encoder = joblib_load(encoder_path)  # ColumnTransformer (as saved in training)
        self.scaler  = joblib_load(scaler_path)   # StandardScaler on numeric cols

        # Use the trained numeric column order from scaler
        self.numeric_cols = list(getattr(self.scaler, "feature_names_in_", []))
        if not self.numeric_cols:
            raise RuntimeError("scaler.pkl has no feature_names_in_. Please save a fitted scaler.")

        # ColumnTransformer should know feature names too
        if not hasattr(self.encoder, "transform"):
            raise RuntimeError("encoder.pkl does not look like a ColumnTransformer.")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()

        # Ensure all expected input columns exist; add missing with neutral defaults
        for col in self.ALL_INPUT_COLS:
            if col not in df.columns:
                # sensible defaults
                df[col] = 0 if col not in self.CAT_COLS else "UNKNOWN"

        # Drop any extra columns the model didn't see (except identities we keep for printing elsewhere)
        df = df[self.ALL_INPUT_COLS + ["src","dst","sport","dport"] if set(["src","dst","sport","dport"]).issubset(df.columns) else self.ALL_INPUT_COLS]

        # Scale numeric columns in the SAME ORDER as during fit
        for c in self.numeric_cols:
            if c not in df.columns:
                df[c] = 0
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])

        # The ColumnTransformer expects full frame with cat + (already scaled) numerics
        X = self.encoder.transform(df[self.ALL_INPUT_COLS])
        # If the encoder returns sparse, make it dense (sklearn >=1.4 often returns ndarray)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return X

# =======================
# Predictor
# =======================
class Predictor:
    def __init__(self, mode=MODE_LOCAL_SKLEARN, model_path=None, keras_path=None,
                 azure_url=None, azure_key=None, timeout=3.0):
        self.mode = mode
        self.timeout = timeout
        self.session = requests.Session()
        self.sklearn_model = None
        self.keras_model = None
        self.azure_url = azure_url
        self.azure_key = azure_key

        if self.mode == MODE_LOCAL_SKLEARN:
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing sklearn model at {model_path}")
            self.sklearn_model = joblib_load(model_path)
        elif self.mode == MODE_LOCAL_KERAS:
            if tf is None:
                raise RuntimeError("TensorFlow not installed")
            if not keras_path or not os.path.exists(keras_path):
                raise FileNotFoundError(f"Missing Keras model at {keras_path}")
            self.keras_model = tf.keras.models.load_model(keras_path)
        elif self.mode == MODE_AZURE:
            if not azure_url:
                raise ValueError("Azure URL required")

    def predict(self, X: np.ndarray):
        if self.mode == MODE_LOCAL_SKLEARN:
            return self.sklearn_model.predict(X).tolist()
        if self.mode == MODE_LOCAL_KERAS:
            probs = self.keras_model.predict(X, verbose=0)
            return (np.argmax(probs, axis=1) if probs.ndim == 2 else (probs > 0.5).astype(int).ravel()).tolist()
        if self.mode == MODE_AZURE:
            headers = {"Content-Type": "application/json"}
            if self.azure_key:
                headers["Authorization"] = f"Bearer {self.azure_key}"
            resp = self.session.post(self.azure_url, headers=headers, json={"data": X.tolist()}, timeout=self.timeout)
            resp.raise_for_status()
            out = resp.json()
            return out.get("prediction", out)
        raise RuntimeError("Invalid prediction mode")

# =======================
# Threads
# =======================
def sniffer_thread(iface: str, flow_table: FlowTable, stop_event: threading.Event):
    def onpkt(pkt):
        res = scapy_to_flowkey(pkt)
        if res is None: return
        key, (ts, pkt_len, flags) = res
        flow_table.upsert(key, ts, pkt_len, flags)

    sniff_kwargs = dict(store=False, prn=onpkt, iface=iface)
    while not stop_event.is_set():
        try:
            sniff(timeout=1, **sniff_kwargs)
        except Exception as e:
            print(f"[sniffer] error: {e}", file=sys.stderr)
            time.sleep(0.5)

def collector_thread(flow_table: FlowTable, batch_q: queue.Queue, stop_event: threading.Event):
    last_emit = now()
    batch = []
    while not stop_event.is_set():
        time.sleep(0.2)
        frozen = flow_table.collect_inactive(now(), idle_timeout=1.0)  # plus réactif pour DoS
        for fs in frozen:
            batch.append(fs.to_feature_row())
        if batch and (len(batch) >= BATCH_MAX_FLOWS or (now() - last_emit) >= 1.0):
            batch_q.put(pd.DataFrame(batch))
            batch = []
            last_emit = now()
    rem = flow_table.flush_all()
    if rem:
        batch_q.put(pd.DataFrame([fs.to_feature_row() for fs in rem]))

# Définir les labels globaux
LABELS = ['Normal', 'Reconnaissance', 'Backdoor', 'DoS', 'Exploits',
          'Analysis', 'Fuzzers', 'Worms', 'Shellcode', 'Generic']

def inference_thread(batch_q: queue.Queue, preproc: Preprocessor, predictor: Predictor,
                     stop_event: threading.Event, print_labels=True):
    while not stop_event.is_set():
        try:
            df = batch_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if df.empty:
            continue

        # Keep identity columns for printing
        ident_cols = ["src","dst","sport","dport","proto"]
        view = df[ident_cols].copy() if set(ident_cols).issubset(df.columns) else None

        # Preprocess
        try:
            X = preproc.transform(df)
        except Exception as e:
            print(f"[preprocess] error: {e}", file=sys.stderr)
            continue

        # Predict
        try:
            y = predictor.predict(X)  # retourne des indices 0-9
            # Mapper les indices vers les labels
            y_labels = [LABELS[i] if i < len(LABELS) else "Unknown" for i in y]
        except Exception as e:
            print(f"[predict] error: {e}", file=sys.stderr)
            continue

        if print_labels and view is not None:
            out = view.copy()
            out["pred_label"] = y_labels

            # Colorize output
            for _, row in out.head(20).iterrows():
                label = row["pred_label"]
                color = LABEL_COLORS.get(label, Fore.WHITE)
                print(f"[predict] {row['src']:>15}:{row['sport']:<5} -> {row['dst']:>15}:{row['dport']:<5} "
                    f"[{row['proto']}]  {color}{label}{Style.RESET_ALL}")
        else:
            print(f"[predict] batch size={len(y_labels)}")

# =======================
# Main
# =======================
def main():
    parser = argparse.ArgumentParser(description="Realtime IDS streaming (UNSW-NB15 features)")
    parser.add_argument("--iface", required=True, help="Network interface to sniff")
    parser.add_argument("--encoder", default="encoder.pkl", help="ColumnTransformer (OneHot on cats)")
    parser.add_argument("--scaler",  default="scaler.pkl",  help="StandardScaler on numeric features")
    parser.add_argument("--mode", choices=[MODE_LOCAL_SKLEARN, MODE_LOCAL_KERAS, MODE_AZURE],
                        default=MODE_LOCAL_SKLEARN)
    parser.add_argument("--sk_model", default="rf_model.pkl")
    parser.add_argument("--keras_model", default="cnn_lstm_model.h5")
    parser.add_argument("--azure_url", default=None)
    parser.add_argument("--azure_key", default=None)
    args = parser.parse_args()

    try:
        preproc = Preprocessor(args.encoder, args.scaler)
        predictor = Predictor(args.mode, args.sk_model, args.keras_model, args.azure_url, args.azure_key)
    except Exception as e:
        print(f"[init] {e}", file=sys.stderr)
        sys.exit(1)

    flow_table = FlowTable()
    batch_q = queue.Queue()
    stop_event = threading.Event()

    def handle_sig(sig, frame):
        print("\n[main] stopping...")
        stop_event.set()
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    t_sniff = threading.Thread(target=sniffer_thread, args=(args.iface, flow_table, stop_event), daemon=True)
    t_coll  = threading.Thread(target=collector_thread, args=(flow_table, batch_q, stop_event), daemon=True)
    t_infer = threading.Thread(target=inference_thread, args=(batch_q, preproc, predictor, stop_event), daemon=True)

    t_sniff.start(); t_coll.start(); t_infer.start()
    print(f"[main] running on iface={args.iface} mode={args.mode}")
    print("[main] press Ctrl+C to stop")

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        stop_event.set()

if __name__ == "__main__":
    main()
