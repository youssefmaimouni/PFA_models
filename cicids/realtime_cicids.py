#!/usr/bin/env python3
# realtime_cicids.py
# Sniffer -> Flows -> Features -> Preprocess -> Predict (SAE/LSTM/BiLSTM)
# Usage example:
# python realtime_cicids.py --iface "Wi-Fi" --scaler scaler.pkl --model best_sae_model.h5 --le label_encoder.pkl --out preds.json

import argparse
import time
import threading
import queue
from dataclasses import dataclass
import os
import sys
import signal
import json

import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP
from joblib import load as joblib_load

try:
    import tensorflow as tf
except Exception:
    tf = None

from colorama import Fore, Style, init as colorama_init
import warnings
warnings.filterwarnings("ignore")

colorama_init(autoreset=True)

# -----------------------
# Config
# -----------------------
BATCH_MAX_FLOWS   = 16
BATCH_MAX_SECONDS = 2.0
FLOW_IDLE_TIMEOUT = 10.0
MAX_FLOW_AGE      = 60.0

# -----------------------
# CICIDS Features (ordered)
# -----------------------
CICIDS_FEATURES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets','Total Length of Bwd Packets','Fwd Packet Length Max',
    'Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std','Bwd Packet Length Max',
    'Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s',
    'Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean',
    'Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std',
    'Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Fwd URG Flags','Fwd Header Length','Bwd Header Length',
    'Fwd Packets/s','Bwd Packets/s','Min Packet Length','Max Packet Length','Packet Length Mean',
    'Packet Length Std','Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count',
    'PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Down/Up Ratio',
    'Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size','Fwd Header Length.1',
    'Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes',
    'Init_Win_bytes_forward','Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward',
    'Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min'
]

# -----------------------
# Flow Aggregation
# -----------------------
@dataclass(frozen=True)
class FlowKey:
    src: str
    sport: int
    dst: str
    dport: int
    proto: str

class FlowState:
    __slots__ = ("first_ts","last_ts","pkt_cnt","byte_cnt","lens","iat","tcp_flags",
                 "src","dst","sport","dport","proto")

    def __init__(self, key: FlowKey, ts: float, pkt_len: int, flags: int|None):
        self.first_ts = ts
        self.last_ts  = ts
        self.pkt_cnt  = 1
        self.byte_cnt = pkt_len
        self.lens     = [pkt_len]
        self.iat      = []
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

    def to_feature_row(self):
        """
        Return a dict with CICIDS_FEATURES keys + identity columns src,dst,sport,dport,proto.
        Many features are approximated / defaulted because we build features from packets on-the-fly.
        """
        duration = max(self.last_ts - self.first_ts, 0.0001)
        mean_len = float(np.mean(self.lens)) if self.lens else 0.0

        # default zeros for all features then update partials we can compute
        row = {f: 0 for f in CICIDS_FEATURES}
        row.update({
            'Destination Port': int(self.dport),
            'Flow Duration': float(duration),
            'Total Fwd Packets': int(self.pkt_cnt),
            'Total Backward Packets': int(self.pkt_cnt),
            'Total Length of Fwd Packets': int(self.byte_cnt),
            'Total Length of Bwd Packets': int(self.byte_cnt),
            'Fwd Packet Length Mean': float(mean_len),
            'Bwd Packet Length Mean': float(mean_len),
            'Fwd PSH Flags': int(self.tcp_flags['psh']),
            'Fwd URG Flags': int(self.tcp_flags['urg']),
            'SYN Flag Count': int(self.tcp_flags['syn']),
            'ACK Flag Count': int(self.tcp_flags['ack']),
            'FIN Flag Count': int(self.tcp_flags['fin']),
            'RST Flag Count': int(self.tcp_flags['rst']),
        })

        # identity columns for printing
        row['src'] = self.src
        row['dst'] = self.dst
        row['sport'] = int(self.sport)
        row['dport'] = int(self.dport)
        # include proto so view contains it
        row['proto'] = self.proto

        return row

class FlowTable:
    def __init__(self):
        self._flows = {}
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

# -----------------------
# Preprocessor
# -----------------------
class CICIDSPreprocessor:
    def __init__(self, scaler_path):
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler at {scaler_path}")
        self.scaler = joblib_load(scaler_path)
        self.features = CICIDS_FEATURES

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # ensure features exist
        df = df.copy()
        for col in self.features:
            if col not in df.columns:
                df[col] = 0
        # scaler expects numeric array in the same order as features
        X = self.scaler.transform(df[self.features].values)
        return X

# -----------------------
# Predictor
# -----------------------
class CICIDSPredictor:
    def __init__(self, model_path, label_encoder_path):
        if tf is None:
            raise RuntimeError("TensorFlow not installed")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model at {model_path}")
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Missing label encoder at {label_encoder_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.le = joblib_load(label_encoder_path)

    def predict(self, X: np.ndarray, time_steps=1):
        """
        X: (N, D) numeric features
        Returns: list of string labels of length N
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D (N, D)")

        # Detect model expected input shape robustly (works if .inputs shapes return TensorShape OR tuples)
        model_inputs = getattr(self.model, "inputs", None)
        X_in = X
        if model_inputs:
            try:
                # obtain first input's shape - robustly handle TensorShape or tuple/list
                shape_obj = model_inputs[0].shape
                if hasattr(shape_obj, "as_list"):
                    ref_shape = shape_obj.as_list()  # list like [None, timesteps, features] or [None, features]
                else:
                    # shape_obj could be a tuple; convert to list
                    ref_shape = list(shape_obj)
            except Exception:
                ref_shape = None

            # If model wants sequences (3D) and features dimension matches last dim, add axis
            if isinstance(ref_shape, list) and len(ref_shape) == 3:
                # common forms: [None, timesteps, features] or [None, None, features]
                # if timesteps is 1 or None -> we can reshape to (N, timesteps, D) using timesteps=1
                timesteps = ref_shape[1]
                features_expected = ref_shape[2]
                # if last dim matches or is None, reshape
                if features_expected in (None, X.shape[1]) or features_expected == X.shape[1]:
                    # reshape to (N, timesteps, features)
                    # choose timesteps=1 if model expects None or 1
                    t = 1 if (timesteps is None or timesteps == 1) else timesteps
                    X_in = X.reshape((X.shape[0], t, X.shape[1]))
            else:
                # keep X as-is (2D) for dense models
                X_in = X

        # final prediction
        probs = self.model.predict(X_in, verbose=0)
        if probs.ndim == 2:
            idx = np.argmax(probs, axis=1)
        elif probs.ndim == 1:
            idx = (probs > 0.5).astype(int)
        else:
            idx = np.argmax(probs.reshape((probs.shape[0], -1)), axis=1)

        return self.le.inverse_transform(idx)

# -----------------------
# Color map for classes
# -----------------------
CLASS_COLORAMA_MAP = {
    "BENIGN": Fore.GREEN,
    "Bot": Fore.MAGENTA,
    "DDoS": Fore.RED,
    "DoS GoldenEye": Fore.YELLOW,
    "DoS Hulk": Fore.YELLOW,
    "DoS Slowhttptest": Fore.YELLOW,
    "DoS slowloris": Fore.YELLOW,
    "FTP-Patator": Fore.BLUE,
    "Heartbleed": Fore.RED + Style.BRIGHT,
    "Infiltration": Fore.RED + Style.DIM,
    "PortScan": Fore.CYAN,
    "SSH-Patator": Fore.CYAN + Style.BRIGHT,
    "Web Attack": Fore.MAGENTA + Style.BRIGHT
}

# -----------------------
# Threads
# -----------------------
def sniffer_thread(iface, flow_table, stop_event):
    def onpkt(pkt):
        res = scapy_to_flowkey(pkt)
        if res is None:
            return
        key, (ts, pkt_len, flags) = res
        flow_table.upsert(key, ts, pkt_len, flags)

    sniff_kwargs = dict(store=False, prn=onpkt, iface=iface)
    while not stop_event.is_set():
        try:
            sniff(timeout=1, **sniff_kwargs)
        except Exception as e:
            print(f"[sniffer] {e}", file=sys.stderr)
            time.sleep(0.5)

def collector_thread(flow_table, batch_q: queue.Queue, stop_event):
    last_emit = time.time()
    batch = []
    while not stop_event.is_set():
        time.sleep(0.2)
        frozen = flow_table.collect_inactive(time.time(), idle_timeout=1.0)
        for fs in frozen:
            batch.append(fs.to_feature_row())
        if batch and (len(batch) >= BATCH_MAX_FLOWS or (time.time() - last_emit) >= 1.0):
            # create dataframe with identity columns present
            df = pd.DataFrame(batch)
            batch_q.put(df)
            batch = []
            last_emit = time.time()
    # flush remaining flows on shutdown
    rem = flow_table.flush_all()
    if rem:
        df = pd.DataFrame([fs.to_feature_row() for fs in rem])
        batch_q.put(df)

def inference_thread(batch_q: queue.Queue, preproc: CICIDSPreprocessor, predictor: CICIDSPredictor,
                     stop_event: threading.Event, out_file: str|None = None):
    # optional output file (JSON lines)
    if out_file:
        out_fh = open(out_file, "a", encoding="utf-8")
    else:
        out_fh = None

    while not stop_event.is_set():
        try:
            df = batch_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if df is None or df.empty:
            continue

        # keep identity columns
        ident_cols = ["src", "dst", "sport", "dport", "proto"]
        view = df[ident_cols].copy() if set(ident_cols).issubset(df.columns) else None

        try:
            X = preproc.transform(df)
        except Exception as e:
            print(f"[preproc] error: {e}", file=sys.stderr)
            continue

        try:
            labels = predictor.predict(X)
        except Exception as e:
            print(f"[predict] error: {e}", file=sys.stderr)
            continue

        # print with color and optionally log
        for i, lbl in enumerate(labels):
            color = CLASS_COLORAMA_MAP.get(lbl, Fore.WHITE)
            if view is not None:
                row = view.iloc[i]
                out_str = f"{color}[{lbl}]{Style.RESET_ALL} {row['src']}:{row['sport']} -> {row['dst']}:{row['dport']} | proto={row['proto']}"
                print(out_str)
                if out_fh:
                    rec = {
                        "ts": time.time(),
                        "label": lbl,
                        "src": row['src'],
                        "sport": int(row['sport']),
                        "dst": row['dst'],
                        "dport": int(row['dport']),
                        "proto": row['proto']
                    }
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_fh.flush()
            else:
                print(f"{color}[{lbl}]{Style.RESET_ALL}")
                if out_fh:
                    out_fh.write(json.dumps({"ts": time.time(), "label": lbl}) + "\n")
                    out_fh.flush()

    if out_fh:
        out_fh.close()

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Realtime CICIDS inference from live flows")
    parser.add_argument("--iface", required=True, help="Network interface to sniff (e.g. 'Wi-Fi')")
    parser.add_argument("--scaler", default="scaler.pkl", help="joblib StandardScaler fitted on CICIDS_FEATURES")
    parser.add_argument("--model", default="best_sae_model.h5", help="Keras model file (h5 or SavedModel dir)")
    parser.add_argument("--le", default="label_encoder.pkl", help="joblib LabelEncoder mapping indices->class names")
    parser.add_argument("--out", default=None, help="optional JSON-lines output file to append predictions")
    args = parser.parse_args()

    try:
        preproc = CICIDSPreprocessor(args.scaler)
        predictor = CICIDSPredictor(args.model, args.le)
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

    threads = [
        threading.Thread(target=sniffer_thread, args=(args.iface, flow_table, stop_event), daemon=True),
        threading.Thread(target=collector_thread, args=(flow_table, batch_q, stop_event), daemon=True),
        threading.Thread(target=inference_thread, args=(batch_q, preproc, predictor, stop_event, args.out), daemon=True)
    ]
    for t in threads:
        t.start()

    print(f"[main] Running on iface={args.iface}, press Ctrl+C to stop")
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        stop_event.set()
        # give threads a moment to flush
        time.sleep(0.5)

if __name__ == "__main__":
    main()
