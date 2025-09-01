#!/usr/bin/env python3
import time
import threading
import pandas as pd
from collections import defaultdict, Counter
from scapy.all import sniff, IP, TCP, UDP, Raw

# --- Configuration ---
INTERFACE = "Wi-Fi"      # Replace with your Wi-Fi interface name
FLOW_TIMEOUT = 60         # seconds
EXPORT_INTERVAL = 60
CSV_FILENAME = "unsw_flows_full.csv"

# --- Global Flow storage ---
flows = defaultdict(lambda: {
    "proto": None, "state": None, "dur": 0,
    "sbytes": 0, "dbytes": 0,
    "sttl": [], "dttl": [],
    "sloss": 0, "dloss": 0,
    "service": None, "Sload": 0, "Dload": 0,
    "Spkts": 0, "Dpkts": 0,
    "swin": [], "dwin": [],
    "stcpb": 0, "dtcpb": 0,
    "smeansz": 0, "dmeansz": 0,
    "trans_depth": 0, "res_bdy_len": 0,
    "Sjit": [], "Djit": [],
    "Stime": None, "Ltime": None,
    "Sintpkt": [], "Dintpkt": [],
    "tcprtt": None, "synack": 0, "ackdat": 0,
    "is_sm_ips_ports": 0,
    "ct_state_ttl": Counter(),
    "ct_flw_http_mthd": Counter(),
    "is_ftp_login": 0, "ct_ftp_cmd": Counter(),
    "ct_srv_src": Counter(), "ct_srv_dst": Counter(),
    "ct_dst_ltm": Counter(), "ct_src_ltm": Counter(),
    "ct_src_dport_ltm": Counter(), "ct_dst_sport_ltm": Counter(),
    "ct_dst_src_ltm": Counter(),
    "prev_pkt_time_src": None,
    "prev_pkt_time_dst": None,
    "pkt_sizes_src": [],
    "pkt_sizes_dst": [],
    "syn_time": None,
    "synack_time": None
})

# --- Helper functions ---
def get_flow_key(pkt):
    if IP not in pkt:
        return None
    src = pkt[IP].src
    dst = pkt[IP].dst
    proto = pkt.getlayer(TCP) and "TCP" or (pkt.getlayer(UDP) and "UDP" or str(pkt.proto))
    sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
    dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
    return tuple(sorted([(src, sport), (dst, dport)]) + [proto])

def detect_service(pkt):
    if TCP in pkt and pkt[TCP].dport in [80, 8080]:
        return "HTTP"
    elif TCP in pkt and pkt[TCP].dport in [21]:
        return "FTP"
    elif UDP in pkt and pkt[UDP].dport in [53]:
        return "DNS"
    return None

def process_packet(pkt):
    key = get_flow_key(pkt)
    if key is None:
        return
    flow = flows[key]
    now = time.time()
    size = len(pkt)
    if flow["Stime"] is None:
        flow["Stime"] = now
        flow["proto"] = key[2]
        flow["service"] = detect_service(pkt)
    flow["Ltime"] = now
    flow["dur"] = flow["Ltime"] - flow["Stime"]

    src_ip = pkt[IP].src
    dst_ip = pkt[IP].dst

    # TTL and Window sizes
    if hasattr(pkt[IP], "ttl"):
        if src_ip == key[0][0]:
            flow["sttl"].append(pkt[IP].ttl)
            if TCP in pkt:
                flow["swin"].append(pkt[TCP].window)
        else:
            flow["dttl"].append(pkt[IP].ttl)
            if TCP in pkt:
                flow["dwin"].append(pkt[TCP].window)

    # Directional stats
    if src_ip == key[0][0]:
        flow["sbytes"] += size
        flow["Spkts"] += 1
        flow["pkt_sizes_src"].append(size)
        if flow["prev_pkt_time_src"]:
            flow["Sintpkt"].append(now - flow["prev_pkt_time_src"])
        flow["prev_pkt_time_src"] = now
        if TCP in pkt:
            flow["stcpb"] += len(pkt[TCP].payload)
            if pkt[TCP].flags & 0x02:  # SYN
                flow["syn_time"] = now
        if Raw in pkt and flow["service"]=="HTTP":
            payload = pkt[Raw].load.decode(errors="ignore")
            for m in ["GET", "POST", "PUT", "DELETE"]:
                if payload.startswith(m):
                    flow["ct_flw_http_mthd"][m] += 1
                    flow["trans_depth"] += 1
            flow["res_bdy_len"] += len(payload)
    else:
        flow["dbytes"] += size
        flow["Dpkts"] += 1
        flow["pkt_sizes_dst"].append(size)
        if flow["prev_pkt_time_dst"]:
            flow["Dintpkt"].append(now - flow["prev_pkt_time_dst"])
        flow["prev_pkt_time_dst"] = now
        if TCP in pkt:
            flow["dtcpb"] += len(pkt[TCP].payload)
            if pkt[TCP].flags & 0x12:  # SYN-ACK
                flow["synack"] += 1
                if flow["syn_time"]:
                    flow["tcprtt"] = now - flow["syn_time"]
            if pkt[TCP].flags & 0x10:  # ACK
                flow["ackdat"] += 1

    # Compute mean sizes
    flow["smeansz"] = sum(flow["pkt_sizes_src"]) / max(1, len(flow["pkt_sizes_src"]))
    flow["dmeansz"] = sum(flow["pkt_sizes_dst"]) / max(1, len(flow["pkt_sizes_dst"]))

# --- Periodic export ---
def export_flows():
    data = []
    for f in flows.values():
        row = f.copy()
        row["Sttl"] = sum(f["sttl"])/max(1,len(f["sttl"])) if f["sttl"] else 0
        row["Dttl"] = sum(f["dttl"])/max(1,len(f["dttl"])) if f["dttl"] else 0
        row["Swin"] = sum(f["swin"])/max(1,len(f["swin"])) if f["swin"] else 0
        row["Dwin"] = sum(f["dwin"])/max(1,len(f["dwin"])) if f["dwin"] else 0
        row["Sintpkt"] = sum(f["Sintpkt"])/max(1,len(f["Sintpkt"])) if f["Sintpkt"] else 0
        row["Dintpkt"] = sum(f["Dintpkt"])/max(1,len(f["Dintpkt"])) if f["Dintpkt"] else 0
        data.append(row)
    df = pd.DataFrame(data)
    df.to_csv(CSV_FILENAME, index=False)
    print(f"[INFO] Exported {len(data)} flows to {CSV_FILENAME}")

def periodic_export():
    while True:
        time.sleep(EXPORT_INTERVAL)
        export_flows()

def cleanup_flows():
    while True:
        time.sleep(FLOW_TIMEOUT)
        now = time.time()
        keys_to_remove = [k for k,f in flows.items() if f["Ltime"] and now - f["Ltime"]>FLOW_TIMEOUT]
        for k in keys_to_remove:
            del flows[k]

# --- Start background threads ---
threading.Thread(target=periodic_export, daemon=True).start()
threading.Thread(target=cleanup_flows, daemon=True).start()

# --- Start live capture ---
print(f"[INFO] Starting live capture on {INTERFACE} ... (Run as Administrator!)")
sniff(iface=INTERFACE, prn=process_packet, store=False)
