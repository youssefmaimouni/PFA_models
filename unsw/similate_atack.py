# simulate_unsw_attack.py
import socket
import threading
import time

# Target (localhost for testing)
TARGET_IP = "127.0.0.1"
TARGET_PORT = 80  # HTTP port, matches your UNSW model mapping
NUM_FLOWS = 5
PACKETS_PER_FLOW = 20
INTERVAL = 0.05  # seconds between packets

def simulate_tcp_syn_flood(flow_id):
    """Simulate a SYN flood-like pattern to match UNSW features."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        for i in range(PACKETS_PER_FLOW):
            try:
                sock.connect((TARGET_IP, TARGET_PORT))
            except:
                pass  # ignore connection errors
            time.sleep(INTERVAL)
        sock.close()
        print(f"[Attack] Flow {flow_id} finished")
    except Exception as e:
        print(f"[Attack] Flow {flow_id} error: {e}")

threads = []
for f in range(NUM_FLOWS):
    t = threading.Thread(target=simulate_tcp_syn_flood, args=(f,))
    t.start()
    threads.append(t)
    time.sleep(0.1)  # stagger flows slightly

for t in threads:
    t.join()

print("[Attack] All flows finished")
