import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 1. Charger scaler + encoder + modèle RF
# =========================
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
rf = joblib.load("rf_model.pkl")

# Labels des classes (à adapter si ton RF a un autre ordre)
labels = ['Normal', 'Reconnaissance', 'Backdoor', 'DoS', 'Exploits', 
          'Analysis', 'Fuzzers', 'Worms', 'Shellcode', 'Generic']

# =========================
# 2. Fonction de prétraitement
# =========================
def preprocess(df):
    cat_col = ['proto', 'state', 'service']
    
    # 🔹 Récupérer l'ordre des colonnes numériques exact utilisé à l'entraînement
    num_col = list(scaler.feature_names_in_)
    
    # Normalisation avec les colonnes dans le bon ordre
    df[num_col] = scaler.transform(df[num_col])
    
    # Encodage one-hot des colonnes catégorielles (encoder garde déjà l'ordre)
    X = np.array(encoder.transform(df))
    return X


# =========================
# 3. Exemple de ligne (remplis avec tes valeurs)
# =========================
sample_dict = {
    "proto": "tcp",
    "state": "FIN",
    "dur": 0.2,
    "sbytes": 350,
    "dbytes": 1200,
    "sttl": 60,
    "dttl": 60,
    "sloss": 0,
    "dloss": 0,
    "service": "http",
    "Sload": 10.5,
    "Dload": 25.1,
    "Spkts": 8,
    "Dpkts": 10,
    "swin": 20000,
    "dwin": 15000,
    "stcpb": 1000,
    "dtcpb": 2000,
    "smeansz": 44,
    "dmeansz": 120,
    "trans_depth": 1,
    "res_bdy_len": 0,
    "Sjit": 0.0,
    "Djit": 0.0,
    "Stime": 123456,
    "Ltime": 123466,
    "Sintpkt": 0.01,
    "Dintpkt": 0.02,
    "tcprtt": 0.05,
    "synack": 0.02,
    "ackdat": 0.03,
    "is_sm_ips_ports": 0,
    "ct_state_ttl": 12,
    "ct_flw_http_mthd": 1,
    "is_ftp_login": 0,
    "ct_ftp_cmd": 0,
    "ct_srv_src": 3,
    "ct_srv_dst": 1,
    "ct_dst_ltm": 5,
    "ct_src_ ltm": 1,   # ⚠️ attention il y a un espace dans ton dataset
    "ct_src_dport_ltm": 2,
    "ct_dst_sport_ltm": 1,
    "ct_dst_src_ltm": 4,
    "attack_cat": "Normal"  # pas utilisé pour la prédiction
}

sample_df = pd.DataFrame([sample_dict])

# =========================
# 4. Prétraitement + Prédiction RF
# =========================
X_sample = preprocess(sample_df)

pred_rf = labels[rf.predict(X_sample)[0]]
print("🔹 Prediction RF :", pred_rf)
