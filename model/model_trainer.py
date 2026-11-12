import joblib
import sys
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

PROXY_CONFIG = {
    "A00 (Toán - Lý - Hóa)": {"features": ["toan", "vat_li"], "target": "hoa_hoc"},
    "A01 (Toán - Lý - Anh)": {"features": ["toan", "vat_li"], "target": "ngoai_ngu"},
    "B00 (Toán - Hóa - Sinh)": {"features": ["toan", "hoa_hoc"], "target": "sinh_hoc"},
    "C00 (Văn - Sử - Địa)": {"features": ["ngu_van", "lich_su"], "target": "dia_li"},
    "D01 (Toán - Văn - Anh)": {"features": ["toan", "ngu_van"], "target": "ngoai_ngu"},
}

@st.cache_resource
def train_model(sub, ten_to_hop):
    features = PROXY_CONFIG[ten_to_hop]["features"]
    X = sub[features].values
    y = sub["target_24"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify= y, random_state= 42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    acc_test = accuracy_score(y_test, y_pred)
    auc_test = roc_auc_score(y_test, y_proba)
    
    stats = {
        "n": len(sub),
        "ty_le_dat": (sub["target_24"].mean()) * 100,
        "acc_test" : acc_test,
        "auc_test": auc_test,
        "features" : features,
        "phan_tram_2mon" : sub[features].sum(axis = 1).values
    }
    return model, stats

def du_doan(model, diem_2_mon):
    return model.predict_proba([diem_2_mon])[0,1]

def phan_tram(stats, tong_2_mon):
    return (stats["phan_tram_2mon"] < tong_2_mon).mean()*100

if __name__ == "__main__":
    from dataprocessor import DataProcessor
    FILE_PATH = r"D:\Project_THPTQG_2024\data\diem_thi_thpt_2024.csv"
    
    proc = DataProcessor(FILE_PATH)
    proc.load_and_clean()
    ten_to_hop = "A00 (Toán - Lý - Hóa)"
    sub = proc.get_combo_data(ten_to_hop)
    features = PROXY_CONFIG[ten_to_hop]["features"]
    
    model, stats = train_model(sub, ten_to_hop)
    print(f"Huan luyen thanh cong! Do chinh xac: {stats['acc_test'] * 100:.2f}%")
    
    diem_toan = 7.0
    diem_ly = 10.0
    xac_suat = du_doan(model, [diem_toan, diem_ly])
    pct = phan_tram(stats, diem_toan + diem_ly)
    print(f"Xac suat dat 24+ diem la: {xac_suat * 100:.2f}%")
    print(f"Vuot qua {pct:.2f}% thi sinh cung to hop")
    
    
    
