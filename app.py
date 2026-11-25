import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'model')))

from dataprocessor import DataProcessor
from model_trainer import train_model, PROXY_CONFIG, du_doan, phan_tram

SUBJECT_LABEL = {
                "toan": "Toán", "ngu_van": "Ngữ văn", "ngoai_ngu": "Ngoại ngữ",
                "vat_li": "Vật lí", "hoa_hoc": "Hóa học", "sinh_hoc": "Sinh học",
                "lich_su": "Lịch sử", "dia_li": "Địa lí", "gdcd": "GDCD",
            }

sns.set_theme(style= "whitegrid")


DATA_FILES = {
    2024: r"D:\Project_THPTQG_2024\data\diem_thi_thpt_2024.csv",
    2023: r"D:\Project_THPTQG_2024\data\diem_thi_thpt_2023.csv",
}

@st.cache_resource
def get_processor(filepath):
    proc = DataProcessor(filepath)
    proc.load_and_clean()
    return proc

st.set_page_config(page_title= "Tra cứu & Phân tích điểm thi THPTQG 2024", layout= "wide")
st.title("Làm việc với dữ liệu điểm thi THPTQG")


year = st.sidebar.selectbox("Chọn năm", list(DATA_FILES))
processor = get_processor(DATA_FILES[year])
df = processor.df

page = st.sidebar.radio(
    "Chọn chức năng", 
    ["Tra cứu điểm thi", "AI Cố vấn & Phân tích", "Dashboard phổ điểm"],    
)
st.sidebar.caption(f"Tổng số thí sinh trong dữ liệu **{len(df):,}**")

if page == "Tra cứu điểm thi":
    st.header("Tra cứu điểm thi theo số báo danh")
    sbd_input = st.text_input("Nhập số báo danh:", placeholder="Ví dụ: 01000001")
    
    if sbd_input:
        sbd_clean = sbd_input.strip()
        row = df[df["sbd"] == sbd_clean]
        
        if row.empty:
            st.error("Không tìm thấy số báo danh này trong dữ liệu")
        else:
            st.success(f"Tìm thấy thí sinh: SBD **{sbd_clean}**")
            st.header("Bảng điểm chi tiết")
            cols = st.columns(3)
            r = row.iloc[0]
            
            all_subj = list(SUBJECT_LABEL.keys())
            i = 0
            for subj in (all_subj):
                val = r[subj]
                display_val = f"{val:.2f}" if pd.notna(val) else "--"
                display_cols = cols[i % 3].metric(label = subj, value=display_val)
                i += 1
            st.divider()
            st.header("Tổng điểm theo tổ hợp xét tuyển")
            l = {}
            diem_cao = {}
            for combo_name, combo_cols in processor.khoi_thi.items():
                diem_thi = [r[c] for c in combo_cols]
                if all(pd.notna(x) for x in diem_thi):
                    diem_lam_tron = f"{sum(diem_thi):.2f}"
                    st.write(f"{combo_name}: {diem_lam_tron}")
                    l.update({combo_name : diem_lam_tron})
            st.divider()
            max_val = max(x for x in l.values())
            st.header("Tổ hợp có điểm cao nhất", divider="blue")
            for to_hop, diem_thi in l.items():
                if diem_thi == max_val:
                    st.write(f"{to_hop} : {diem_thi}")
                    diem_cao.update({to_hop : diem_thi})
            st.divider()
            st.header("Xếp hạng theo tổ hợp")
            for selected_combo, diem_so in diem_cao.items():
                sub_combo = processor.get_combo_data(selected_combo)
                tong = sub_combo.loc[sub_combo["sbd"] == sbd_clean, "tong_diem"].iloc[0]
                hang_ca_nuoc = (sub_combo["tong_diem"] > tong).sum() + 1

                sub_tinh = sub_combo[sub_combo["ten_tinh"] == r["ten_tinh"]]
                hang_tinh = (sub_tinh["tong_diem"] > tong).sum() + 1

                st.write(f"Tổ hợp {selected_combo} ({diem_so} điểm)")
                c1, c2 = st.columns(2)
                c1.metric("Hạng cả nước", f"{hang_ca_nuoc:,} / {len(sub_combo):,}")
                c2.metric(f"Hạng tại {r['ten_tinh']}", f"{hang_tinh:,} / {len(sub_tinh):,}")
            
elif page == "AI Cố vấn & Phân tích":
    st.header("AI Cố vấn: Dự đoán khả năng đạt mốc 24+ điểm")
    st.caption(
        "Nhập điểm 3 môn của tổ hợp bạn quan tâm, mô hình sẽ ước lượng xác suất "
        "đạt tổng ≥ 24 điểm và giải thích môn nào ảnh hưởng nhiều nhất."
        )
    to_hop = st.selectbox("Chọn tổ hợp:", list(PROXY_CONFIG.keys()))
    diem_2_mon = PROXY_CONFIG[to_hop]["features"]
    score = []
    for subj in diem_2_mon:
        v = st.number_input(
            label = SUBJECT_LABEL[subj],
            min_value = 0.0,
            max_value = 10.0,
            value = 5.0,
            step = 0.25,
            key = f"input_{subj}",
        )
        score.append(v)
    if st.button("Phân tích", type= "primary"):
        sub = processor.get_combo_data(to_hop)
        model, stats = train_model(sub, to_hop)
        prob = du_doan(model, score)
        tong_2_mon = sum(score)
        pct = phan_tram(stats, tong_2_mon)
        c1,c2,c3 = st.columns(3)
        c1.metric("Tổng 2 môn đã nhập", f"{tong_2_mon}")
        c2.metric("Xác suất đạt 24+", f"{prob * 100:.2f}%")
        c3.metric("Vượt qua bao nhiêu % thí sinh", f"{pct:.2f}%")
        st.divider()
        st.subheader("Mức độ ảnh hưởng của từng môn")
        ten_mon = [SUBJECT_LABEL[s] for s in diem_2_mon]
        gia_tri = model.coef_[0]
        fig, ax = plt.subplots(figsize = (6,4))
        sns.barplot(x = gia_tri, y = ten_mon, ax = ax)
        ax.set_xlabel("Hệ số hồi quy")
        st.pyplot(fig)
else:
    st.header("Dashboard phổ điểm thi THPTQG")
    subj = st.selectbox("Chọn môn để xem phổ điểm",SUBJECT_LABEL.keys(), format_func= lambda x : SUBJECT_LABEL[x])
    vals = df[subj].dropna() 
    fig, ax = plt.subplots(figsize = (12,6))
    sns.histplot(vals, bins = 40, kde = True)
    plt.xlabel("Điểm")
    plt.ylabel(f"Phổ điểm môn {SUBJECT_LABEL[subj]}")
    st.pyplot(fig)
    plt.show()
    
    st.metric("Số thí sinh dự thi", f"{len(vals):,}")
    st.metric("Điểm trung bình", f"{vals.mean():.2f}")
    st.metric("Độ lệch chuẩn", f"{vals.std():.2f}")
        
    st.header("Top thủ khoa theo tổ hợp")    
    combo_dash = st.selectbox("Chọn tổ hợp", processor.khoi_thi.keys())
    top_n = st.slider("Số lượng thủ khoa hiển thị", 5, 50, 10)
    
    sub_dash = processor.get_combo_data(combo_dash)
    top = sub_dash.sort_values(by = "tong_diem", ascending = False).head(top_n)
    st.write(top)
    
        
            
        
            
        

        
    
