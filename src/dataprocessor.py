import pandas as pd

PROVINCE_MAP = {
    "01": "Hà Nội", "02": "TP. Hồ Chí Minh", "03": "Hải Phòng", "04": "Đà Nẵng", "05": "Hà Giang",
    "06": "Cao Bằng", "07": "Lai Châu", "08": "Lào Cai", "09": "Tuyên Quang", "10": "Lạng Sơn",
    "11": "Bắc Kạn", "12": "Thái Nguyên", "13": "Yên Bái", "14": "Sơn La", "15": "Phú Thọ",
    "16": "Vĩnh Phúc", "17": "Quảng Ninh", "18": "Bắc Giang", "19": "Bắc Ninh", "21": "Hải Dương",
    "22": "Hưng Yên", "23": "Hoà Bình", "24": "Hà Nam", "25": "Nam Định", "26": "Thái Bình",
    "27": "Ninh Bình", "28": "Thanh Hoá", "29": "Nghệ An", "30": "Hà Tĩnh", "31": "Quảng Bình",
    "32": "Quảng Trị", "33": "Thừa Thiên Huế", "34": "Quảng Nam", "35": "Quảng Ngãi", "36": "Kon Tum",
    "37": "Bình Định", "38": "Gia Lai", "39": "Phú Yên", "40": "Đắk Lắk", "41": "Khánh Hoà",
    "42": "Lâm Đồng", "43": "Bình Phước", "44": "Bình Dương", "45": "Ninh Thuận", "46": "Tây Ninh",
    "47": "Bình Thuận", "48": "Đồng Nai", "49": "Long An", "50": "Đồng Tháp", "51": "An Giang",
    "52": "Bà Rịa - Vũng Tàu", "53": "Tiền Giang", "54": "Kiên Giang", "55": "Cần Thơ", "56": "Bến Tre",
    "57": "Vĩnh Long", "58": "Trà Vinh", "59": "Sóc Trăng", "60": "Bạc Liêu", "61": "Cà Mau",
    "62": "Điện Biên", "63": "Đắk Nông", "64": "Hậu Giang", "65": "Bộ Quốc phòng",
}


class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.khoi_thi = {
            "A00 (Toán - Lý - Hóa)": ["toan", "vat_li", "hoa_hoc"],
            "A01 (Toán - Lý - Anh)": ["toan", "vat_li", "ngoai_ngu"],
            "B00 (Toán - Hóa - Sinh)": ["toan", "hoa_hoc", "sinh_hoc"],
            "C00 (Văn - Sử - Địa)": ["ngu_van", "lich_su", "dia_li"],
            "D01 (Toán - Văn - Anh)": ["toan", "ngu_van", "ngoai_ngu"],
        }
        print(f"Ready to read file: {filepath}")
        
    def load_and_clean(self):
        self.df = pd.read_csv(self.filepath, dtype= {"sbd" : str})
        self.df["ma_tinh"] = self.df["sbd"].str[:2]
        self.df["ten_tinh"] = self.df["ma_tinh"].map(PROVINCE_MAP).fillna("Không xác định")
        print(f"Read {len(self.df):,} lines, {self.df.shape[1]} cols")
        
    def get_combo_data(self, ten_to_hop):
        cols = self.khoi_thi[ten_to_hop]
        sub = self.df.dropna(subset = cols).copy()
        sub["tong_diem"] = sub[cols].sum(axis = 1)
        sub["target_24"] = (sub["tong_diem"] >= 24).astype(int)
        print(f"[{ten_to_hop}] {len(sub):,} thí sinh đủ điểm, "f"{sub['target_24'].mean()*100:.1f}% đạt 24+")
        return sub

if __name__ == "__main__":
    FILE_PATH  = r"D:\Project_THPTQG_2024\data\diem_thi_thpt_2024.csv"
    try:
        proc = DataProcessor(FILE_PATH)
        proc.load_and_clean()
        df_a00 = proc.get_combo_data("A00 (Toán - Lý - Hóa)")
        print("\n 5 dòng dữ liệu đầu tiên của khối A00:")
        cols_to_show = ["sbd", "toan", "vat_li", "hoa_hoc", "tong_diem", "target_24"]
        print(df_a00[cols_to_show].head())
    except FileNotFoundError:
        print(f"ERROR: Cannot find the file {FILE_PATH}")
    except Exception as e:
        print(f"Another error: {e}")