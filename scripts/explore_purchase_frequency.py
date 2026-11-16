"""scripts/explore_purchase_frequency.py — xem phân phối số lần mua / HH trước khi chia bin."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/Dunnhumby/transaction_data.csv")

def purchase_frequency_profile(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path, usecols=["household_key", "BASKET_ID"],
                     dtype={"household_key": str, "BASKET_ID": str})
    freq = df.groupby("household_key")["BASKET_ID"].nunique().rename("purchase_count")

    print("=" * 55)
    print("  Phân phối số lần mua / household")
    print("=" * 55)

    # Thống kê cơ bản
    desc = freq.describe(percentiles=[.1,.2,.25,.5,.75,.8,.9,.95,.99])
    print(desc.to_string())

    # Phân phối thô theo từng giá trị nhỏ (1–20)
    print("\n--- Số HH có đúng N lần mua (N = 1..20) ---")
    vc = freq.value_counts().sort_index()
    for n in range(1, 21):
        cnt = vc.get(n, 0)
        bar = "█" * min(cnt // 5, 50)
        print(f"  {n:3d} lần : {cnt:5,}  {bar}")
    rest = vc[vc.index > 20].sum()
    print(f"  >20 lần : {rest:5,}")

    # Gợi ý chia bin tự động (quantile-based)
    print("\n--- Gợi ý bin theo quantile ---")
    for q in [0.25, 0.50, 0.75, 0.90, 0.95]:
        print(f"  p{int(q*100):02d} = {int(freq.quantile(q))}")

    return freq

if __name__ == "__main__":
    purchase_frequency_profile()
