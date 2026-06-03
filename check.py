import pandas as pd

for name in ["cpi", "gdp", "tcb_price", "vnindex", "usdvnd", "tcb_fundamentals"]:
    df = pd.read_parquet(f"data/raw/{name}.parquet")
    print(f"\n=== {name}: {df.shape} ===")
    print(df.head(5))
    print(df.tail(5))