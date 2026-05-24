# test_read.py — quick CSV check
import pandas as pd
from pathlib import Path
import sys

proj = Path.cwd()
csv_path = proj / "data" / "cleaned_sales_data.csv"


print(f"Working folder: {proj}")
print(f"Trying to open: {csv_path}")

if not csv_path.exists():
    print("ERROR: file not found. Check file name and its location (should be data/sales_data.csv).")
    sys.exit(2)

# Try reading with utf-8, fallback to latin1 if utf-8 fails
for enc in ("utf-8", "latin1"):
    try:
        df = pd.read_csv(csv_path, encoding=enc, on_bad_lines="skip")
        print(f"Read OK with encoding={enc}")
        break
    except Exception as e:
        print(f"Failed to read with encoding={enc}: {e}")
        df = None

if df is None:
    print("ERROR: couldn't read CSV with tested encodings.")
    sys.exit(3)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:50])   # show first 50 column names
print("First 5 rows:")
print(df.head().to_string(index=False))
