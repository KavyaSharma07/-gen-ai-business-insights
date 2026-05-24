# app/app.py — GenAI Business Insights
# Handles: Loading, Cleaning, EDA, ML

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────
# 0. SMART COLUMN DETECTOR
# ─────────────────────────────────────

def detect_columns(df):
    """
    Detects column roles from ANY dataset automatically.
    Returns dict with: target, date, customer, product, country, order
    """
    detected = {}
    cols_lower = {c.lower().replace(" ", "").replace("_", ""): c
                  for c in df.columns}

    # ── Target / revenue column ──────────────────────────────
    for kw in ["totalsales", "revenue", "sales", "amount",
               "total", "price", "value", "income", "profit"]:
        for cl, ca in cols_lower.items():
            if kw in cl and pd.api.types.is_numeric_dtype(df[ca]):
                detected["target"] = ca
                break
        if "target" in detected:
            break
    # Fallback: numeric column with highest sum
    if "target" not in detected:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            detected["target"] = df[num_cols].sum().idxmax()

    # ── Date column ──────────────────────────────────────────
    # First check dtype
    date_dtype_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    if date_dtype_cols:
        detected["date"] = date_dtype_cols[0]
    else:
        for kw in ["date", "time", "invoicedate", "orderdate", "created"]:
            for cl, ca in cols_lower.items():
                if kw in cl:
                    try:
                        pd.to_datetime(df[ca].dropna().head(10), errors="raise")
                        detected["date"] = ca
                        break
                    except Exception:
                        pass
            if "date" in detected:
                break

    # ── Customer / ID column ─────────────────────────────────
    for kw in ["customerid", "customer", "client",
               "userid", "user", "buyerid"]:
        for cl, ca in cols_lower.items():
            if kw in cl:
                detected["customer"] = ca
                break
        if "customer" in detected:
            break

    # ── Product / description column ─────────────────────────
    for kw in ["description", "product", "productname",
               "item", "title", "name", "sku"]:
        for cl, ca in cols_lower.items():
            if kw in cl:
                detected["product"] = ca
                break
        if "product" in detected:
            break

    # ── Country / region column ──────────────────────────────
    for kw in ["country", "region", "location",
               "city", "state", "market"]:
        for cl, ca in cols_lower.items():
            if kw in cl:
                detected["country"] = ca
                break
        if "country" in detected:
            break

    # ── Order / invoice column ───────────────────────────────
    for kw in ["invoiceno", "invoice", "orderid",
               "order", "transactionid", "transaction"]:
        for cl, ca in cols_lower.items():
            if kw in cl:
                detected["order"] = ca
                break
        if "order" in detected:
            break

    # ── Category column ──────────────────────────────────────
    for kw in ["category", "type", "segment",
               "department", "division", "class"]:
        for cl, ca in cols_lower.items():
            if kw in cl:
                detected["category"] = ca
                break
        if "category" in detected:
            break

    print(f"\n🔍 Detected columns: {detected}")
    return detected


# ─────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────

def load_data(source):
    """Loads CSV from file path or Streamlit UploadedFile."""
    encodings = ["utf-8", "latin1", "ISO-8859-1"]
    for enc in encodings:
        try:
            if hasattr(source, "read"):
                source.seek(0)
                df = pd.read_csv(source, encoding=enc, on_bad_lines="skip")
            else:
                df = pd.read_csv(source, encoding=enc, on_bad_lines="skip")
            print(f"✅ Loaded with encoding: {enc} — shape {df.shape}")
            return df
        except Exception:
            continue
    raise ValueError("❌ Could not read file. Please upload a valid CSV.")


# ─────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────

def clean_data(df):
    """Auto-cleans any dataframe. Returns cleaned df + log."""
    log = []
    original_rows = len(df)

    # Drop fully empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        df.drop(columns=empty_cols, inplace=True)
        log.append(f"Dropped {len(empty_cols)} empty columns")

    # Parse date columns by name
    for col in df.columns:
        if df[col].dtype == object:
            if any(k in col.lower() for k in ["date", "time"]):
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().sum() > len(df) * 0.5:
                        df[col] = parsed
                        log.append(f"Parsed '{col}' as datetime")
                except Exception:
                    pass

    # Strip whitespace from text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("nan", np.nan)

    # Remove negative/zero from quantity & price columns
    suspicious = [c for c in df.columns
                  if any(k in c.lower() for k in
                         ["qty", "quantity", "price", "amount", "cost"])]
    for col in suspicious:
        if pd.api.types.is_numeric_dtype(df[col]):
            before = len(df)
            df = df[df[col] > 0]
            removed = before - len(df)
            if removed:
                log.append(f"Removed {removed} rows with non-positive '{col}'")

    # Fill missing numeric with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            log.append(f"Filled missing '{col}' with median ({med:.2f})")

    # Drop duplicates
    dupes = df.duplicated().sum()
    if dupes:
        df.drop_duplicates(inplace=True)
        log.append(f"Removed {dupes} duplicate rows")

    # Create TotalSales if Quantity + UnitPrice exist
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        if "TotalSales" not in df.columns:
            df["TotalSales"] = df["Quantity"] * df["UnitPrice"]
            log.append("Created 'TotalSales' (Quantity × UnitPrice)")

    removed_total = original_rows - len(df)
    log.append(f"Cleaning done — {original_rows} → {len(df)} rows "
               f"({removed_total} removed)")

    return df.reset_index(drop=True), log


# ─────────────────────────────────────
# 3. BASIC METRICS
# ─────────────────────────────────────

def calculate_basic_metrics(df):
    """Calculates core metrics from ANY dataset."""
    results = {}
    detected = detect_columns(df)
    results["detected"] = detected

    target   = detected.get("target")
    date_col = detected.get("date")
    cust_col = detected.get("customer")
    prod_col = detected.get("product")
    ctry_col = detected.get("country")

    # Core numbers
    results["total_revenue"]   = float(df[target].sum()) if target else 0
    results["avg_order_value"] = float(df[target].mean()) if target else 0
    results["total_customers"] = int(df[cust_col].nunique()) if cust_col else len(df)
    results["total_products"]  = int(df[prod_col].nunique()) if prod_col else 0

    # Top products
    if prod_col and target:
        tp = (df.groupby(prod_col)[target]
              .sum()
              .sort_values(ascending=False)
              .head(10))
        results["top_products"] = tp
        results["top_products_col"] = prod_col
    else:
        results["top_products"] = pd.Series(dtype="float64")
        results["top_products_col"] = None

    # Top countries
    if ctry_col and target:
        tc = (df.groupby(ctry_col)[target]
              .sum()
              .sort_values(ascending=False)
              .head(10))
        results["top_countries"] = tc
        results["top_countries_col"] = ctry_col
    else:
        results["top_countries"] = pd.Series(dtype="float64")
        results["top_countries_col"] = None

    # Monthly sales trend
    if date_col and target:
        tmp = df.copy()
        tmp["_month"] = pd.to_datetime(
            tmp[date_col], errors="coerce"
        ).dt.to_period("M")
        monthly = (tmp.groupby("_month")[target]
                   .sum()
                   .reset_index())
        monthly.columns = ["Month", "Value"]
        monthly["Month"] = monthly["Month"].astype(str)
        results["monthly_sales"] = monthly
        results["target_label"]  = target
    else:
        results["monthly_sales"] = pd.DataFrame(columns=["Month", "Value"])
        results["target_label"]  = target or "Value"

    return results


# ─────────────────────────────────────
# 4. FORECASTING
# ─────────────────────────────────────

def run_forecasting(df, periods=3):
    """Forecasts future values using Linear Regression."""
    print("\n🔮 Running Forecasting...")
    detected = detect_columns(df)
    target   = detected.get("target")
    date_col = detected.get("date")

    if not target or not date_col:
        print("  ⚠️ Skipping — no date/target column found")
        return {"available": False}

    tmp = df.copy()
    tmp["_month"] = pd.to_datetime(
        tmp[date_col], errors="coerce"
    ).dt.to_period("M")

    monthly = (tmp.groupby("_month")[target]
               .sum()
               .reset_index()
               .sort_values("_month"))
    monthly.columns = ["Month", "Actual"]
    monthly["t"] = np.arange(len(monthly))

    if len(monthly) < 3:
        return {"available": False}

    X = monthly[["t"]].values
    y = monthly["Actual"].values

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    monthly["Fitted"] = model.predict(X)

    last_t     = monthly["t"].max()
    last_month = monthly["Month"].max()

    future_months = [last_month + i + 1 for i in range(periods)]
    future_t      = np.array([[last_t + i + 1] for i in range(periods)])
    forecast_vals = np.maximum(model.predict(future_t), 0)

    forecast_df = pd.DataFrame({
        "Month":    [str(m) for m in future_months],
        "Forecast": forecast_vals
    })

    monthly["Month"] = monthly["Month"].astype(str)
    monthly = monthly[["Month", "Actual", "Fitted"]]
    trend   = "📈 Upward" if model.coef_[0] > 0 else "📉 Downward"

    print(f"  → Trend: {trend}, R²: {r2:.3f}")
    return {
        "available":  True,
        "historical": monthly,
        "forecast":   forecast_df,
        "r2":         round(r2, 3),
        "trend":      trend,
        "target":     target,
    }


# ─────────────────────────────────────
# 5. CUSTOMER SEGMENTATION
# ─────────────────────────────────────

def run_segmentation(df, n_clusters=3):
    """Segments using KMeans on any numeric columns."""
    print("\n👥 Running Segmentation...")
    detected  = detect_columns(df)
    target    = detected.get("target")
    cust_col  = detected.get("customer")
    order_col = detected.get("order")

    if not target:
        print("  ⚠️ Skipping — no target column found")
        return {"available": False}

    # Build aggregation
    if cust_col and cust_col in df.columns:
        agg = {target: "sum"}
        if order_col and order_col in df.columns:
            agg[order_col] = "nunique"
        seg_df = df.groupby(cust_col).agg(agg).reset_index()
        seg_df.columns = (
            [cust_col, "total_spent"] if len(agg) == 1
            else [cust_col, "total_spent", "order_count"]
        )
        if "order_count" not in seg_df.columns:
            seg_df["order_count"] = 1
    else:
        seg_df = pd.DataFrame({
            "total_spent":  df[target].values,
            "order_count":  np.ones(len(df))
        })

    features = ["total_spent", "order_count"]
    k  = min(n_clusters, len(seg_df) - 1)
    if k < 2:
        return {"available": False}

    scaler = StandardScaler()
    X = scaler.fit_transform(seg_df[features].fillna(0))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    seg_df["Cluster"] = km.fit_predict(X)

    # Label by average spending
    means     = seg_df.groupby("Cluster")["total_spent"].mean().sort_values()
    labels    = ["Low Value", "Mid Value", "High Value"][:k]
    label_map = {cid: labels[i] for i, cid in enumerate(means.index)}
    seg_df["Segment"] = seg_df["Cluster"].map(label_map)

    summary = (seg_df.groupby("Segment")
               .agg(num_customers=("total_spent", "count"),
                    avg_spent=("total_spent", "mean"),
                    avg_orders=("order_count", "mean"))
               .round(2)
               .reset_index())

    counts = seg_df["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]

    print(summary.to_string(index=False))
    return {
        "available":   True,
        "customer_df": seg_df,
        "summary":     summary,
        "counts":      counts,
    }


# ─────────────────────────────────────
# 6. ANOMALY DETECTION
# ─────────────────────────────────────

def run_anomaly(df, contamination=0.05):
    """Detects anomalies using Isolation Forest on numeric columns."""
    print("\n🚨 Running Anomaly Detection...")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        print("  ⚠️ Skipping — no numeric columns")
        return {"available": False}

    X   = df[num_cols].fillna(0).values
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(X)

    result_df = df.copy()
    result_df["is_anomaly"]    = clf.predict(X) == -1
    result_df["anomaly_score"] = clf.score_samples(X)

    anomalies   = result_df[result_df["is_anomaly"]]
    n_anomalies = len(anomalies)
    pct         = round(n_anomalies / len(df) * 100, 1)

    print(f"  → {n_anomalies} anomalies ({pct}%)")
    return {
        "available":   True,
        "full_df":     result_df,
        "anomalies":   anomalies,
        "normal":      result_df[~result_df["is_anomaly"]],
        "n_anomalies": n_anomalies,
        "pct":         pct,
        "num_cols":    num_cols,
    }


# ─────────────────────────────────────
# RUN DIRECTLY (testing only)
# ─────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR  = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "sales_data.csv"

    df = load_data(DATA_PATH)
    df, log = clean_data(df)

    print("\n🧹 Cleaning Log:")
    for step in log:
        print(f"  → {step}")

    metrics = calculate_basic_metrics(df)
    print(f"\n📊 Metrics:")
    print(f"  → Revenue:   {metrics['total_revenue']:,.2f}")
    print(f"  → Customers: {metrics['total_customers']:,}")

    fc  = run_forecasting(df)
    seg = run_segmentation(df)
    ano = run_anomaly(df)

    print(f"\n🔮 Forecast available: {fc.get('available')}")
    print(f"👥 Segmentation available: {seg.get('available')}")
    print(f"🚨 Anomalies: {ano.get('n_anomalies', 0)}")