# app/Dashboard/dashboard.py
# GenAI Business Insights — Universal Dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from app.app import (
    load_data, clean_data, calculate_basic_metrics,
    run_forecasting, run_segmentation, run_anomaly,
)
from ai_engine.ai_engine import (
    build_data_summary, build_ml_summary,
    create_business_ai_chain, ask_question,
)

# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────

st.set_page_config(
    page_title="GenAI Business Insights",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────
# CSS — Light Purple & Pink Theme
# ─────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #FAF5FF !important; }

[data-testid="stMetric"] {
    background: white !important;
    border: 1px solid #E9D5FF !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 2px 8px rgba(147,51,234,0.08) !important;
}
[data-testid="stMetricValue"] {
    color: #7C3AED !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}
[data-testid="stMetric"] label {
    color: #9333EA !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

h1, h2, h3, h4 { color: #3B0764 !important; }

[data-testid="stSidebar"] {
    background: white !important;
    border-right: 1px solid #E9D5FF !important;
}

[data-testid="stFileUploader"] {
    background: #FAF5FF !important;
    border: 2px dashed #C084FC !important;
    border-radius: 14px !important;
}

[data-baseweb="tab-list"] {
    background: #F3E8FF !important;
    border-radius: 12px !important;
    padding: 4px !important;
}
[aria-selected="true"] {
    background: white !important;
    color: #7C3AED !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 4px rgba(124,58,237,0.15) !important;
}
[data-baseweb="tab"] { color: #9333EA !important; }

.stTextInput input {
    background: white !important;
    border: 1px solid #DDD6FE !important;
    border-radius: 10px !important;
    color: #3B0764 !important;
}

.stButton button {
    background: white !important;
    border: 1px solid #DDD6FE !important;
    color: #7C3AED !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
}
.stButton button:hover {
    background: #F3E8FF !important;
    border-color: #C084FC !important;
}

[data-testid="baseButton-primary"] {
    background: linear-gradient(90deg, #7C3AED, #DB2777) !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.ai-bubble {
    background: #F3E8FF;
    border: 1px solid #DDD6FE;
    border-radius: 4px 14px 14px 14px;
    padding: 14px 18px;
    color: #3B0764;
    font-size: 14px;
    line-height: 1.75;
    margin-top: 10px;
}
.user-bubble {
    background: #FDF2F8;
    border: 1px solid #FBCFE8;
    border-radius: 14px 4px 14px 14px;
    padding: 14px 18px;
    color: #831843;
    font-size: 14px;
    margin-top: 10px;
    text-align: right;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E9D5FF !important;
    border-radius: 12px !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #FAF5FF; }
::-webkit-scrollbar-thumb { background: #C084FC; border-radius: 3px; }

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# HEADER
# ─────────────────────────────────────

st.markdown("""
<div style='padding: 8px 0 24px'>
    <div style='font-size:38px; font-weight:700;
                background: linear-gradient(90deg, #7C3AED, #DB2777);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;'>
        🧠 GenAI Business Insights
    </div>
    <div style='color:#9333EA; font-size:14px; margin-top:6px'>
        Upload any dataset &rarr; Auto clean &rarr; ML Analysis &rarr; Ask anything
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get free key at console.groq.com"
    )
    st.divider()
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload any CSV file", type=["csv"])
    st.divider()
    st.markdown("""
    <div style='font-size:12px; color:#7C3AED; line-height:2.2'>
    <b style='color:#3B0764'>Tech Stack</b><br>
    🐍 Python &middot; Streamlit<br>
    🤖 Groq LLaMA 3 &middot; LangChain<br>
    📊 Plotly &middot; Pandas<br>
    🔬 scikit-learn
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────

if uploaded_file:
    if ("last_file" not in st.session_state or
            st.session_state.last_file != uploaded_file.name):

        with st.status("🔄 Running Intelligence Pipeline...",
                       expanded=True) as status:

            st.write("📥 Loading dataset...")
            df = load_data(uploaded_file)

            st.write("🧹 Cleaning data...")
            df, log = clean_data(df)
            for step in log[-5:]:
                st.write(f"  → {step}")

            st.write("📊 Calculating metrics...")
            metrics = calculate_basic_metrics(df)

            st.write("🔮 Running forecasting...")
            forecast_results = run_forecasting(df)

            st.write("👥 Running segmentation...")
            seg_results = run_segmentation(df)

            st.write("🚨 Running anomaly detection...")
            anomaly_results = run_anomaly(df)

            st.write("🧠 Preparing AI context...")
            data_summary = build_data_summary(df)
            ml_summary = build_ml_summary(
                forecast_results if forecast_results.get("available") else {},
                seg_results if seg_results.get("available") else {},
                anomaly_results if anomaly_results.get("available") else {},
            )

            st.session_state.update({
                "df":               df,
                "metrics":          metrics,
                "forecast_results": forecast_results,
                "seg_results":      seg_results,
                "anomaly_results":  anomaly_results,
                "data_summary":     data_summary,
                "ml_summary":       ml_summary,
                "last_file":        uploaded_file.name,
                "chat_history":     [],
            })

            status.update(label="✅ Pipeline Complete!", state="complete")

# ─────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────

if "df" in st.session_state:

    df               = st.session_state.df
    metrics          = st.session_state.metrics
    forecast_results = st.session_state.forecast_results
    seg_results      = st.session_state.seg_results
    anomaly_results  = st.session_state.anomaly_results
    detected         = metrics.get("detected", {})

    target_label = metrics.get("target_label", "Value")

    # ── Detected column chips ─────────────────────────────────
    chip_html = "<div style='margin-bottom:16px'>"
    for role, col in detected.items():
        chip_html += (
            f"<span style='display:inline-block; background:#F3E8FF; "
            f"border:1px solid #DDD6FE; border-radius:999px; "
            f"padding:2px 12px; font-size:12px; color:#7C3AED; "
            f"margin:2px; font-family:monospace'>"
            f"{role}: {col}</span>"
        )
    chip_html += "</div>"
    st.markdown(chip_html, unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total " + target_label,
                  f"{metrics['total_revenue']:,.0f}")
    with col2:
        st.metric("Avg per Transaction",
                  f"{metrics['avg_order_value']:,.2f}")
    with col3:
        st.metric("Unique Customers",
                  f"{metrics['total_customers']:,}")
    with col4:
        if anomaly_results.get("available"):
            st.metric("Anomalies Found",
                      f"{anomaly_results['n_anomalies']:,} "
                      f"({anomaly_results['pct']}%)")
        else:
            st.metric("Total Columns", len(df.columns))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔮 Forecasting",
        "👥 Segmentation", "🚨 Anomalies", "🤖 AI Assistant"
    ])

    # ────────────────────────────────────────────────────────
    # TAB 1: Overview
    # ────────────────────────────────────────────────────────
    with tab1:

        # Monthly trend
        monthly = metrics.get("monthly_sales", pd.DataFrame())
        if not monthly.empty:
            st.subheader(f"📅 {target_label} Over Time")
            fig = px.area(
                monthly, x="Month", y="Value",
                color_discrete_sequence=["#9333EA"],
                labels={"Value": target_label, "Month": ""}
            )
            fig.update_traces(
                fillcolor="rgba(147,51,234,0.1)",
                line_color="#C084FC"
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#4B5563",
            )
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        # Top products/items
        with col1:
            top_p  = metrics.get("top_products", pd.Series())
            prod_c = metrics.get("top_products_col", "Item")
            if not top_p.empty:
                st.subheader(f"🛒 Top 10 by {prod_c}")
                tp_df = top_p.reset_index()
                tp_df.columns = [prod_c, target_label]
                fig_p = px.bar(
                    tp_df,
                    x=target_label,
                    y=prod_c,
                    orientation="h",
                    color_discrete_sequence=["#A855F7"],
                )
                fig_p.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#4B5563",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("No product/item column detected")

        # Top countries/regions
        with col2:
            top_c  = metrics.get("top_countries", pd.Series())
            ctry_c = metrics.get("top_countries_col", "Region")
            if not top_c.empty:
                st.subheader(f"🌍 Top {ctry_c}")
                fig_c = px.pie(
                    values=top_c.values,
                    names=top_c.index,
                    hole=0.55,
                    color_discrete_sequence=[
                        "#9333EA", "#C084FC", "#A855F7",
                        "#7C3AED", "#DB2777", "#F472B6",
                        "#EC4899", "#E879F9"
                    ]
                )
                fig_c.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#4B5563",
                )
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.info("No country/region column detected")

        with st.expander("📋 View Raw Data"):
            st.dataframe(df.head(100), use_container_width=True)

    # ────────────────────────────────────────────────────────
    # TAB 2: Forecasting
    # ────────────────────────────────────────────────────────
    with tab2:
        st.subheader("🔮 Sales Forecasting")
        if forecast_results.get("available"):
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Model", "Linear Regression")
            with c2: st.metric("R² Score", forecast_results["r2"])
            with c3: st.metric("Trend",    forecast_results["trend"])

            hist = forecast_results["historical"]
            fc   = forecast_results["forecast"]
            tgt  = forecast_results.get("target", target_label)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist["Month"], y=hist["Actual"],
                name="Actual",
                line=dict(color="#9333EA", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=hist["Month"], y=hist["Fitted"],
                name="Trend Line",
                line=dict(color="#C084FC", width=2, dash="dot")
            ))
            fig.add_trace(go.Scatter(
                x=fc["Month"], y=fc["Forecast"],
                name="Forecast",
                line=dict(color="#DB2777", width=3, dash="dash"),
                mode="lines+markers",
                marker=dict(size=10, symbol="diamond")
            ))
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#4B5563",
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Historical")
                st.dataframe(hist, use_container_width=True)
            with c2:
                st.markdown("#### Forecast")
                st.dataframe(fc, use_container_width=True)
        else:
            st.info("⚠️ Forecasting requires a date column and a numeric target column in your dataset.")

    # ────────────────────────────────────────────────────────
    # TAB 3: Segmentation
    # ────────────────────────────────────────────────────────
    with tab3:
        st.subheader("👥 Customer Segmentation")
        if seg_results.get("available"):
            c1, c2 = st.columns(2)
            with c1:
                counts = seg_results["counts"]
                fig_s = px.bar(
                    counts,
                    x="Segment", y="Count",
                    color="Segment",
                    color_discrete_map={
                        "Low Value":  "#C084FC",
                        "Mid Value":  "#9333EA",
                        "High Value": "#DB2777",
                    },
                    text="Count"
                )
                fig_s.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#4B5563",
                    showlegend=False,
                )
                st.plotly_chart(fig_s, use_container_width=True)

            with c2:
                st.markdown("#### Segment Details")
                st.dataframe(seg_results["summary"],
                             use_container_width=True)
                st.caption(
                    "Segmentation based on total spending + order "
                    "frequency using K-Means clustering (k=3)"
                )
        else:
            st.info("⚠️ Segmentation requires a numeric target column.")

    # ────────────────────────────────────────────────────────
    # TAB 4: Anomalies
    # ────────────────────────────────────────────────────────
    with tab4:
        st.subheader("🚨 Anomaly Detection")
        if anomaly_results.get("available"):
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Anomalies",
                                f"{anomaly_results['n_anomalies']:,}")
            with c2: st.metric("Rate",
                                f"{anomaly_results['pct']}%")
            with c3: st.metric("Model", "Isolation Forest")

            full_df  = anomaly_results["full_df"]
            num_cols = anomaly_results.get("num_cols", [])

            if len(num_cols) >= 2:
                fig_a = px.scatter(
                    full_df,
                    x=num_cols[0],
                    y=num_cols[1],
                    color="is_anomaly",
                    color_discrete_map={
                        True:  "#DB2777",
                        False: "#9333EA"
                    },
                    opacity=0.6,
                    labels={"is_anomaly": "Anomaly"}
                )
                fig_a.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#4B5563",
                )
                st.plotly_chart(fig_a, use_container_width=True)

            with st.expander("🔍 View Anomalous Records"):
                display_cols = [c for c in full_df.columns
                                if c not in ["is_anomaly", "anomaly_score"]]
                st.dataframe(
                    anomaly_results["anomalies"][display_cols].head(50),
                    use_container_width=True
                )
        else:
            st.info("⚠️ Anomaly detection requires numeric columns.")

    # ────────────────────────────────────────────────────────
    # TAB 5: AI Assistant
    # ────────────────────────────────────────────────────────
    with tab5:
        st.subheader("🤖 Business AI Assistant")
        st.caption("Powered by Groq LLaMA 3 70B + LangChain")

        if not groq_key:
            st.warning(
                "⚠️ Enter your Groq API key in the sidebar.\n\n"
                "Get a free key at **console.groq.com** — takes 2 minutes!"
            )
        else:
            st.markdown("#### 💡 Suggested Questions")
            suggestions = [
                f"What is the total {target_label} and top contributors?",
                "Which customer segment should we focus on for growth?",
                "What does the forecast trend suggest for next period?",
                "Are there any anomalies I should be concerned about?",
            ]
            cols = st.columns(2)
            for i, q in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(q, key=f"sug_{i}",
                                 use_container_width=True):
                        st.session_state.chat_history.append(
                            {"role": "user", "content": q}
                        )

            st.divider()

            for msg in st.session_state.get("chat_history", []):
                if msg["role"] == "user":
                    st.markdown(
                        f"<div class='user-bubble'>👤 {msg['content']}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='ai-bubble'>🤖 {msg['content']}</div>",
                        unsafe_allow_html=True
                    )

            user_q = st.text_input(
                "Ask anything about your data...",
                placeholder="e.g. What are the top 3 business insights?",
                key="user_question"
            )

            if st.button("Ask AI →", type="primary") and user_q:
                with st.spinner("🧠 Thinking..."):
                    chain  = create_business_ai_chain(groq_key)
                    answer = ask_question(
                        chain, user_q,
                        st.session_state.data_summary,
                        st.session_state.ml_summary,
                    )
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_q}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.rerun()

            if st.session_state.get("chat_history"):
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()

# ─────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────

else:
    st.markdown("""
    <div style='text-align:center; padding:60px 0'>
        <div style='font-size:72px'>🧠</div>
        <div style='font-size:28px; font-weight:700;
                    background: linear-gradient(90deg, #7C3AED, #DB2777);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 16px 0 8px'>
            Upload Your Business Dataset
        </div>
        <div style='color:#9333EA; font-size:15px'>
            Drop any CSV in the sidebar &mdash;
            the system handles everything automatically
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in zip(
        [c1, c2, c3, c4],
        ["🧹", "🤖", "📊", "🧠"],
        ["Auto Clean", "Auto ML", "Smart Charts", "AI Q&A"],
        ["Handles any messy CSV",
         "Forecast + Segment + Anomaly",
         "Adapts to your columns",
         "Ask anything freely"]
    ):
        with col:
            st.markdown(f"""
            <div style='background:white;
                        border:1px solid #E9D5FF;
                        border-radius:16px;
                        padding:24px;
                        text-align:center;
                        box-shadow:0 2px 8px rgba(147,51,234,0.06)'>
                <div style='font-size:32px'>{icon}</div>
                <div style='color:#7C3AED; font-weight:600;
                            margin:10px 0 6px; font-size:15px'>{title}</div>
                <div style='color:#9CA3AF; font-size:12px'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)