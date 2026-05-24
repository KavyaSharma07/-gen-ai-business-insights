# ai_engine/ai_engine.py
# Powered by Groq LLaMA 3 + LangChain

# NEW
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ─────────────────────────────────────
# 1. BUILD DATA SUMMARY
# ─────────────────────────────────────

def build_data_summary(df: pd.DataFrame) -> str:
    """
    Converts dataframe into a text summary for the AI.
    This becomes the AI's 'knowledge' about your dataset.
    """

    summary_lines = [
        f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
        f"Columns available: {', '.join(df.columns.tolist())}",
    ]

    # Revenue stats
    if "TotalSales" in df.columns:
        summary_lines += [
            f"Total Revenue: £{df['TotalSales'].sum():,.2f}",
            f"Average Order Value: £{df['TotalSales'].mean():,.2f}",
            f"Max Single Transaction: £{df['TotalSales'].max():,.2f}",
        ]

    # Customer stats
    if "CustomerID" in df.columns:
        summary_lines.append(
            f"Total Unique Customers: {df['CustomerID'].nunique():,}"
        )

    # Top country
    if "Country" in df.columns and "TotalSales" in df.columns:
        top_country = (
            df.groupby("Country")["TotalSales"]
            .sum()
            .idxmax()
        )
        top_country_val = (
            df.groupby("Country")["TotalSales"]
            .sum()
            .max()
        )
        summary_lines.append(
            f"Top Country: {top_country} (£{top_country_val:,.2f})"
        )

    # Top product
    if "Description" in df.columns and "TotalSales" in df.columns:
        top_product = (
            df.groupby("Description")["TotalSales"]
            .sum()
            .idxmax()
        )
        summary_lines.append(f"Top Product: {top_product}")

    # Date range
    if "InvoiceDate" in df.columns:
        summary_lines.append(
            f"Date Range: {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}"
        )

    return "\n".join(summary_lines)


# ─────────────────────────────────────
# 2. BUILD ML SUMMARY
# ─────────────────────────────────────

def build_ml_summary(forecast_results, seg_results, anomaly_results):
    lines = []

    # Forecasting — only if available
    if forecast_results and forecast_results.get("available"):
        lines.append(
            f"Sales Trend: {forecast_results['trend']} "
            f"(R² = {forecast_results['r2']})"
        )
        forecast_df = forecast_results["forecast"]
        for _, row in forecast_df.iterrows():
            lines.append(
                f"Forecast {row['Month']}: £{row['Forecast']:,.2f}"
            )

    # Segmentation — only if available
    if seg_results and seg_results.get("available"):
        summary = seg_results["summary"]
        for _, row in summary.iterrows():
            lines.append(
                f"Segment '{row['Segment']}': "
                f"{int(row['num_customers'])} customers, "
                f"avg spend £{row['avg_spent']:,.2f}"
            )

    # Anomaly — only if available
    if anomaly_results and anomaly_results.get("available"):
        lines.append(
            f"Anomalies: {anomaly_results['n_anomalies']:,} "
            f"({anomaly_results['pct']}% of data)"
        )

    return "\n".join(lines) if lines else "No ML results available."
# ─────────────────────────────────────
# 3. CREATE AI CHAIN
# ─────────────────────────────────────

def create_business_ai_chain(groq_api_key: str):
    """
    Creates a LangChain pipeline using Groq LLaMA 3.
    
    groq_api_key: get free at console.groq.com
    """

    # Initialize Groq LLaMA 3
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.3,
        max_tokens=512,
    )

    # Prompt template
    prompt = PromptTemplate(
        input_variables=["data_summary", "ml_summary", "question"],
        template="""
You are an expert business data analyst AI assistant.
You have been given a summary of a business dataset and ML analysis results.
Answer the user's question clearly with data-driven reasoning.

Rules:
- Always refer to actual numbers from the context when available.
- Be specific. Do not give generic business advice.
- Use bullet points where helpful.
- If something is not in the context, say "This information is not in the dataset."
- Keep answers concise — under 150 words unless the question needs more.

Dataset Summary:
{data_summary}

ML Analysis Results:
{ml_summary}

Question: {question}

Answer:
"""
    )

    # Connect prompt → LLM
    chain = prompt | llm | StrOutputParser()
    return chain


# ─────────────────────────────────────
# 4. ASK A QUESTION
# ─────────────────────────────────────

def ask_question(
    chain,
    question: str,
    data_summary: str,
    ml_summary: str
) -> str:
    """
    Sends question + context to LLaMA 3 and returns answer.
    """
    try:
        response = chain.invoke({
        "data_summary": data_summary,
        "ml_summary":   ml_summary,
        "question":     question,
    })
        return response.strip()

    except Exception as e:
        return f"❌ AI Error: {str(e)}"