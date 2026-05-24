# 🧠 GenAI Business Insights Dashboard
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![LangChain](https://img.shields.io/badge/LLM-LangChain-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
> An end-to-end **Generative AI powered Business Intelligence system** that accepts any structured dataset, automatically cleans it, runs machine learning models, generates an interactive dashboard, and answers business questions in plain English using **Groq LLaMA 3 + LangChain**.

---

## 🎯 Problem Statement

Most businesses collect data but can't analyze it — hiring data analysts is expensive, Excel breaks on large datasets, and no single tool works for every dataset. This system solves all three problems in one upload.

---

## 🚀 Live Demo

> Upload any CSV → Auto clean → ML Analysis → Interactive Dashboard → Ask anything

<img width="1902" height="815" alt="Screenshot 2026-05-24 003004" src="https://github.com/user-attachments/assets/27458163-a7fb-4274-a166-c5832253f232" />


---

## ✨ Features

### 🧹 Auto Data Cleaning
- Handles missing values, duplicates, wrong encodings
- Auto-detects and parses date columns
- Removes negative/zero values from transactional columns
- Fills numeric nulls with median (outlier-safe)
- Shows full cleaning log to user

### 🔍 Smart Schema Detection
- Automatically detects column roles from ANY dataset
- Finds: target, date, customer, product, country, order columns
- No hardcoding — works on retail, HR, finance, e-commerce data

### 🤖 Auto ML Pipeline
| Model | Purpose | Library |
|-------|---------|---------|
| Linear Regression | Sales/Revenue Forecasting | scikit-learn |
| K-Means Clustering | Customer Segmentation | scikit-learn |
| Isolation Forest | Anomaly Detection | scikit-learn |

### 📊 Interactive Dashboard
- Monthly trend chart (auto-adapts to detected date + target)
- Top 10 products/items by revenue
- Country/region revenue distribution (donut chart)
- Forecasting chart with trend line + future predictions
- Customer segmentation bar chart with segment details
- Anomaly scatter plot with flagged transaction table

### 🧠 AI Business Assistant (RAG-style)
- Powered by **Groq LLaMA 3 70B** — fastest open-source LLM
- Orchestrated with **LangChain** prompt pipeline
- Injected with data summary + ML results as context
- Answers any free-form business question with data-backed reasoning
- Suggested questions + free-form chat interface

---

## 🏗️ Architecture

```
User uploads CSV
      ↓
┌─────────────────────────────┐
│     Data Engine (app.py)    │
│  Load → Clean → Detect      │
│  Schema → Basic Metrics     │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│       ML Pipeline           │
│  Forecasting (LinearReg)    │
│  Segmentation (KMeans)      │
│  Anomaly (IsolationForest)  │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│    AI Assistant             │
│  Data Summary + ML Results  │
│  → LangChain Prompt Chain   │
│  → Groq LLaMA 3 70B         │
│  → Natural Language Answer  │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│   Streamlit Dashboard       │
│  5 Tabs: Overview,          │
│  Forecast, Segmentation,    │
│  Anomalies, AI Assistant    │
└─────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI** | Streamlit | Interactive web dashboard |
| **LLM** | Groq LLaMA 3 70B | Natural language Q&A |
| **AI Orchestration** | LangChain | Prompt pipeline management |
| **ML** | scikit-learn | Forecasting, clustering, anomaly detection |
| **Data** | Pandas, NumPy | Data manipulation and cleaning |
| **Visualization** | Plotly | Interactive charts |
| **Language** | Python 3.10+ | Core language |

---

## 📁 Project Structure

```
GEN_AI_Business_Insights/
│
├── app/
│   ├── app.py                 # Data engine + ML pipeline
│   └── Dashboard/
│       └── dashboard.py       # Streamlit UI
│
├── ai_engine/
│   └── ai_engine.py           # Groq + LangChain AI assistant
│
├── data/
│   └── sales_data.csv         # Sample dataset
│
├── .streamlit/
│   └── config.toml            # Theme configuration
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/gen-ai-business-insights.git
cd gen-ai-business-insights
```

### 2. Create virtual environment
```bash
conda create -n genai python=3.10
conda activate genai
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get your free Groq API key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free
3. Create an API key
4. Paste it in the dashboard sidebar

### 5. Run the dashboard
```bash
streamlit run app/Dashboard/dashboard.py
```

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.3.0
langchain>=1.3.0
langchain-groq>=1.1.0
langchain-core>=1.4.0
```

---

## 🔑 How It Works

### Step 1 — Upload Any CSV
Drop any structured business dataset. The system handles the rest.

### Step 2 — Auto Pipeline Runs
```
Loading → Cleaning → Schema Detection → 
Metrics → Forecasting → Segmentation → Anomaly Detection
```

### Step 3 — Explore the Dashboard
Five tabs give you complete visibility into your data:
- **Overview** — key metrics and trends
- **Forecasting** — future revenue predictions  
- **Segmentation** — customer value groups
- **Anomalies** — suspicious transactions flagged
- **AI Assistant** — ask anything in plain English

### Step 4 — Ask Questions
```
"Which customer segment drives the most revenue?"
"What does the forecast say about next quarter?"
"Are the anomalies I see concerning?"
"What are the top 3 actionable insights?"
```

---

## 🧪 Sample Datasets This Works On

| Dataset Type | Columns Needed | ML Features Available |
|-------------|---------------|----------------------|
| Retail Sales | Date, Product, Sales, Country | All 3 |
| E-Commerce | OrderDate, Category, Amount | All 3 |
| HR Data | JoiningDate, Department, Salary | Segmentation + Anomaly |
| Finance | Date, Revenue, Expenses | Forecasting + Anomaly |
| Any CSV | At least 1 numeric column | Anomaly minimum |

---

## 💡 Key Engineering Decisions

**Why Groq over OpenAI?**
Groq runs LLaMA 3 on custom LPU chips — 10x faster inference, free tier available, and open-source model shows real AI engineering knowledge.

**Why K-Means for segmentation?**
Interpretable, scalable to 500k+ rows, maps naturally to Low/Mid/High value business segments. StandardScaler used instead of MinMaxScaler to handle business data outliers correctly.

**Why Isolation Forest for anomalies?**
Distribution-free, handles multi-dimensional data, catches anomalies across multiple columns simultaneously — unlike Z-score or IQR methods.

**Why Linear Regression for forecasting?**
Works on any dataset size with any time granularity. No assumptions about seasonality or stationarity. Universally applicable for a dataset-agnostic system.

**Why RAG-style AI?**
The LLM is grounded with actual data statistics + ML results injected into the prompt — preventing hallucination and ensuring answers reference real numbers.

---

## 📸 Screenshots

| Overview Tab |
|-------------|
|<img width="1909" height="715" alt="Screenshot 2026-05-24 003127" src="https://github.com/user-attachments/assets/2a08c208-3f2b-4757-a8bc-67ca2182dd8c" />|
|<img width="1909" height="873" alt="Screenshot 2026-05-24 003200" src="https://github.com/user-attachments/assets/f530472e-cb6f-433f-b14c-60d46f551bfe" />|

| Forcasting Tab |
|----------------|
|<img width="1891" height="871" alt="Screenshot 2026-05-24 005205" src="https://github.com/user-attachments/assets/14d08888-2309-42a2-8869-6d75f1429e67" />|

| Segmentation Tab |
|-----------------|
|<img width="1878" height="851" alt="Screenshot 2026-05-24 005223" src="https://github.com/user-attachments/assets/38b55847-79d9-45e9-8a20-b1f3fd18c096" />|
 
| Anomalies Tab |
|--------------|
|<img width="1897" height="858" alt="Screenshot 2026-05-24 005251" src="https://github.com/user-attachments/assets/f38029a0-d134-48ed-b26b-d47ced205c2f" />|

---


## Future Improvements
- Multi-file dataset support
- Cloud deployment
- PDF insight reports
- Vector database integration
- Advanced forecasting models
---

⭐ **Star this repo if you found it useful!**
