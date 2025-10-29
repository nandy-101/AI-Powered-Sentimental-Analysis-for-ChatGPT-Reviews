# 🤖 AI-Powered Sentiment Analysis for ChatGPT Reviews

This project performs **Sentiment Analysis** on ChatGPT user reviews to uncover insights about user satisfaction, recurring complaints, and version performance.  
It integrates **data preprocessing, exploratory analysis, and machine learning/deep learning models**, and presents insights interactively through a **Streamlit Dashboard**.

---

## 📘 Project Overview

With ChatGPT becoming a global tool, understanding user sentiment across platforms, versions, and geographies is crucial.  
This project:
- Cleans and preprocesses user review data
- Explores trends through EDA and visualizations
- Builds sentiment classification models (ML + DL)
- Deploys an interactive Streamlit dashboard for insights

---

## 📂 Dataset Description

**File:** `chatgpt_style_reviews_dataset.xlsx`

| Column | Description |
|---------|-------------|
| `date` | Date of review |
| `title` | Review headline/title |
| `review` | Full review text |
| `rating` | Star rating (1–5) |
| `helpful_votes` | Number of helpful votes |
| `platform` | Platform (Web / Mobile) |
| `verified_purchase` | Whether user was verified |
| `version` | ChatGPT app/version reviewed |
| `location` | User country/region |
| `language` | Language of the review |

---

## 🚀 Deliverables

| Deliverable | Description |
|--------------|-------------|
| 🧹 **Cleaned Dataset** | `clean_chatgpt_reviews.csv` (preprocessed, ready for ML) |
| 📊 **EDA Visualizations** | Plots for 10 key questions (ratings, sentiment, keywords, etc.) |
| 🤖 **Sentiment Models** | TF-IDF + Logistic Regression & SBERT + XGBoost |
| 🧩 **Interactive Dashboard** | `app.py` (Streamlit app for insights & live predictions) |
| 📈 **Model Performance Report** | `model_performance_report.csv` |
| 📦 **Deployment Files** | `deliverables.zip` (contains all outputs & models) |

---

