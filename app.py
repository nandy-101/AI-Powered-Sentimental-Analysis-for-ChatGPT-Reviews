
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(layout="wide", page_title="ChatGPT Reviews Sentiment Dashboard")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

@st.cache_resource
def load_model(path):
    model = joblib.load(path)
    return model

# Paths (adjust if needed)
DATA_PATH = "clean_chatgpt_reviews.csv"
MODEL_PATH = "sentiment_pipeline.joblib"

# Load data and model
df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

st.title("🤖 ChatGPT Reviews - Sentiment Insights Dashboard")

# Sidebar filters
st.sidebar.header("🔍 Filters")
platforms = st.sidebar.multiselect(
    "Select Platform(s):", options=df["platform"].unique().tolist(),
    default=df["platform"].unique().tolist()
)
locations = st.sidebar.multiselect(
    "Select Locations (Top 20):",
    options=df["location"].value_counts().index[:20].tolist(),
    default=df["location"].value_counts().index[:10].tolist()
)
helpful_min = st.sidebar.slider("Minimum Helpful Votes:", 0, int(df["helpful_votes"].max()), 0)

# Apply filters
filtered_df = df[
    (df["platform"].isin(platforms)) &
    (df["location"].isin(locations)) &
    (df["helpful_votes"] >= helpful_min)
]

st.markdown("### 📊 Overview Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", len(filtered_df))
col2.metric("Average Rating", round(filtered_df["rating"].mean(), 2))
col3.metric("Verified Users (%)", round((filtered_df["verified_purchase"] == "Yes").mean() * 100, 2))

# --- Rating Distribution ---
st.markdown("### ⭐ Rating Distribution")
fig1, ax1 = plt.subplots(figsize=(8,4))
sns.countplot(data=filtered_df, x="rating", ax=ax1, palette="viridis", order=sorted(filtered_df["rating"].unique()))
ax1.set_title("Distribution of Ratings (1–5)")
st.pyplot(fig1)

# --- Sentiment Distribution ---
st.markdown("### 💬 Sentiment Distribution (Based on Rating Category)")
sent_counts = filtered_df["rating_sentiment"].value_counts(normalize=True).mul(100).reset_index()
sent_counts.columns = ["Sentiment", "Percentage"]

fig2, ax2 = plt.subplots(figsize=(6,4))
sns.barplot(data=sent_counts, x="Sentiment", y="Percentage", ax=ax2, palette="coolwarm")
ax2.set_title("Sentiment Split (%)")
st.pyplot(fig2)

# --- Average Rating by Platform ---
st.markdown("### 🧩 Average Rating by Platform")
avg_platform = filtered_df.groupby("platform")["rating"].mean().reset_index()
fig3, ax3 = plt.subplots(figsize=(8,4))
sns.barplot(data=avg_platform, x="platform", y="rating", ax=ax3, palette="mako")
ax3.set_title("Average Rating by Platform")
st.pyplot(fig3)

# --- Average Rating by Version ---
st.markdown("### 🧠 Average Rating by ChatGPT Version")
avg_version = filtered_df.groupby("version")["rating"].mean().reset_index().sort_values(by="rating", ascending=False)
fig4, ax4 = plt.subplots(figsize=(10,4))
sns.barplot(data=avg_version, x="version", y="rating", ax=ax4, palette="rocket")
ax4.set_title("Average Rating by ChatGPT Version")
ax4.tick_params(axis='x', rotation=45)
st.pyplot(fig4)

# --- Predict Sentiment for Custom Review ---
st.markdown("### 🔮 Try It Yourself: Predict Sentiment from Text")
user_input = st.text_area("Enter a review to analyze sentiment:")
if st.button("Predict Sentiment"):
    if user_input.strip():
        pred = model.predict([user_input])[0]
        st.success(f"Predicted Sentiment: **{pred.upper()}**")
    else:
        st.warning("Please enter some text before predicting.")

# Footer
st.markdown("---")
st.caption("Developed by Nandy | Sentiment Analysis Project | Powered by Streamlit + Scikit-Learn")
