import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from transformers import pipeline

# 1. Page Config
st.set_page_config(page_title="Reddit Sentiment MLOps", layout="wide")
st.title("📊 Real-Time Reddit Sentiment Dashboard")

# 2. Connection & Model Setup
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@st.cache_resource
def get_db_connection():
    return psycopg2.connect(
        host="localhost", database="sentiment_db",
        user="mlops_user", password="mlops_password", port="5433"
    )

model = load_model()

# 3. Sidebar - Stats & Controls
st.sidebar.header("Controls")
subreddit_input = st.sidebar.text_input("Enter Subreddit", value="technology")

# 4. Main UI Logic
conn = get_db_connection()
query = f"SELECT * FROM reddit_sentiment WHERE subreddit ILIKE %s"
df = pd.read_sql(query, conn, params=(f'%{subreddit_input}%',))

if not df.empty:
    st.subheader(f"Analysis for r/{subreddit_input}")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    total = len(df)
    pos_pct = len(df[df['sentiment'] == 'Positive']) / total * 100
    
    col1.metric("Total Posts", total)
    col2.metric("Overall Sentiment", df['sentiment'].mode()[0])
    col3.metric("Positive Ratio", f"{pos_pct:.1f}%")

    # Visualizations
    fig_pie = px.pie(df, names='sentiment', title="Sentiment Distribution", color='sentiment',
                    color_discrete_map={'Positive':'green', 'Neutral':'gray', 'Negative':'red'})
    st.plotly_chart(fig_pie, use_container_width=True)

    # Raw Data
    st.write("### Recent Analyzed Posts")
    st.dataframe(df[['title', 'sentiment']].tail(10), use_container_width=True)
else:
    st.warning(f"No data found for r/{subreddit_input}. Start your pipeline.py to ingest data!")
