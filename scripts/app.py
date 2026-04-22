import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px

# 1. DATABASE CONNECTION
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="sentiment_db",
        user="mlops_user",
        password="mlops_password",
        port="5432"
    )

# 2. STREAMLIT UI SETUP
st.set_page_config(page_title="Reddit MLOps Dashboard", layout="wide")
st.title("📊 Real-Time Reddit Sentiment Dashboard")

# --- SIDEBAR FOR INTERACTIVITY ---
st.sidebar.header("Search Parameters")
# This is the search box you were missing
subreddit_input = st.sidebar.text_input("Enter Subreddit Name", value="technology")

# 3. DATA FETCHING LOGIC
try:
    conn = get_db_connection()
    
    # Query only for the subreddit typed in the sidebar
    # We use ILIKE so 'Technology' matches 'technology'
    query = "SELECT * FROM reddit_sentiment WHERE subreddit ILIKE %s ORDER BY created_at DESC"
    df = pd.read_sql(query, conn, params=(subreddit_input,))
    
    if not df.empty:
        # --- METRICS SECTION ---
        col1, col2, col3 = st.columns(3)
        total_posts = len(df)
        most_common = df['sentiment'].mode()[0]
        pos_ratio = (len(df[df['sentiment'] == 'Positive']) / total_posts) * 100

        col1.metric("Total Posts", total_posts)
        col2.metric("Overall Sentiment", most_common)
        col3.metric("Positive Ratio", f"{pos_ratio:.1f}%")

        # --- VISUALIZATION ---
        st.subheader(f"Sentiment Distribution for r/{subreddit_input}")
        fig = px.pie(df, names='sentiment', 
                     color='sentiment',
                     color_discrete_map={'Positive':'green', 'Neutral':'gray', 'Negative':'red'})
        st.plotly_chart(fig, use_container_width=True)

        # --- DATA TABLE ---
        st.subheader("Recent Analyzed Posts")
        st.dataframe(df[['title', 'sentiment', 'created_at']], use_container_width=True)
        
    else:
        st.warning(f"No data found for r/{subreddit_input}. Is the worker running?")

    conn.close()

except Exception as e:
    st.error(f"Database Connection Error: {e}")

# Add a refresh button to trigger a manual update
if st.sidebar.button('Refresh Data'):
    st.rerun()
