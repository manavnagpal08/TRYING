import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# --- Page Styling ---
st.markdown("""
<style>
.analytics-box {
    padding: 2rem;
    background: rgba(255, 255, 255, 0.96);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    animation: fadeInSlide 0.7s ease-in-out;
    margin-bottom: 2rem;
}
@keyframes fadeInSlide {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
h3 {
    color: #00cec9;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="analytics-box">', unsafe_allow_html=True)
st.markdown("## 📊 Screening Analytics Dashboard")

# --- Load Data ---
df = pd.DataFrame() # Initialize an empty DataFrame

# Prioritize loading from session state
if 'screening_results' in st.session_state and st.session_state['screening_results']:
    try:
        df = pd.DataFrame(st.session_state['screening_results'])
        st.info("✅ Loaded screening results from current session.")
    except Exception as e:
        st.error(f"Error loading results from session state: {e}")
        df = pd.DataFrame() # Reset to empty if error
else:
    # Fallback to results.csv if session state is empty or not set
    data_source = "results.csv"
    if os.path.exists(data_source):
        try:
            df = pd.read_csv(data_source)
            st.info("📁 Loaded existing results from `results.csv` (No session data found).")
        except pd.errors.EmptyDataError:
            st.warning("`results.csv` is empty. No screening data to display yet.")
        except Exception as e:
            st.error(f"Error reading `results.csv`: {e}")
    else:
        st.warning("⚠️ No screening data found in current session or `results.csv`. Please run the screener first.")
        st.stop() # Stop execution if no data is available

# Check if DataFrame is still empty after loading attempts
if df.empty:
    st.info("No data available for analytics. Please screen some resumes first.")
    st.stop()

# --- Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("📈 Avg. Score", f"{df['Score (%)'].mean():.2f}%")
col2.metric("🧓 Avg. Experience", f"{df['Years Experience'].mean():.1f} yrs")
# Assuming a default cutoff of 80 for analytics dashboard if not explicitly passed
shortlisted_count = (df['Score (%)'] >= 80).sum() 
col3.metric("✅ Shortlisted", f"{shortlisted_count}")

st.divider()

# --- Top Candidates ---
st.markdown("### 🥇 Top 5 Candidates by Score")
st.dataframe(df.sort_values(by="Score (%)", ascending=False).head(5)[['File Name', 'Score (%)', 'Years Experience', 'Candidate Name']], use_container_width=True)

# --- WordCloud ---
if 'Matched Keywords' in df.columns and not df['Matched Keywords'].empty:
    st.markdown("### ☁️ Common Skills WordCloud")
    all_keywords = [kw.strip() for kws in df['Matched Keywords'].dropna() for kw in kws.split(',')]
    if all_keywords: # Ensure there are keywords before generating word cloud
        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_keywords))
        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.info("No common skills to display in the WordCloud.")
else:
    st.info("No 'Matched Keywords' data available for WordCloud.")


# --- Missing Skills ---
if 'Missing Skills' in df.columns and not df['Missing Skills'].empty:
    st.markdown("### ❌ Top Missing Skills")
    all_missing = pd.Series([s.strip() for row in df['Missing Skills'].dropna() for s in row.split(',') if s.strip()])
    if not all_missing.empty: # Ensure there are missing skills before plotting
        top_missing = all_missing.value_counts().head(10)
        sns.set_style("whitegrid")
        fig_ms, ax_ms = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top_missing.values, y=top_missing.index, ax=ax_ms, palette="coolwarm")
        ax_ms.set_xlabel("Count")
        ax_ms.set_ylabel("Missing Skill")
        st.pyplot(fig_ms)
    else:
        st.info("No top missing skills to display.")
else:
    st.info("No 'Missing Skills' column found in data or it's empty.")


# --- Score Distribution ---
st.markdown("### 📊 Score Distribution")
fig_hist, ax_hist = plt.subplots()
sns.histplot(df['Score (%)'], bins=10, kde=True, color="#00cec9", ax=ax_hist)
ax_hist.set_xlabel("Score (%)")
ax_hist.set_ylabel("Number of Candidates")
st.pyplot(fig_hist)

# --- Experience Distribution ---
st.markdown("### 💼 Experience Distribution")
fig_exp, ax_exp = plt.subplots()
sns.boxplot(x=df['Years Experience'], color="#fab1a0", ax=ax_exp)
ax_exp.set_xlabel("Years of Experience")
st.pyplot(fig_exp)

st.markdown("</div>", unsafe_allow_html=True)
