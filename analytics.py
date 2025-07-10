import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import plotly.express as px # For more interactive plots

# --- Page Styling ---
st.set_page_config(layout="wide", page_title="Resume Screening Analytics")

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
.stMetric {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
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
            df = pd.DataFrame() # Ensure df is empty on empty file
        except Exception as e:
            st.error(f"Error reading `results.csv`: {e}")
            df = pd.DataFrame() # Reset to empty if error
    else:
        st.warning("⚠️ No screening data found in current session or `results.csv`. Please run the screener first.")
        df = pd.DataFrame() # Ensure df is empty if file not found

# Check if DataFrame is still empty after loading attempts
if df.empty:
    st.info("No data available for analytics. Please screen some resumes first.")
    st.stop() # Stop execution if no data is available


# --- Filters Section ---
st.sidebar.header("🎯 Filter Results")
min_score, max_score = float(df['Score (%)'].min()), float(df['Score (%)'].max())
score_range = st.sidebar.slider(
    "Filter by Score (%)",
    min_value=min_score,
    max_value=max_score,
    value=(min_score, max_score),
    step=1.0
)

min_exp, max_exp = float(df['Years Experience'].min()), float(df['Years Experience'].max())
exp_range = st.sidebar.slider(
    "Filter by Years Experience",
    min_value=min_exp,
    max_value=max_exp,
    value=(min_exp, max_exp),
    step=0.5
)

shortlist_threshold = st.sidebar.slider(
    "Set Shortlisting Cutoff Score (%)",
    min_value=0,
    max_value=100,
    value=80, # Default cutoff
    step=1
)

# Apply filters
filtered_df = df[
    (df['Score (%)'] >= score_range[0]) & (df['Score (%)'] <= score_range[1]) &
    (df['Years Experience'] >= exp_range[0]) & (df['Years Experience'] <= exp_range[1])
].copy() # Use .copy() to avoid SettingWithCopyWarning

if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust your criteria.")
    st.stop()

# Add Shortlisted/Not Shortlisted column to filtered_df for plotting
filtered_df['Shortlisted'] = filtered_df['Score (%)'].apply(lambda x: f"Yes (Score >= {shortlist_threshold}%)" if x >= shortlist_threshold else "No")


# --- Metrics ---
st.markdown("### 📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg. Score", f"{filtered_df['Score (%)'].mean():.2f}%")
col2.metric("Avg. Experience", f"{filtered_df['Years Experience'].mean():.1f} yrs")
col3.metric("Total Candidates", f"{len(filtered_df)}")
shortlisted_count_filtered = (filtered_df['Score (%)'] >= shortlist_threshold).sum()
col4.metric("Shortlisted", f"{shortlisted_count_filtered}")

st.divider()

# --- Detailed Candidate Table ---
st.markdown("### 📋 Filtered Candidates List")
st.dataframe(
    filtered_df[['File Name', 'Candidate Name', 'Score (%)', 'Years Experience', 'Shortlisted', 'Matched Keywords', 'Missing Skills', 'AI Suggestion']]
    .sort_values(by="Score (%)", ascending=False),
    use_container_width=True
)

# --- Download Filtered Data ---
@st.cache_data # Cache this function to avoid re-running on every interaction
def convert_df_to_csv(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(filtered_df)
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_screening_results.csv",
    mime="text/csv",
    help="Download the data currently displayed in the table above."
)

st.divider()

# --- Visualizations ---
st.markdown("### 📊 Visualizations")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Score Distribution", "Experience Distribution", "Shortlist Breakdown", "Score vs. Experience", "Skill Clouds"])

with tab1:
    st.markdown("#### Score Distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_df['Score (%)'], bins=10, kde=True, color="#00cec9", ax=ax_hist)
    ax_hist.set_xlabel("Score (%)")
    ax_hist.set_ylabel("Number of Candidates")
    st.pyplot(fig_hist)

with tab2:
    st.markdown("#### Experience Distribution")
    fig_exp, ax_exp = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_df['Years Experience'], bins=5, kde=True, color="#fab1a0", ax=ax_exp)
    ax_exp.set_xlabel("Years of Experience")
    ax_exp.set_ylabel("Number of Candidates")
    st.pyplot(fig_exp)

with tab3:
    st.markdown("#### Shortlist Breakdown")
    shortlist_counts = filtered_df['Shortlisted'].value_counts()
    if not shortlist_counts.empty:
        fig_pie = px.pie(
            names=shortlist_counts.index,
            values=shortlist_counts.values,
            title=f"Candidates Shortlisted vs. Not Shortlisted (Cutoff: {shortlist_threshold}%)",
            color_discrete_sequence=px.colors.qualitative.Pastel # Use a nice color sequence
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Not enough data to generate Shortlist Breakdown.")

with tab4:
    st.markdown("#### Score vs. Years Experience")
    fig_scatter = px.scatter(
        filtered_df,
        x="Years Experience",
        y="Score (%)",
        hover_name="Candidate Name", # Show candidate name on hover
        color="Shortlisted", # Color points by shortlisted status
        title="Candidate Score vs. Years Experience",
        labels={"Years Experience": "Years of Experience", "Score (%)": "Matching Score (%)"},
        trendline="ols", # Add a linear regression trendline
        color_discrete_map={"Yes (Score >= {})".format(shortlist_threshold): "green", "No": "red"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


with tab5:
    col_wc1, col_wc2 = st.columns(2)
    with col_wc1:
        st.markdown("#### ☁️ Common Skills WordCloud")
        if 'Matched Keywords' in filtered_df.columns and not filtered_df['Matched Keywords'].empty:
            # Flatten list of lists for keywords, handle NaN and empty strings
            all_keywords = [
                kw.strip() for kws in filtered_df['Matched Keywords'].dropna()
                for kw in str(kws).split(',') if kw.strip()
            ]
            if all_keywords:
                wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_keywords))
                fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
            else:
                st.info("No common skills to display in the WordCloud for filtered data.")
        else:
            st.info("No 'Matched Keywords' data available for WordCloud.")
    
    with col_wc2:
        st.markdown("#### ❌ Top Missing Skills")
        if 'Missing Skills' in filtered_df.columns and not filtered_df['Missing Skills'].empty:
            # Flatten list of lists for missing skills, handle NaN and empty strings
            all_missing = pd.Series([
                s.strip() for row in filtered_df['Missing Skills'].dropna()
                for s in str(row).split(',') if s.strip()
            ])
            if not all_missing.empty:
                top_missing = all_missing.value_counts().head(10)
                sns.set_style("whitegrid")
                fig_ms, ax_ms = plt.subplots(figsize=(8, 4))
                sns.barplot(x=top_missing.values, y=top_missing.index, ax=ax_ms, palette="coolwarm")
                ax_ms.set_xlabel("Count")
                ax_ms.set_ylabel("Missing Skill")
                st.pyplot(fig_ms)
            else:
                st.info("No top missing skills to display for filtered data.")
        else:
            st.info("No 'Missing Skills' column found in data or it's empty.")

st.markdown("</div>", unsafe_allow_html=True)
