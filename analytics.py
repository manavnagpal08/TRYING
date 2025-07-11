import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import plotly.express as px
import sqlite3 # Import sqlite3 for database interaction

# --- Database Configuration (MUST match screener.py) ---
DATABASE_FILE = "screening_data.db"

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

# --- Database Data Loading Function ---
@st.cache_data(show_spinner="Loading data from database...")
def get_all_screening_results_from_db():
    """Fetches all screening results from the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute('SELECT * FROM results ORDER BY predicted_score DESC')
        
        columns = [description[0] for description in c.description]
        results = c.fetchall()
        
        df = pd.DataFrame(results, columns=columns)
        
        # Rename columns to match expected names in analytics dashboard
        df = df.rename(columns={
            'candidate_name': 'Candidate Name',
            'predicted_score': 'Score (%)',
            'keyword_match': 'Keyword Match',
            'section_completeness': 'Section Completeness',
            'semantic_similarity': 'Semantic Similarity',
            'length_score': 'Length Score',
            'ai_suggestion': 'AI Suggestion',
            'full_resume_text': 'Resume Raw Text'
            # Add other renames if necessary for columns used in analytics.py
        })
        
        # Convert numeric columns to appropriate types
        for col in ['Score (%)', 'Keyword Match', 'Section Completeness', 'Semantic Similarity', 'Length Score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Years Experience might need specific handling if not directly in DB or needs calculation
        # Assuming 'years_experience' is not directly stored but derived if needed, or if it was stored as 'years_exp'
        # For now, let's assume 'Years Experience' is a column if it was stored from screener.py.
        # If not, you might need to add a function here to extract it from resume_text if necessary for analytics.
        # For simplicity, let's assume it's already present or will be added from the DB.
        # If 'years_exp' is the DB column, rename it:
        if 'years_exp' in df.columns:
            df = df.rename(columns={'years_exp': 'Years Experience'})
        elif 'years_experience' not in df.columns: # Fallback if neither is present
             st.warning("Warning: 'Years Experience' column not found in database. Some analytics may be limited.")
             df['Years Experience'] = 0.0 # Placeholder to prevent errors

        st.info("✅ Loaded screening results from database.")
        return df
    except sqlite3.OperationalError as e:
        st.error(f"Database error: {e}. Ensure the database file is accessible and not locked.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

df = get_all_screening_results_from_db()

# Check if DataFrame is still empty after loading attempts
if df.empty:
    st.info("No data available for analytics. Please screen some resumes first in the main app.")
    st.stop()

# --- Essential Column Check ---
# Updated essential columns to reflect what's expected from the DB and used in analytics
essential_core_columns = ['Score (%)', 'Years Experience', 'Candidate Name']

missing_essential_columns = [col for col in essential_core_columns if col not in df.columns]

if missing_essential_columns:
    st.error(f"Error: The loaded data is missing essential core columns: {', '.join(missing_essential_columns)}."
             " Please ensure your screening process generates at least these required data fields and they are correctly mapped/renamed.")
    st.stop()

# --- Filters Section ---
st.markdown("### 🔍 Filter Results")
filter_cols = st.columns(3)

with filter_cols[0]:
    # Ensure Score (%) is numeric before finding min/max
    df['Score (%)'] = pd.to_numeric(df['Score (%)'], errors='coerce').fillna(0)
    min_score, max_score = float(df['Score (%)'].min()), float(df['Score (%)'].max())
    score_range = st.slider(
        "Filter by Score (%)",
        min_value=min_score,
        max_value=max_score,
        value=(min_score, max_score),
        step=1.0,
        key="score_filter"
    )

with filter_cols[1]:
    # Ensure Years Experience is numeric before finding min/max
    df['Years Experience'] = pd.to_numeric(df['Years Experience'], errors='coerce').fillna(0)
    min_exp, max_exp = float(df['Years Experience'].min()), float(df['Years Experience'].max())
    exp_range = st.slider(
        "Filter by Years Experience",
        min_value=min_exp,
        max_value=max_exp,
        value=(min_exp, max_exp),
        step=0.5,
        key="exp_filter"
    )

with filter_cols[2]:
    shortlist_threshold = st.slider(
        "Set Shortlisting Cutoff Score (%)",
        min_value=0,
        max_value=100,
        value=80,
        step=1,
        key="shortlist_filter"
    )

# Apply filters
filtered_df = df[
    (df['Score (%)'] >= score_range[0]) & (df['Score (%)'] <= score_range[1]) &
    (df['Years Experience'] >= exp_range[0]) & (df['Years Experience'] <= exp_range[1])
].copy()

if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust your criteria.")
    st.stop()

# Add Shortlisted/Not Shortlisted column to filtered_df for plotting
# Use the 'shortlisted' column from the DB if available, otherwise apply threshold
if 'shortlisted' in filtered_df.columns:
    filtered_df['Shortlisted'] = filtered_df['shortlisted'].apply(lambda x: "Yes (Shortlisted in App)" if x else "No (Not Shortlisted in App)")
else:
    filtered_df['Shortlisted'] = filtered_df['Score (%)'].apply(lambda x: f"Yes (Score >= {shortlist_threshold}%)" if x >= shortlist_threshold else "No")


# --- Metrics ---
st.markdown("### 📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg. Score", f"{filtered_df['Score (%)'].mean():.2f}%")
col2.metric("Avg. Experience", f"{filtered_df['Years Experience'].mean():.1f} yrs")
col3.metric("Total Candidates", f"{len(filtered_df)}")
# Calculate shortlisted count based on the 'Shortlisted' column, which now reflects DB status or threshold
shortlisted_count_filtered = filtered_df[filtered_df['Shortlisted'].str.contains("Yes")].shape[0]
col4.metric("Shortlisted", f"{shortlisted_count_filtered}")

st.divider()

# --- Detailed Candidate Table ---
st.markdown("### 📋 Filtered Candidates List")
# Ensure column names match the renamed ones from the DB
display_cols_for_table = ['Candidate Name', 'Score (%)', 'Years Experience', 'Shortlisted']

if 'Keyword Match' in filtered_df.columns:
    display_cols_for_table.append('Keyword Match')
if 'Section Completeness' in filtered_df.columns:
    display_cols_for_table.append('Section Completeness')
if 'Semantic Similarity' in filtered_df.columns:
    display_cols_for_table.append('Semantic Similarity')
if 'Length Score' in filtered_df.columns:
    display_cols_for_table.append('Length Score')
if 'AI Suggestion' in filtered_df.columns:
    display_cols_for_table.append('AI Suggestion')
if 'Matched Keywords' in filtered_df.columns:
    display_cols_for_table.append('Matched Keywords')
if 'Missing Skills' in filtered_df.columns:
    display_cols_for_table.append('Missing Skills')
if 'File Name' in filtered_df.columns: # Add back if you want to see original file name
    display_cols_for_table.insert(0, 'File Name')


st.dataframe(
    filtered_df[display_cols_for_table].sort_values(by="Score (%)", ascending=False),
    use_container_width=True
)

# --- Download Filtered Data ---
@st.cache_data
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
    plt.close(fig_hist)

with tab2:
    st.markdown("#### Experience Distribution")
    fig_exp, ax_exp = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_df['Years Experience'], bins=5, kde=True, color="#fab1a0", ax=ax_exp)
    ax_exp.set_xlabel("Years of Experience")
    ax_exp.set_ylabel("Number of Candidates")
    st.pyplot(fig_exp)
    plt.close(fig_exp)

with tab3:
    st.markdown("#### Shortlist Breakdown")
    shortlist_counts = filtered_df['Shortlisted'].value_counts()
    if not shortlist_counts.empty:
        fig_pie = px.pie(
            names=shortlist_counts.index,
            values=shortlist_counts.values,
            title=f"Candidates Shortlisted vs. Not Shortlisted", # Removed cutoff from title as it might be from DB
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Not enough data to generate Shortlist Breakdown.")

with tab4:
    st.markdown("#### Score vs. Years Experience")
    # Ensure 'Shortlisted' column is used for color, with appropriate mapping
    color_map = {
        "Yes (Shortlisted in App)": "green",
        f"Yes (Score >= {shortlist_threshold}%)": "green", # For cases where DB 'shortlisted' is not used
        "No (Not Shortlisted in App)": "red",
        "No": "red"
    }
    fig_scatter = px.scatter(
        filtered_df,
        x="Years Experience",
        y="Score (%)",
        hover_name="Candidate Name",
        color="Shortlisted",
        title="Candidate Score vs. Years Experience",
        labels={"Years Experience": "Years of Experience", "Score (%)": "Matching Score (%)"},
        trendline="ols",
        color_discrete_map=color_map
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


with tab5:
    col_wc1, col_wc2 = st.columns(2)
    with col_wc1:
        st.markdown("#### ☁️ Common Skills WordCloud")
        # Assuming 'Matched Keywords' is stored as a comma-separated string in the DB
        if 'Matched Keywords' in filtered_df.columns and not filtered_df['Matched Keywords'].empty:
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
                plt.close(fig_wc)
            else:
                st.info("No common skills to display in the WordCloud for filtered data.")
        else:
            st.info("No 'Matched Keywords' data available or column not found for WordCloud.")
    
    with col_wc2:
        st.markdown("#### ❌ Top Missing Skills")
        # Assuming 'Missing Skills' is stored as a comma-separated string in the DB
        if 'Missing Skills' in filtered_df.columns and not filtered_df['Missing Skills'].empty:
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
                plt.close(fig_ms)
            else:
                st.info("No top missing skills to display for filtered data.")
        else:
            st.info("No 'Missing Skills' data available or column not found.")

st.markdown("</div>", unsafe_allow_html=True)
