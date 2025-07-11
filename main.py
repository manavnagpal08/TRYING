import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from login import login_section
# from email_sender import send_email_to_candidate # This import seems unused and might cause issues if email_sender.py doesn't exist
import os
import json
import pdfplumber
import runpy
import sqlite3 # Import sqlite3 for database interaction

# --- Database Configuration (MUST match screener.py) ---
DATABASE_FILE = "screening_data.db"

# --- Page Config ---
st.set_page_config(page_title="ScreenerPro – AI Hiring Dashboard", layout="wide")


# --- Dark Mode Toggle ---
dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_main")
if dark_mode:
    st.markdown("""
    <style>
    body { background-color: #121212 !important; color: white !important; }
    .block-container { background-color: #1e1e1e !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Global Fonts & UI Styling ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main .block-container {
    padding: 2rem;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 12px 30px rgba(0,0,0,0.1);
    animation: fadeIn 0.8s ease-in-out;
}
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
.dashboard-card {
    padding: 2rem;
    text-align: center;
    font-weight: 600;
    border-radius: 16px;
    background: linear-gradient(145deg, #f1f2f6, #ffffff);
    border: 1px solid #e0e0e0;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.3s ease;
    cursor: pointer;
}
.dashboard-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.1); /* Corrected box-shadow property */
    background: linear-gradient(145deg, #e0f7fa, #f1f1f1);
}
.dashboard-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #222;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #00cec9;
    display: inline-block;
    margin-bottom: 2rem;
    animation: slideInLeft 0.8s ease-out;
}
@keyframes slideInLeft {
    0% { transform: translateX(-40px); opacity: 0; }
    100% { transform: translateX(0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# --- Branding ---
# Check if logo.png exists, otherwise use a placeholder or text
if os.path.exists("logo.png"):
    st.image("logo.png", width=300)
else:
    st.title("ScreenerPro") # Fallback to text title

st.title("🧠 ScreenerPro – AI Hiring Assistant")


# --- Auth ---
if not login_section():
    st.stop()

# --- Database Data Loading for Dashboard ---
@st.cache_data(show_spinner="Loading dashboard data...")
def get_dashboard_data_from_db():
    """Fetches summary statistics for the dashboard from the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        # Get total resumes screened
        c.execute('SELECT COUNT(DISTINCT candidate_name) FROM results')
        resume_count = c.fetchone()[0]

        # Get total job descriptions managed (from 'data' folder, as DB only stores screened JDs)
        jd_count = len([f for f in os.listdir("data") if f.endswith(".txt")]) if os.path.exists("data") else 0

        # Get shortlisted candidates count
        c.execute('SELECT COUNT(DISTINCT candidate_name) FROM results WHERE shortlisted = TRUE')
        shortlisted_count = c.fetchone()[0]

        # Get average score
        c.execute('SELECT AVG(predicted_score) FROM results')
        avg_score = c.fetchone()[0]
        avg_score = avg_score if avg_score is not None else 0.0 # Handle case with no data

        # Get all data for insights if needed
        c.execute('SELECT candidate_name, predicted_score, years_experience, keyword_match FROM results')
        columns = [description[0] for description in c.description]
        all_results = c.fetchall()
        df_results = pd.DataFrame(all_results, columns=columns)
        
        # Rename columns for consistency with previous dashboard logic
        df_results = df_results.rename(columns={
            'predicted_score': 'Score (%)',
            'years_experience': 'Years Experience',
            'keyword_match': 'Matched Keywords' # Assuming 'keyword_match' in DB stores the comma-separated string
        })
        
        # Ensure numeric types
        for col in ['Score (%)', 'Years Experience']:
            if col in df_results.columns:
                df_results[col] = pd.to_numeric(df_results[col], errors='coerce').fillna(0)

        return resume_count, jd_count, shortlisted_count, avg_score, df_results

    except sqlite3.OperationalError as e:
        st.error(f"Database error loading dashboard data: {e}. Ensure the database file is accessible.")
        return 0, 0, 0, 0.0, pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        return 0, 0, 0, 0.0, pd.DataFrame()
    finally:
        if conn:
            conn.close()


# --- Navigation Control ---
default_tab = st.session_state.get("tab_override", "🏠 Dashboard")
tab = st.sidebar.radio("📍 Navigate", [
    "🏠 Dashboard", "🧠 Resume Screener", "📁 Manage JDs", "📊 Screening Analytics",
    "📤 Email Candidates", "🔍 Search Resumes", "📝 Candidate Notes", "🚪 Logout"
], index=[
    "🏠 Dashboard", "🧠 Resume Screener", "📁 Manage JDs", "📊 Screening Analytics",
    "📤 Email Candidates", "🔍 Search Resumes", "📝 Candidate Notes", "🚪 Logout"
].index(default_tab))

if "tab_override" in st.session_state:
    del st.session_state.tab_override

# ======================
# 🏠 Dashboard Section
# ======================
if tab == "🏠 Dashboard":
    st.markdown('<div class="dashboard-header">📊 Overview Dashboard</div>', unsafe_allow_html=True)

    # Load data from database
    resume_count, jd_count, shortlisted, avg_score, df_results = get_dashboard_data_from_db()

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""<div class="dashboard-card">📂 <br><b>{resume_count}</b><br>Resumes Screened</div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="dashboard-card">📝 <br><b>{jd_count}</b><br>Job Descriptions</div>""", unsafe_allow_html=True)
    col3.markdown(f"""<div class="dashboard-card">✅ <br><b>{shortlisted}</b><br>Shortlisted Candidates</div>""", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    col4.markdown(f"""<div class="dashboard-card">📈 <br><b>{avg_score:.1f}%</b><br>Avg Score</div>""", unsafe_allow_html=True)
    with col5:
        if st.button("🧠 Resume Screener", use_container_width=True):
            st.session_state.tab_override = "🧠 Resume Screener"
            st.rerun()
    with col6:
        if st.button("📊 Screening Analytics", use_container_width=True): # Changed to Analytics
            st.session_state.tab_override = "📊 Screening Analytics"
            st.rerun()

    # Optional: Dashboard Insights
    if not df_results.empty:
        try:
            # Ensure 'Years Experience' is numeric before using in cut
            df_results['Years Experience'] = pd.to_numeric(df_results['Years Experience'], errors='coerce').fillna(0)

            df_results['Tag'] = df_results.apply(lambda row:
                "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3
                else "✅ Good Fit" if row['Score (%)'] >= 75
                else "⚠️ Needs Review", axis=1)

            st.markdown("### 📊 Dashboard Insights")

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                st.markdown("##### 🔥 Candidate Distribution")
                pie_data = df_results['Tag'].value_counts().reset_index()
                pie_data.columns = ['Tag', 'Count']
                fig_pie, ax1 = plt.subplots(figsize=(4.5, 4.5))
                ax1.pie(pie_data['Count'], labels=pie_data['Tag'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
                ax1.axis('equal')
                st.pyplot(fig_pie)
                plt.close(fig_pie) # Close the figure

            with col_g2:
                st.markdown("##### 📊 Experience Distribution")
                bins = [0, 2, 5, 10, 20]
                labels = ['0-2 yrs', '3-5 yrs', '6-10 yrs', '10+ yrs']
                # Ensure 'Years Experience' is numeric before cutting
                df_results['Experience Group'] = pd.cut(df_results['Years Experience'], bins=bins, labels=labels, right=False)
                exp_counts = df_results['Experience Group'].value_counts().sort_index()
                fig_bar, ax2 = plt.subplots(figsize=(5, 4))
                sns.barplot(x=exp_counts.index, y=exp_counts.values, palette="coolwarm", ax=ax2)
                ax2.set_ylabel("Candidates")
                ax2.set_xlabel("Experience Range")
                ax2.tick_params(axis='x', labelrotation=0)
                st.pyplot(fig_bar)
                plt.close(fig_bar)
                
            # 📋 Top 5 Most Common Skills - Enhanced & Resized
            st.markdown("##### 🧠 Top 5 Most Common Skills")

            # Check if 'Matched Keywords' column exists and is not empty
            if 'Matched Keywords' in df_results.columns and not df_results['Matched Keywords'].empty:
                all_skills = []
                # Iterate through each entry, split by comma, clean, and add to list
                for skills_str in df_results['Matched Keywords'].dropna():
                    all_skills.extend([s.strip().lower() for s in str(skills_str).split(",") if s.strip()])

                if all_skills: # Check if list is not empty after processing
                    skill_counts = pd.Series(all_skills).value_counts().head(5)

                    fig_skills, ax3 = plt.subplots(figsize=(5.8, 3))
                    sns.barplot(
                        x=skill_counts.values,
                        y=skill_counts.index,
                        palette=sns.color_palette("cool", len(skill_counts)),
                        ax=ax3
                    )
                    ax3.set_title("Top 5 Skills", fontsize=13, fontweight='bold')
                    ax3.set_xlabel("Frequency", fontsize=11)
                    ax3.set_ylabel("Skill", fontsize=11)
                    ax3.tick_params(labelsize=10)
                    for i, v in enumerate(skill_counts.values):
                        ax3.text(v + 0.3, i, str(v), color='black', va='center', fontweight='bold', fontsize=9)

                    fig_skills.tight_layout()
                    st.pyplot(fig_skills)
                    plt.close(fig_skills)
                else:
                    st.info("No skill data available in results for Top 5 Skills.")
            else:
                st.info("No 'Matched Keywords' data available or column not found for Top 5 Skills.")

        except Exception as e:
            st.warning(f"⚠️ Could not render dashboard insights due to data error: {e}")

# ======================
# Page Routing via exec
# ======================
elif tab == "🧠 Resume Screener":
    runpy.run_path("screener.py")

elif tab == "📁 Manage JDs":
    with open("manage_jds.py", encoding="utf-8") as f:
        exec(f.read())

elif tab == "📊 Screening Analytics":
    with open("analytics.py", encoding="utf-8") as f:
        exec(f.read())

elif tab == "📤 Email Candidates":
    with open("email_page.py", encoding="utf-8") as f:
        exec(f.read())

elif tab == "🔍 Search Resumes":
    with open("search.py", encoding="utf-8") as f:
        exec(f.read())

elif tab == "📝 Candidate Notes":
    with open("notes.py", encoding="utf-8") as f:
        exec(f.read())

elif tab == "🚪 Logout":
    st.session_state.authenticated = False
    st.success("✅ Logged out.")
    st.stop()
