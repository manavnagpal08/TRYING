import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from login import login_section
from email_sender import send_email_to_candidate
import os
import json
import pdfplumber
import runpy


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
    box_shadow: 0 10px 24px rgba(0,0,0,0.1);
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
st.image("logo.png", width=300)
st.title("🧠 ScreenerPro – AI Hiring Assistant")


# --- Auth ---
if not login_section():
    st.stop()

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

    # Initialize metrics
    resume_count = 0
    jd_count = len([f for f in os.listdir("data") if f.endswith(".txt")]) if os.path.exists("data") else 0
    shortlisted = 0
    avg_score = 0.0
    df_results = pd.DataFrame() # Initialize empty DataFrame

    # Load results from session state instead of results.csv
    if 'screening_results' in st.session_state and st.session_state['screening_results']:
        try:
            df_results = pd.DataFrame(st.session_state['screening_results'])
            resume_count = df_results["File Name"].nunique() # Count unique resumes screened
            
            # Define cutoff for shortlisted candidates (consistent with streamlit_app.py)
            # Make sure these values match the sliders in streamlit_app.py for consistency
            cutoff_score = 80 
            min_exp_required = 2

            shortlisted = df_results[(df_results["Score (%)"] >= cutoff_score) & 
                                     (df_results["Years Experience"] >= min_exp_required)].shape[0]
            avg_score = df_results["Score (%)"].mean()
        except Exception as e:
            st.error(f"Error processing screening results from session state: {e}")
            df_results = pd.DataFrame() # Reset df_results if error occurs
    else:
        st.info("No screening results available in this session yet. Please run the Resume Screener.")


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
        if st.button("📤 Email Candidates", use_container_width=True):
            st.session_state.tab_override = "📤 Email Candidates"
            st.rerun()

    # Optional: Dashboard Insights
    if not df_results.empty: # Use df_results loaded from session state
        try:
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

            with col_g2:
                st.markdown("##### 📊 Experience Distribution")
                bins = [0, 2, 5, 10, 20]
                labels = ['0-2 yrs', '3-5 yrs', '6-10 yrs', '10+ yrs']
                df_results['Experience Group'] = pd.cut(df_results['Years Experience'], bins=bins, labels=labels, right=False)
                exp_counts = df_results['Experience Group'].value_counts().sort_index()
                fig_bar, ax2 = plt.subplots(figsize=(5, 4))
                sns.barplot(x=exp_counts.index, y=exp_counts.values, palette="coolwarm", ax=ax2)
                ax2.set_ylabel("Candidates")
                ax2.set_xlabel("Experience Range")
                ax2.tick_params(axis='x', labelrotation=0)
                st.pyplot(fig_bar)
            
            # 📋 Top 5 Most Common Skills - Enhanced & Resized
            st.markdown("##### 🧠 Top 5 Most Common Skills")

            if 'Matched Keywords' in df_results.columns: # Use df_results
                all_skills = []
                for skills in df_results['Matched Keywords'].dropna(): # Use df_results
                    all_skills.extend([s.strip().lower() for s in skills.split(",") if s.strip()])

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

            else:
                st.info("No skill data available in results.")

        except Exception as e: # Catch specific exceptions or log for debugging
            st.warning(f"⚠️ Could not render insights due to data error: {e}")

# ======================
# Page Routing via exec
# ======================
elif tab == "🧠 Resume Screener":
    runpy.run_path("screener.py") # Changed to screener.py

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
