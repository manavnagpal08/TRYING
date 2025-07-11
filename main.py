import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from login import login_section
# from email_sender import send_email_to_candidate # This import is not used in the provided snippet
import os
# import json # This import is not used in the provided snippet
import pdfplumber
import runpy


# --- Page Config ---
st.set_page_config(page_title="ScreenerPro – AI Hiring Dashboard", layout="wide", initial_sidebar_state="expanded")


# --- Global Fonts & UI Styling ---
# Using Inter font and comprehensive styling for a modern look
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    /* Base font for the entire app */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #333333; /* Default text color for light mode */
    }

    /* Main container styling for a clean, elevated look */
    .main .block-container {
        padding: 2.5rem 3rem; /* Increased padding for more breathing room */
        border-radius: 25px; /* More rounded corners */
        background: linear-gradient(180deg, #ffffff, #f0f2f6); /* Subtle gradient background */
        box-shadow: 0 15px 40px rgba(0,0,0,0.1); /* Stronger, softer shadow */
        animation: fadeIn 0.9s ease-in-out; /* Slightly longer fade-in */
    }

    /* Keyframe for fade-in animation */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Dashboard card styling for metrics */
    .dashboard-card {
        padding: 1.8rem; /* Adjusted padding */
        text-align: center;
        font-weight: 600;
        border-radius: 20px; /* More rounded */
        background: linear-gradient(145deg, #e0f7fa, #f1f2f6); /* Light blue-ish gradient */
        border: 1px solid #d0e0e0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08); /* Enhanced shadow */
        transition: transform 0.3s ease, box-shadow 0.4s ease; /* Smoother transitions */
        cursor: pointer;
        color: #004d40; /* Darker text for better contrast */
        margin-bottom: 1.5rem; /* Space between rows of cards */
    }
    .dashboard-card:hover {
        transform: translateY(-8px); /* More pronounced lift on hover */
        box-shadow: 0 15px 35px rgba(0,0,0,0.15); /* Darker shadow on hover */
        background: linear-gradient(145deg, #b2ebf2, #e0f7fa); /* More vibrant hover gradient */
    }
    .dashboard-card b {
        font-size: 2.5rem; /* Larger numbers */
        color: #00796b; /* Accent color for numbers */
        display: block;
        margin-bottom: 0.5rem;
    }
    .dashboard-card span { /* For the icon */
        font-size: 1.8rem;
        display: block;
        margin-bottom: 0.5rem;
    }

    /* Main dashboard header styling */
    .dashboard-header {
        font-size: 2.5rem; /* Larger header */
        font-weight: 700;
        color: #1a1a1a; /* Very dark grey for strong contrast */
        padding-bottom: 0.8rem; /* More padding */
        border-bottom: 4px solid #00cec9; /* Thicker accent line */
        display: inline-block;
        margin-bottom: 2.5rem; /* More space below header */
        animation: slideInLeft 0.9s ease-out; /* Slightly longer animation */
    }
    /* Keyframe for slide-in animation */
    @keyframes slideInLeft {
        0% { transform: translateX(-60px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }

    /* Streamlit button styling */
    .stButton>button {
        background-color: #00cec9; /* Accent color for buttons */
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 206, 201, 0.3);
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00b3a8; /* Darker on hover */
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 206, 201, 0.4);
    }

    /* Streamlit sidebar styling */
    .css-1d391kg, .css-1lcbmhc { /* Targeting sidebar background */
        background-color: #f0f2f6; /* Light grey background for sidebar */
        border-right: 1px solid #e0e0e0;
        box-shadow: 5px 0 15px rgba(0,0,0,0.05);
    }
    .css-1d391kg .stRadio > label { /* Sidebar radio button labels */
        font-weight: 600;
        color: #333333;
    }
    .css-1d391kg .stRadio > label > div { /* Sidebar radio button icons/text */
        padding: 0.8rem 1rem;
        border-radius: 10px;
        transition: background-color 0.2s ease, color 0.2s ease;
    }
    .css-1d391kg .stRadio > label > div:hover {
        background-color: #e0f7fa;
        color: #00796b;
    }
    .css-1d391kg .stRadio > label[data-baseweb="radio"] > div[aria-selected="true"] {
        background-color: #00cec9 !important;
        color: white !important;
        font-weight: 700;
    }
    .css-1d391kg .stRadio > label[data-baseweb="radio"] > div[aria-selected="true"] svg {
        color: white !important;
    }

    /* Dark Mode Specific Styles */
    body.dark-mode-active {
        background-color: #1a1a1a !important; /* Darker background */
        color: #f0f0f0 !important; /* Lighter text */
    }
    body.dark-mode-active .main .block-container {
        background: linear-gradient(180deg, #2a2a2a, #3a3a3a); /* Darker gradient */
        box-shadow: 0 15px 40px rgba(0,0,0,0.3); /* Darker shadow */
        color: #f0f0f0;
    }
    body.dark-mode-active .dashboard-card {
        background: linear-gradient(145deg, #3a3a3a, #4a4a4a); /* Darker card background */
        border: 1px solid #555555;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        color: #e0e0e0;
    }
    body.dark-mode-active .dashboard-card:hover {
        background: linear-gradient(145deg, #4a4a4a, #5a5a5a);
        box-shadow: 0 15px 35px rgba(0,0,0,0.6);
    }
    body.dark-mode-active .dashboard-card b {
        color: #00cec9; /* Keep accent color for numbers */
    }
    body.dark-mode-active .dashboard-header {
        color: #f0f0f0;
        border-bottom-color: #00cec9;
    }
    body.dark-mode-active .stButton>button {
        background-color: #00cec9;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 206, 201, 0.3);
    }
    body.dark-mode-active .stButton>button:hover {
        background-color: #00b3a8;
        box-shadow: 0 6px 20px rgba(0, 206, 201, 0.4);
    }
    body.dark-mode-active .css-1d391kg, body.dark-mode-active .css-1lcbmhc {
        background-color: #2a2a2a;
        border-right: 1px solid #444444;
        box-shadow: 5px 0 15px rgba(0,0,0,0.2);
    }
    body.dark-mode-active .css-1d391kg .stRadio > label {
        color: #f0f0f0;
    }
    body.dark-mode-active .css-1d391kg .stRadio > label > div:hover {
        background-color: #3a3a3a;
        color: #00cec9;
    }
    body.dark-mode-active .stDataFrame { /* Dark mode for dataframes */
        background-color: #333333;
        color: #f0f0f0;
        border-radius: 10px;
    }
    body.dark-mode-active .stDataFrame .header {
        background-color: #444444;
        color: #f0f0f0;
    }
    body.dark-mode-active .stDataFrame .data-row {
        background-color: #333333;
        color: #f0f0f0;
    }
    body.dark-mode-active .stDataFrame .data-row:nth-child(even) {
        background-color: #3a3a3a;
    }
    body.dark-mode-active .stAlert { /* Dark mode for alerts */
        background-color: #333333;
        color: #f0f0f0;
        border-radius: 10px;
    }
    body.dark-mode-active .stAlert.success {
        background-color: #1a4a3a; /* Darker green */
        color: #d0f0e0;
    }
    body.dark-mode-active .stAlert.warning {
        background-color: #5a4a1a; /* Darker yellow */
        color: #f0e0b0;
    }
    body.dark-mode-active .stAlert.info {
        background-color: #1a3a5a; /* Darker blue */
        color: #b0e0f0;
    }
    body.dark-mode-active .stAlert.error {
        background-color: #5a1a1a; /* Darker red */
        color: #f0b0b0;
    }
</style>
""", unsafe_allow_html=True)

# --- Dark Mode Toggle ---
dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_main")
if dark_mode:
    st.markdown("<script>document.body.classList.add('dark-mode-active');</script>", unsafe_allow_html=True)
else:
    st.markdown("<script>document.body.classList.remove('dark-mode-active');</script>", unsafe_allow_html=True)


# --- Branding ---
st.image("https://placehold.co/300x100/00cec9/ffffff?text=ScreenerPro+Logo", width=300) # Placeholder logo
st.markdown('<h1 class="dashboard-header">🧠 ScreenerPro – AI Hiring Assistant</h1>', unsafe_allow_html=True)


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
    if 'screening_results_df' in st.session_state and not st.session_state['screening_results_df'].empty:
        try:
            df_results = st.session_state['screening_results_df'].copy()
            resume_count = df_results["File Name"].nunique() # Count unique resumes screened
            
            # Define cutoff for shortlisted candidates (consistent with streamlit_app.py)
            # Make sure these values match the sliders in streamlit_app.py for consistency
            # Assuming 'shortlisted' column is set in screener.py based on these cutoffs
            # If 'shortlisted' column is directly set in screener.py, these cutoffs might not be needed here.
            # For consistency, we'll use the 'shortlisted' boolean column if it exists.
            
            if 'shortlisted' in df_results.columns:
                shortlisted = df_results[df_results['shortlisted'] == True].shape[0]
            else:
                # Fallback if 'shortlisted' column is not explicitly set
                cutoff_score = 80 # Example cutoff
                min_exp_required = 2 # Example cutoff
                shortlisted = df_results[(df_results["Score (%)"] >= cutoff_score) & 
                                         (df_results["Years Experience"] >= min_exp_required)].shape[0]

            avg_score = df_results["Score (%)"].mean()
        except Exception as e:
            st.error(f"Error processing screening results from session state: {e}")
            df_results = pd.DataFrame() # Reset df_results if error occurs
    else:
        st.info("No screening results available in this session yet. Please run the Resume Screener.")


    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""<div class="dashboard-card"><span>📂</span> <br><b>{resume_count}</b><br>Resumes Screened</div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="dashboard-card"><span>📝</span> <br><b>{jd_count}</b><br>Job Descriptions</div>""", unsafe_allow_html=True)
    col3.markdown(f"""<div class="dashboard-card"><span>✅</span> <br><b>{shortlisted}</b><br>Shortlisted Candidates</div>""", unsafe_allow_html=True)

    st.markdown("---") # Visual separator

    col4, col5, col6 = st.columns(3)
    col4.markdown(f"""<div class="dashboard-card"><span>📈</span> <br><b>{avg_score:.1f}%</b><br>Avg Score</div>""", unsafe_allow_html=True)
    with col5:
        if st.button("🧠 Go to Resume Screener", use_container_width=True): # More descriptive button text
            st.session_state.tab_override = "🧠 Resume Screener"
            st.rerun()
    with col6:
        if st.button("📤 Go to Email Candidates", use_container_width=True): # More descriptive button text
            st.session_state.tab_override = "📤 Email Candidates"
            st.rerun()

    # Optional: Dashboard Insights
    if not df_results.empty: # Use df_results loaded from session state
        try:
            # Ensure 'Tag' column is created only if it doesn't exist to avoid re-calculation issues
            if 'Tag' not in df_results.columns:
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
                # Adjust text color for pie chart labels based on dark mode
                text_color = 'white' if dark_mode else 'black'
                ax1.pie(pie_data['Count'], labels=pie_data['Tag'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'color': text_color})
                ax1.axis('equal')
                fig_pie.patch.set_facecolor('none') # Make background transparent for dark mode
                ax1.set_facecolor('none') # Make axes background transparent
                st.pyplot(fig_pie)

            with col_g2:
                st.markdown("##### 📊 Experience Distribution")
                bins = [0, 2, 5, 10, 20, 100] # Added an upper bound for the last bin
                labels = ['0-2 yrs', '3-5 yrs', '6-10 yrs', '11-20 yrs', '20+ yrs'] # Adjusted labels
                df_results['Experience Group'] = pd.cut(df_results['Years Experience'], bins=bins, labels=labels, right=False)
                exp_counts = df_results['Experience Group'].value_counts().sort_index()
                fig_bar, ax2 = plt.subplots(figsize=(5.8, 4)) # Adjusted size for better readability
                
                # Adjust bar plot colors and text for dark mode
                bar_color_palette = "viridis" if dark_mode else "coolwarm"
                sns.barplot(x=exp_counts.index, y=exp_counts.values, palette=bar_color_palette, ax=ax2)
                
                ax2.set_ylabel("Candidates", color=text_color)
                ax2.set_xlabel("Experience Range", color=text_color)
                ax2.tick_params(axis='x', labelrotation=0, colors=text_color)
                ax2.tick_params(axis='y', colors=text_color)
                ax2.set_title("Experience Distribution", fontsize=13, fontweight='bold', color=text_color)
                fig_bar.patch.set_facecolor('none') # Make background transparent for dark mode
                ax2.set_facecolor('none') # Make axes background transparent
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
                ax3.set_title("Top 5 Skills", fontsize=13, fontweight='bold', color=text_color)
                ax3.set_xlabel("Frequency", fontsize=11, color=text_color)
                ax3.set_ylabel("Skill", fontsize=11, color=text_color)
                ax3.tick_params(labelsize=10, colors=text_color)
                for i, v in enumerate(skill_counts.values):
                    ax3.text(v + 0.3, i, str(v), color=text_color, va='center', fontweight='bold', fontsize=9)

                fig_skills.tight_layout()
                fig_skills.patch.set_facecolor('none') # Make background transparent for dark mode
                ax3.set_facecolor('none') # Make axes background transparent
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
