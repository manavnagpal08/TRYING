import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
import sklearn
import joblib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
import nltk
import collections
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse # For encoding mailto links

# For Generative AI (Google Gemini Pro) - COMMENTED OUT AS PER USER REQUEST
# import google.generativeai as genai

# --- Configure Google Gemini API Key --- - COMMENTED OUT AS PER USER REQUEST
# Store your API key securely in Streamlit Secrets.
# Create a .streamlit/secrets.toml file in your app's directory:
# GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
# try:
#     genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
# except AttributeError:
#     st.error("🚨 Google API Key not found in Streamlit Secrets. Please add it to your .streamlit/secrets.toml file.")
#     st.stop() # Stop the app if API key is missing

# Download NLTK stopwords data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Load Embedding + ML Model ---
@st.cache_resource
def load_ml_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ml_model = joblib.load("ml_screening_model.pkl")
        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory.")
        return None, None

model, ml_model = load_ml_model()

# --- Stop Words List (Using NLTK) ---
NLTK_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
CUSTOM_STOP_WORDS = set([
    "work", "experience", "years", "year", "months", "month", "day", "days", "project", "projects",
    "team", "teams", "developed", "managed", "led", "created", "implemented", "designed",
    "responsible", "proficient", "knowledge", "ability", "strong", "proven", "demonstrated",
    "solution", "solutions", "system", "systems", "platform", "platforms", "framework", "frameworks",
    "database", "databases", "server", "servers", "cloud", "computing", "machine", "learning",
    "artificial", "intelligence", "api", "apis", "rest", "graphql", "agile", "scrum", "kanban",
    "devops", "ci", "cd", "testing", "qa",
    "security", "network", "networking", "virtualization",
    "containerization", "docker", "kubernetes", "git", "github", "gitlab", "bitbucket", "jira",
    "confluence", "slack", "microsoft", "google", "amazon", "azure", "oracle", "sap", "crm", "erp",
    "salesforce", "servicenow", "tableau", "powerbi", "qlikview", "excel", "word", "powerpoint",
    "outlook", "visio", "html", "css", "js", "web", "data", "science", "analytics", "engineer",
    "software", "developer", "analyst", "business", "management", "reporting", "analysis", "tools",
    "python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "swift", "kotlin", "r",
    "sql", "nosql", "linux", "unix", "windows", "macos", "ios", "android", "mobile", "desktop",
    "application", "applications", "frontend", "backend", "fullstack", "ui", "ux", "design",
    "architecture", "architect", "engineering", "scientist", "specialist", "consultant",
    "associate", "senior", "junior", "lead", "principal", "director", "manager", "head", "chief",
    "officer", "president", "vice", "executive", "ceo", "cto", "cfo", "coo", "hr", "human",
    "resources", "recruitment", "talent", "acquisition", "onboarding", "training", "development",
    "performance", "compensation", "benefits", "payroll", "compliance", "legal", "finance",
    "accounting", "auditing", "tax", "budgeting", "forecasting", "investments", "marketing",
    "sales", "customer", "service", "support", "operations", "supply", "chain", "logistics",
    "procurement", "manufacturing", "production", "quality", "assurance", "control", "research",
    "innovation", "product", "program", "portfolio", "governance", "risk", "communication",
    "presentation", "negotiation", "problem", "solving", "critical", "thinking", "analytical",
    "creativity", "adaptability", "flexibility", "teamwork", "collaboration", "interpersonal",
    "organizational", "time", "multitasking", "detail", "oriented", "independent", "proactive",
    "self", "starter", "results", "driven", "client", "facing", "stakeholder", "engagement",
    "vendor", "budget", "cost", "reduction", "process", "improvement", "standardization",
    "optimization", "automation", "digital", "transformation", "change", "methodologies",
    "industry", "regulations", "regulatory", "documentation", "technical", "writing",
    "dashboards", "visualizations", "workshops", "feedback", "reviews", "appraisals",
    "offboarding", "employee", "relations", "diversity", "inclusion", "equity", "belonging",
    "corporate", "social", "responsibility", "csr", "sustainability", "environmental", "esg",
    "ethics", "integrity", "professionalism", "confidentiality", "discretion", "accuracy",
    "precision", "efficiency", "effectiveness", "scalability", "robustness", "reliability",
    "vulnerability", "assessment", "penetration", "incident", "response", "disaster",
    "recovery", "continuity", "bcp", "drp", "gdpr", "hipaa", "soc2", "iso", "nist", "pci",
    "dss", "ccpa", "privacy", "protection", "grc", "cybersecurity", "information", "infosec",
    "threat", "intelligence", "soc", "event", "siem", "identity", "access", "iam", "privileged",
    "pam", "multi", "factor", "authentication", "mfa", "single", "sign", "on", "sso",
    "encryption", "decryption", "firewall", "ids", "ips", "vpn", "endpoint", "antivirus",
    "malware", "detection", "forensics", "handling", "assessments", "policies", "procedures",
    "guidelines", "mitre", "att&ck", "modeling", "secure", "lifecycle", "sdlc", "awareness",
    "phishing", "vishing", "smishing", "ransomware", "spyware", "adware", "rootkits",
    "botnets", "trojans", "viruses", "worms", "zero", "day", "exploits", "patches", "patching",
    "updates", "upgrades", "configuration", "ticketing", "crm", "erp", "scm", "hcm", "financial",
    "accounting", "bi", "warehousing", "etl", "extract", "transform", "load", "lineage",
    "master", "mdm", "lakes", "marts", "big", "hadoop", "spark", "kafka", "flink", "mongodb",
    "cassandra", "redis", "elasticsearch", "relational", "mysql", "postgresql", "db2",
    "teradata", "snowflake", "redshift", "synapse", "bigquery", "aurora", "dynamodb",
    "documentdb", "cosmosdb", "graph", "neo4j", "graphdb", "timeseries", "influxdb",
    "timescaledb", "columnar", "vertica", "clickhouse", "vector", "pinecone", "weaviate",
    "milvus", "qdrant", "chroma", "faiss", "annoy", "hnswlib", "scikit", "learn", "tensorflow",
    "pytorch", "keras", "xgboost", "lightgbm", "catboost", "statsmodels", "numpy", "pandas",
    "matplotlib", "seaborn", "plotly", "bokeh", "dash", "flask", "django", "fastapi", "spring",
    "boot", ".net", "core", "node.js", "express.js", "react", "angular", "vue.js", "svelte",
    "jquery", "bootstrap", "tailwind", "sass", "less", "webpack", "babel", "npm", "yarn",
    "ansible", "terraform", "jenkins", "gitlab", "github", "actions", "codebuild", "codepipeline",
    "codedeploy", "build", "deploy", "run", "lambda", "functions", "serverless", "microservices",
    "gateway", "mesh", "istio", "linkerd", "grpc", "restful", "soap", "message", "queues",
    "rabbitmq", "activemq", "bus", "sqs", "sns", "pubsub", "version", "control", "svn",
    "mercurial", "trello", "asana", "monday.com", "smartsheet", "project", "primavera",
    "zendesk", "freshdesk", "itil", "cobit", "prince2", "pmp", "master", "owner", "lean",
    "six", "sigma", "black", "belt", "green", "yellow", "qms", "9001", "27001", "14001",
    "ohsas", "18001", "sa", "8000", "cmii", "cmi", "cism", "cissp", "ceh", "comptia",
    "security+", "network+", "a+", "linux+", "ccna", "ccnp", "ccie", "certified", "solutions",
    "architect", "developer", "sysops", "administrator", "specialty", "professional", "azure",
    "az-900", "az-104", "az-204", "az-303", "az-304", "az-400", "az-500", "az-700", "az-800",
    "az-801", "dp-900", "dp-100", "dp-203", "ai-900", "ai-102", "da-100", "pl-900", "pl-100",
    "pl-200", "pl-300", "pl-400", "pl-500", "ms-900", "ms-100", "ms-101", "ms-203", "ms-500",
    "ms-700", "ms-720", "ms-740", "ms-600", "sc-900", "sc-200", "sc-300", "sc-400", "md-100",
    "md-101", "mb-200", "mb-210", "mb-220", "mb-230", "mb-240", "mb-260", "mb-300", "mb-310",
    "mb-320", "mb-330", "mb-340", "mb-400", "mb-500", "mb-600", "mb-700", "mb-800", "mb-910",
    "mb-920", "gcp-ace", "gcp-pca", "gcp-pde", "gcp-pse", "gcp-pml", "gcp-psa", "gcp-pcd",
    "gcp-pcn", "gcp-psd", "gcp-pda", "gcp-pci", "gcp-pws", "gcp-pwa", "gcp-pme", "gcp-pms",
    "gcp-pmd", "gcp-pma", "gcp-pmc", "gcp-pmg", "cisco", "juniper", "red", "hat", "rhcsa",
    "rhce", "vmware", "vcpa", "vcpd", "vcpi", "vcpe", "vcpx", "citrix", "cc-v", "cc-p",
    "cc-e", "cc-m", "cc-s", "cc-x", "palo", "alto", "pcnsa", "pcnse", "fortinet", "fcsa",
    "fcsp", "fcc", "fcnsp", "fct", "fcp", "fcs", "fce", "fcn", "fcnp", "fcnse"
])
STOP_WORDS = NLTK_STOP_WORDS.union(CUSTOM_STOP_WORDS)

# --- Helpers ---
def clean_text(text):
    """Cleans text by removing newlines, extra spaces, and non-ASCII characters."""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip().lower()

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return ''.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        return f"[ERROR] {str(e)}"

def extract_years_of_experience(text):
    """Extracts years of experience from a given text by parsing date ranges or keywords."""
    text = text.lower()
    total_months = 0
    # Corrected regex: changed '[a-2]*' to '[a-z]*'
    job_date_ranges = re.findall(
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\s*(?:to|–|-)\s*(present|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})',
        text
    )

    for start, end in job_date_ranges:
        try:
            start_date = datetime.strptime(start.strip(), '%b %Y')
        except ValueError:
            try:
                start_date = datetime.strptime(start.strip(), '%B %Y')
            except ValueError:
                continue

        if end.strip() == 'present':
            end_date = datetime.now()
        else:
            try:
                end_date = datetime.strptime(end.strip(), '%b %Y')
            except ValueError:
                try:
                    end_date = datetime.strptime(end.strip(), '%B %Y')
                except ValueError:
                    continue

        delta_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        total_months += max(delta_months, 0)

    if total_months == 0:
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\+)?\s*(year|yrs|years)\b', text)
        if not match:
            match = re.search(r'experience[^\d]{0,10}(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

    return round(total_months / 12, 1)

def extract_email(text):
    """Extracts an email address from the given text."""
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def extract_name(text):
    """
    Attempts to extract a name from the first few lines of the resume text.
    This is a heuristic and might not be perfect for all resume formats.
    """
    lines = text.strip().split('\n')
    if not lines:
        return None

    potential_name_lines = []
    for line in lines[:3]:
        line = line.strip()
        if not re.search(r'[@\d\.\-]', line) and len(line.split()) <= 4 and (line.isupper() or (line and line[0].isupper() and all(word[0].isupper() or not word.isalpha() for word in line.split()))):
            potential_name_lines.append(line)

    if potential_name_lines:
        name = max(potential_name_lines, key=len)
        name = re.sub(r'summary|education|experience|skills|projects|certifications', '', name, flags=re.IGNORECASE).strip()
        if name:
            return name.title()
    return None

# --- Generative AI Suggestion Function (MODIFIED TO BE CONCISE AND RULE-BASED) ---
@st.cache_data(show_spinner="Generating AI Suggestion...")
def generate_ai_suggestion(candidate_name, score, years_exp, semantic_similarity, jd_text, resume_text):
    """
    Generates a concise AI suggestion based on rules.
    """
    overall_fit = ""
    recommendation = ""
    strengths_summary = []
    gaps_summary = []

    # Define thresholds and generate suggestions based on score, years_exp, and semantic_similarity
    # ADJUSTED THRESHOLDS BELOW
    if score >= 85 and years_exp >= 4 and semantic_similarity >= 0.75: # Adjusted Strong Fit
        overall_fit = "Strong Fit"
        recommendation = "Highly Recommended for Interview"
        strengths_summary.append("Exceptional alignment with job requirements and extensive relevant experience.")
    elif score >= 65 and years_exp >= 2 and semantic_similarity >= 0.4: # Adjusted Moderate Fit
        overall_fit = "Moderate Fit"
        recommendation = "Recommended for Interview"
        strengths_summary.append("Good alignment with job description keywords and solid experience.")
        if years_exp < 3:
            gaps_summary.append(f"Experience ({years_exp:.1f} yrs) slightly below ideal; probe depth.")
        if semantic_similarity < 0.7:
            gaps_summary.append("Some conceptual gaps with JD; probe specific skill applications.")
    else:
        overall_fit = "Low Fit"
        recommendation = "Further Review / Likely Decline"
        if score < 65: # Changed from 75 to 65 to match new Moderate Fit threshold
            gaps_summary.append(f"Overall score ({score:.2f}%) indicates significant mismatch.")
        if years_exp < 2:
            gaps_summary.append(f"Insufficient experience ({years_exp:.1f} yrs).")
        if semantic_similarity < 0.4: # Changed from 0.6 to 0.4 to match new Moderate Fit threshold
            gaps_summary.append(f"Low semantic similarity ({semantic_similarity:.2f}) suggests conceptual mismatch.")

    # Construct the concise output
    summary_text = f"**Overall Fit:** {overall_fit}. "
    if strengths_summary:
        summary_text += f"**Strengths:** {'; '.join(strengths_summary)}. "
    if gaps_summary:
        summary_text += f"**Gaps:** {'; '.join(gaps_summary)}. "
    summary_text += f"**Recommendation:** {recommendation}."

    return summary_text


def semantic_score(resume_text, jd_text, years_exp):
    """
    Calculates a semantic score using an ML model and provides additional details.
    Falls back to smart_score if the ML model is not loaded or prediction fails.
    Applies STOP_WORDS filtering for keyword analysis (internally, not for display).
    """
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    score = 0.0
    feedback = "Initial assessment." # This will be overwritten by the generate_ai_suggestion function
    semantic_similarity = 0.0

    if ml_model is None or model is None:
        st.warning("ML models not loaded. Providing basic score and generic feedback.")
        # Simplified fallback for score and feedback
        resume_words = {word for word in re.findall(r'\b\w+\b', resume_clean) if word not in STOP_WORDS}
        jd_words = {word for word in re.findall(r'\b\w+\b', jd_clean) if word not in STOP_WORDS}
        
        overlap_count = len(resume_words & jd_words)
        total_jd_words = len(jd_words)
        
        basic_score = (overlap_count / total_jd_words) * 70 if total_jd_words > 0 else 0
        basic_score += min(years_exp * 5, 30) # Add up to 30 for experience
        score = round(min(basic_score, 100), 2)
        
        feedback = "Due to missing ML models, a detailed AI suggestion cannot be provided. Basic score derived from keyword overlap. Manual review is highly recommended."
        
        return score, feedback, 0.0 # Return 0 for semantic similarity if ML not available


    try:
        jd_embed = model.encode(jd_clean)
        resume_embed = model.encode(resume_clean)

        semantic_similarity = cosine_similarity(jd_embed.reshape(1, -1), resume_embed.reshape(1, -1))[0][0]
        semantic_similarity = float(np.clip(semantic_similarity, 0, 1))

        # Internal calculation for model, not for display
        resume_words_filtered = {word for word in re.findall(r'\b\w+\b', resume_clean) if word not in STOP_WORDS}
        jd_words_filtered = {word for word in re.findall(r'\b\w+\b', jd_clean) if word not in STOP_WORDS}
        keyword_overlap_count = len(resume_words_filtered & jd_words_filtered)
        
        years_exp_for_model = float(years_exp) if years_exp is not None else 0.0

        features = np.concatenate([jd_embed, resume_embed, [years_exp_for_model], [keyword_overlap_count]])

        predicted_score = ml_model.predict([features])[0]

        if len(jd_words_filtered) > 0:
            jd_coverage_percentage = (keyword_overlap_count / len(jd_words_filtered)) * 100
        else:
            jd_coverage_percentage = 0.0

        blended_score = (predicted_score * 0.6) + \
                        (jd_coverage_percentage * 0.1) + \
                        (semantic_similarity * 100 * 0.3)

        if semantic_similarity > 0.7 and years_exp >= 3:
            blended_score += 5

        score = float(np.clip(blended_score, 0, 100))
        
        # The AI suggestion text will be generated separately for display by generate_ai_suggestion.
        return round(score, 2), "AI suggestion will be generated...", round(semantic_similarity, 2) # Placeholder feedback


    except Exception as e:
        st.warning(f"Error during semantic scoring, falling back to basic: {e}")
        # Simplified fallback for score and feedback if ML prediction fails
        resume_words = {word for word in re.findall(r'\b\w+\b', resume_clean) if word not in STOP_WORDS}
        jd_words = {word for word in re.findall(r'\b\w+\b', jd_clean) if word not in STOP_WORDS}
        
        overlap_count = len(resume_words & jd_words)
        total_jd_words = len(jd_words)
        
        basic_score = (overlap_count / total_jd_words) * 70 if total_jd_words > 0 else 0
        basic_score += min(years_exp * 5, 30) # Add up to 30 for experience
        score = round(min(basic_score, 100), 2)

        feedback = "Due to an error in core AI model, a detailed AI suggestion cannot be provided. Basic score derived. Manual review is highly recommended."

        return score, feedback, 0.0 # Return 0 for semantic similarity on fallback


# --- Email Generation Function ---
def create_mailto_link(recipient_email, candidate_name, job_title="Job Opportunity", sender_name="Recruiting Team"):
    """
    Generates a mailto: link with pre-filled subject and body for inviting a candidate.
    """
    subject = urllib.parse.quote(f"Invitation for Interview - {job_title} - {candidate_name}")
    body = urllib.parse.quote(f"""Dear {candidate_name},

We were very impressed with your profile and would like to invite you for an interview for the {job_title} position.

Best regards,

The {sender_name}""")
    return f"mailto:{recipient_email}?subject={subject}&body={body}"

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="ScreenerPro - AI Resume Screener", page_icon="🧠")
st.title("🧠 ScreenerPro – AI-Powered Resume Screener")

# --- Job Description and Controls Section ---
st.markdown("## ⚙️ Define Job Requirements & Screening Criteria")
col1, col2 = st.columns([2, 1])

with col1:
    jd_text = ""
    job_roles = {"Upload my own": None}
    if os.path.exists("data"):
        for fname in os.listdir("data"):
            if fname.endswith(".txt"):
                job_roles[fname.replace(".txt", "").replace("_", " ").title()] = os.path.join("data", fname)

    jd_option = st.selectbox("📌 **Select a Pre-Loaded Job Role or Upload Your Own Job Description**", list(job_roles.keys()))
    if jd_option == "Upload my own":
        jd_file = st.file_uploader("Upload Job Description (TXT)", type="txt", help="Upload a .txt file containing the job description.")
        if jd_file:
            jd_text = jd_file.read().decode("utf-8")
    else:
        jd_path = job_roles[jd_option]
        if jd_path and os.path.exists(jd_path):
            with open(jd_path, "r", encoding="utf-8") as f:
                jd_text = f.read()
    
    if jd_text:
        with st.expander("📝 View Loaded Job Description"):
            st.text_area("Job Description Content", jd_text, height=200, disabled=True, label_visibility="collapsed")

with col2:
    cutoff = st.slider("📈 **Minimum Score Cutoff (%)**", 0, 100, 75, help="Candidates scoring below this percentage will be flagged for closer review or considered less suitable.")
    min_experience = st.slider("💼 **Minimum Experience Required (Years)**", 0, 15, 2, help="Candidates with less than this experience will be noted.")
    st.markdown("---")
    st.info("Once criteria are set, upload resumes below to begin screening.")

resume_files = st.file_uploader("📄 **Upload Resumes (PDF)**", type="pdf", accept_multiple_files=True, help="Upload one or more PDF resumes for screening.")

df = pd.DataFrame()

if jd_text and resume_files:
    # --- Job Description Keyword Cloud ---
    st.markdown("---")
    st.markdown("## ☁️ Job Description Keyword Cloud")
    st.caption("Visualizing the most frequent and important keywords from the Job Description.")
    jd_words_for_cloud = " ".join([word for word in re.findall(r'\b\w+\b', clean_text(jd_text)) if word not in STOP_WORDS])
    if jd_words_for_cloud:
        wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(jd_words_for_cloud)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free up memory
    else:
        st.info("No significant keywords to display for the Job Description. Please ensure your JD has sufficient content.")
    st.markdown("---")

    results = []
    resume_text_map = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(resume_files):
        status_text.text(f"Processing {file.name} ({i+1}/{len(resume_files)})...")
        progress_bar.progress((i + 1) / len(resume_files))

        text = extract_text_from_pdf(file)
        if text.startswith("[ERROR]"):
            st.error(f"Failed to process {file.name}: {text.replace('[ERROR] ', '')}")
            continue

        exp = extract_years_of_experience(text)
        email = extract_email(text)
        candidate_name = extract_name(text) or file.name.replace('.pdf', '').replace('_', ' ').title()

        # Calculate Matched Keywords and Missing Skills
        resume_clean_for_keywords = clean_text(text)
        jd_clean_for_keywords = clean_text(jd_text)

        # Filter out stop words for keyword analysis
        resume_words_set = {word for word in re.findall(r'\b\w+\b', resume_clean_for_keywords) if word not in STOP_WORDS}
        jd_words_set = {word for word in re.findall(r'\b\w+\b', jd_clean_for_keywords) if word not in STOP_WORDS}

        matched_keywords = list(resume_words_set.intersection(jd_words_set))
        missing_skills = list(jd_words_set.difference(jd_words_set)) # Corrected: should be jd_words_set.difference(resume_words_set)

        # semantic_score now returns score, placeholder feedback, semantic_similarity
        score, _, semantic_similarity = semantic_score(text, jd_text, exp)
        
        # Generate the detailed AI suggestion using the modified generate_ai_suggestion function
        detailed_ai_suggestion = generate_ai_suggestion(
            candidate_name=candidate_name,
            score=score,
            years_exp=exp,
            semantic_similarity=semantic_similarity,
            jd_text=jd_text,
            resume_text=text
        )

        results.append({
            "File Name": file.name,
            "Candidate Name": candidate_name,
            "Score (%)": score,
            "Years Experience": exp,
            "Email": email or "Not Found",
            "AI Suggestion": detailed_ai_suggestion, # Renamed from "Feedback" to "AI Suggestion"
            "Matched Keywords": ", ".join(matched_keywords), # Added Matched Keywords
            "Missing Skills": ", ".join(missing_skills),    # Added Missing Skills
            "Semantic Similarity": semantic_similarity,
            "Resume Raw Text": text
        })
        resume_text_map[file.name] = text
    
    progress_bar.empty()
    status_text.empty()


    df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False).reset_index(drop=True)

    st.session_state['screening_results'] = results
    
    # Save results to CSV for analytics.py to use
    df.to_csv("results.csv", index=False)


    # --- Overall Candidate Comparison Chart ---
    st.markdown("## 📊 Candidate Score Comparison")
    st.caption("Visual overview of how each candidate ranks against the job requirements.")
    if not df.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        # Define colors: Green for top, Yellow for moderate, Red for low
        colors = ['#4CAF50' if s >= cutoff else '#FFC107' if s >= (cutoff * 0.75) else '#F44336' for s in df['Score (%)']]
        bars = ax.bar(df['Candidate Name'], df['Score (%)'], color=colors)
        ax.set_xlabel("Candidate", fontsize=14)
        ax.set_ylabel("Score (%)", fontsize=14)
        ax.set_title("Resume Screening Scores Across Candidates", fontsize=16, fontweight='bold')
        ax.set_ylim(0, 100)
        plt.xticks(rotation=60, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free up memory
    else:
        st.info("Upload resumes to see a comparison chart.")

    st.markdown("---")

    # --- TOP CANDIDATE AI RECOMMENDATION (Game Changer Feature) ---
    st.markdown("## 👑 Top Candidate AI Recommendation")
    st.caption("A concise, AI-powered assessment for the most suitable candidate.")
    
    if not df.empty:
        top_candidate = df.iloc[0] # Get the top candidate (already sorted by score)
        st.markdown(f"### **{top_candidate['Candidate Name']}**")
        st.markdown(f"**Score:** {top_candidate['Score (%)']:.2f}% | **Experience:** {top_candidate['Years Experience']:.1f} years | **Semantic Similarity:** {top_candidate['Semantic Similarity']:.2f}")
        st.markdown(f"**AI Assessment:** {top_candidate['AI Suggestion']}") # Use the concise AI suggestion
        
        # Action button for the top candidate
        if top_candidate['Email'] != "Not Found":
            mailto_link_top = create_mailto_link(
                recipient_email=top_candidate['Email'],
                candidate_name=top_candidate['Candidate Name'],
                job_title=jd_option if jd_option != "Upload my own" else "Job Opportunity"
            )
            st.markdown(f'<a href="{mailto_link_top}" target="_blank"><button style="background-color:#00cec9;color:white;border:none;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;border-radius:8px;">📧 Invite Top Candidate for Interview</button></a>', unsafe_allow_html=True)
        else:
            st.info(f"Email address not found for {top_candidate['Candidate Name']}. Cannot send automated invitation.")
        
        st.markdown("---")
        st.info("For detailed analytics, matched keywords, and missing skills for ALL candidates, please navigate to the **Analytics Dashboard**.")

    else:
        st.info("No candidates processed yet to determine the top candidate.")


    # === AI Recommendation for Shortlisted Candidates (Streamlined) ===
    # This section now focuses on a quick summary for *all* shortlisted,
    # with the top one highlighted above.
    st.markdown("## 🌟 Shortlisted Candidates Overview")
    st.caption("Candidates meeting your score and experience criteria.")

    shortlisted_candidates = df[(df['Score (%)'] >= cutoff) & (df['Years Experience'] >= min_experience)]

    if not shortlisted_candidates.empty:
        st.success(f"**{len(shortlisted_candidates)}** candidate(s) meet your specified criteria (Score ≥ {cutoff}%, Experience ≥ {min_experience} years).")
        
        # Display a concise table for shortlisted candidates
        display_shortlisted_summary_cols = [
            'Candidate Name',
            'Score (%)',
            'Years Experience',
            'Semantic Similarity',
            'Email', # Include email here for quick reference
            'AI Suggestion' # Concise AI suggestion
        ]
        
        st.dataframe(
            shortlisted_candidates[display_shortlisted_summary_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score (%)": st.column_config.ProgressColumn(
                    "Score (%)",
                    help="Matching score against job requirements",
                    format="%f",
                    min_value=0,
                    max_value=100,
                ),
                "Years Experience": st.column_config.NumberColumn(
                    "Years Experience",
                    help="Total years of professional experience",
                    format="%.1f years",
                ),
                "Semantic Similarity": st.column_config.NumberColumn(
                    "Semantic Similarity",
                    help="Conceptual similarity between JD and Resume (higher is better)",
                    format="%.2f",
                    min_value=0,
                    max_value=1
                ),
                "AI Suggestion": st.column_config.Column(
                    "AI Suggestion",
                    help="AI's concise overall assessment and recommendation"
                )
            }
        )
        st.info("For individual detailed AI assessments and action steps, please refer to the table above or the Analytics Dashboard.")

    else:
        st.warning("No candidates met the defined screening criteria (score cutoff and minimum experience). You might consider adjusting the sliders or reviewing the uploaded resumes/JD.")

    st.markdown("---")

    # Add a 'Tag' column for quick categorization
    df['Tag'] = df.apply(lambda row: "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3 else (
        "✅ Good Fit" if row['Score (%)'] >= 75 else "⚠️ Needs Review"), axis=1)

    st.markdown("## 📋 Comprehensive Candidate Results Table")
    st.caption("Full details for all processed resumes. **For deep dive analytics and keyword breakdowns, refer to the Analytics Dashboard.**")
    
    # Define columns to display in the comprehensive table
    comprehensive_cols = [
        'Candidate Name',
        'Score (%)',
        'Years Experience',
        'Semantic Similarity',
        'Tag', # Keep the custom tag
        'Email',
        'AI Suggestion', # This will still contain the full AI suggestion text but is in a table, not per-candidate verbose display
        'Matched Keywords',
        'Missing Skills',
        # 'Resume Raw Text' # Removed from default display to keep table manageable, can be viewed in Analytics
    ]
    
    # Ensure all columns exist before trying to display them
    final_display_cols = [col for col in comprehensive_cols if col in df.columns]

    st.dataframe(
        df[final_display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score (%)": st.column_config.ProgressColumn(
                "Score (%)",
                help="Matching score against job requirements",
                format="%f",
                min_value=0,
                max_value=100,
            ),
            "Years Experience": st.column_config.NumberColumn(
                "Years Experience",
                help="Total years of professional experience",
                format="%.1f years",
            ),
            "Semantic Similarity": st.column_config.NumberColumn(
                "Semantic Similarity",
                help="Conceptual similarity between JD and Resume (higher is better)",
                format="%.2f",
                min_value=0,
                max_value=1
            ),
            "AI Suggestion": st.column_config.Column(
                "AI Suggestion",
                help="AI's overall assessment and recommendation"
            ),
            "Matched Keywords": st.column_config.Column(
                "Matched Keywords",
                help="Keywords found in both JD and Resume"
            ),
            "Missing Skills": st.column_config.Column(
                "Missing Skills",
                help="Key skills from JD not found in Resume"
            ),
        }
    )

    st.info("Remember to check the Analytics Dashboard for in-depth visualizations of skill overlaps, gaps, and other metrics!")
else:
    st.info("Please upload a Job Description and at least one Resume to begin the screening process.")
