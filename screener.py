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

def get_top_keywords(text, num_keywords=15):
    """Extracts and returns the top N most frequent keywords from text, excluding stop words."""
    cleaned_text = clean_text(text)
    words = [word for word in re.findall(r'\b\w+\b', cleaned_text) if word not in STOP_WORDS]
    word_counts = collections.Counter(words)
    return [word for word, count in word_counts.most_common(num_keywords)]

def semantic_score(resume_text, jd_text, years_exp):
    """
    Calculates a semantic score using an ML model and provides additional details.
    Falls back to smart_score if the ML model is not loaded or prediction fails.
    Applies STOP_WORDS filtering for keyword analysis (internally, not for display).
    """
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    score = 0.0
    feedback = "Initial assessment."
    semantic_similarity = 0.0
    jd_coverage_percentage = 0.0 # Still calculated, but not explicitly displayed

    if ml_model is None or model is None:
        # Fallback if ML models are not loaded
        # For simplicity, if ML model isn't there, we don't calculate advanced metrics
        # and provide a generic feedback.
        # In a real scenario, you might want to implement a simpler keyword-based fallback score here.
        st.warning("ML models not loaded. Providing basic score and feedback.")
        # Simplified fallback for score and feedback
        resume_words = {word for word in re.findall(r'\b\w+\b', resume_clean) if word not in STOP_WORDS}
        jd_words = {word for word in re.findall(r'\b\w+\b', jd_clean) if word not in STOP_WORDS}
        
        overlap_count = len(resume_words & jd_words)
        total_jd_words = len(jd_words)
        
        basic_score = (overlap_count / total_jd_words) * 70 if total_jd_words > 0 else 0
        basic_score += min(years_exp * 5, 30) # Add up to 30 for experience
        score = round(min(basic_score, 100), 2)
        
        if score > 70:
            feedback = "Good potential based on keyword match and experience. Manual review recommended."
        elif score > 40:
            feedback = "Moderate potential. Review for transferable skills."
        else:
            feedback = "Lower match based on basic keyword alignment. Manual review advised."
        
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

        if score > 90:
            feedback = "Excellent fit: Outstanding alignment with job requirements, high conceptual match, and strong relevant experience. **Highly Recommended for Interview.**"
        elif score >= 75:
            feedback = "Good fit: Solid alignment with the role, good conceptual match, and relevant experience demonstrated. **Recommended for Further Review/Interview.**"
        elif score >= 60:
            feedback = "Moderate fit: Decent potential, but some areas for improvement in deeper experience matching or nuanced skill alignment. **Consider for a deeper dive.**"
        else:
            feedback = "Initial assessment indicates a lower match based on semantic alignment and experience. This candidate may possess transferable skills or unique experiences not immediately highlighted, but a more **in-depth manual review is essential to determine suitability.**"

        if score < 10: # Fallback to a "not a good fit" for extremely low scores
             feedback = "Minimal alignment with the job requirements. This candidate is likely not a good fit for this role."

        return round(score, 2), feedback, round(semantic_similarity, 2)

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

        if score > 70:
            feedback = "Good potential based on keyword match and experience. Manual review recommended."
        elif score > 40:
            feedback = "Moderate potential. Review for transferable skills."
        else:
            feedback = "Lower match based on basic keyword alignment. Manual review advised."

        return score, feedback, 0.0 # Return 0 for semantic similarity on fallback


# --- Email Generation Function ---
def create_mailto_link(recipient_email, candidate_name, job_title="Job Opportunity", sender_name="Recruiting Team"):
    """
    Generates a mailto: link with pre-filled subject and body for inviting a candidate.
    """
    subject = urllib.parse.quote(f"Invitation for Interview - {job_title} - {candidate_name}")
    body = urllib.parse.quote(f"""Dear {candidate_name},

We were very impressed with your profile and would like to invite you for an interview for the {job_title} position.

Please let us know your availability in the coming days.

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

        score, feedback, semantic_similarity = semantic_score(text, jd_text, exp)

        results.append({
            "File Name": file.name,
            "Candidate Name": candidate_name,
            "Score (%)": score,
            "Years Experience": exp,
            "Email": email or "Not Found",
            "Feedback": feedback,
            "Semantic Similarity": semantic_similarity,
            "Resume Raw Text": text
        })
        resume_text_map[file.name] = text
    
    progress_bar.empty()
    status_text.empty()


    df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False).reset_index(drop=True)

    st.session_state['screening_results'] = results

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
    else:
        st.info("Upload resumes to see a comparison chart.")

    st.markdown("---")

    # === Detailed Individual Candidate Analysis ===
    st.markdown("## 🔍 Detailed Candidate Insights")
    st.caption("Dive deeper into each candidate's strengths and areas for improvement relative to the job description.")

    if not df.empty:
        for idx, row in df.iterrows():
            candidate_display_name = row['Candidate Name']
            
            with st.container(border=True):
                st.subheader(f"{idx+1}. {candidate_display_name} - Score: {row['Score (%)']:.2f}%")
                
                col_info, col_exp_match = st.columns([3, 1])

                with col_info:
                    st.markdown(f"**Overall Assessment:** {row['Feedback']}")
                    st.write(f"**Years of Experience:** {row['Years Experience']:.1f} years")
                    st.write(f"**Contact Email:** {row['Email']}")
                    st.write(f"**Semantic Similarity (JD vs. Resume):** **{row['Semantic Similarity']:.2f}** (Higher score indicates closer conceptual match.)")

                with col_exp_match:
                    st.markdown("### Experience Match")
                    exp_ratio = min(row['Years Experience'] / min_experience, 1.0) if min_experience > 0 else 1.0
                    st.progress(exp_ratio)
                    if row['Years Experience'] >= min_experience:
                        st.success(f"Candidate has {row['Years Experience']:.1f} years, meeting or exceeding required {min_experience} years.")
                    else:
                        st.warning(f"Candidate has {row['Years Experience']:.1f} years, less than required {min_experience} years.")

                with st.expander("📄 View Full Resume Text"):
                    st.code(resume_text_map.get(row['File Name'], ''), height=300)
            st.markdown("---")
    else:
        st.info("No candidates to display detailed analysis for yet.")

    st.markdown("---")

    # === AI Recommendation for Shortlisted Candidates ===
    st.markdown("## 🌟 AI Recommendation for Shortlisted Candidates")
    st.caption("An AI-driven assessment to guide your next steps for candidates meeting your criteria.")

    shortlisted_candidates = df[(df['Score (%)'] >= cutoff) & (df['Years Experience'] >= min_experience)]

    if not shortlisted_candidates.empty:
        st.success(f"**{len(shortlisted_candidates)}** candidate(s) meet your specified criteria (Score ≥ {cutoff}%, Experience ≥ {min_experience} years).")
        st.dataframe(shortlisted_candidates[['Candidate Name', 'Score (%)', 'Years Experience', 'Feedback', 'Semantic Similarity']], use_container_width=True)

        st.markdown("### Next Steps Recommendation:")
        for idx, candidate in shortlisted_candidates.iterrows():
            st.markdown(f"#### **{candidate['Candidate Name']}**")
            st.write(f"**Overall Fit:** {candidate['Feedback']}")
            
            # --- AI Suggestion ---
            ai_suggestion_text = f"Given their high score of **{candidate['Score (%)']:.2f}%**, strong **semantic similarity of {candidate['Semantic Similarity']:.2f}**, and solid experience, **we strongly recommend proceeding with an interview for {candidate['Candidate Name']}**. Their profile indicates a high likelihood of success in this role. Focus on exploring their practical application of relevant skills during the interview."
            st.write(f"**AI Suggestion:** {ai_suggestion_text}")

            if candidate['Email'] != "Not Found":
                st.write(f"📧 **Candidate Email:** {candidate['Email']}")
                # --- Email Button for Shortlisted Candidate ---
                mailto_link = create_mailto_link(
                    recipient_email=candidate['Email'],
                    candidate_name=candidate['Candidate Name'],
                    job_title=jd_option if jd_option != "Upload my own" else "Job Opportunity" # Use selected JD name or default
                )
                st.markdown(f'<a href="{mailto_link}" target="_blank"><button style="background-color:#4CAF50;color:white;border:none;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;border-radius:8px;">📧 Send Interview Invitation</button></a>', unsafe_allow_html=True)
            else:
                st.info(f"Email address not found for {candidate['Candidate Name']}. Cannot send automated invitation.")
            st.markdown("---")
    else:
        st.warning("No candidates met the defined screening criteria (score cutoff and minimum experience). You might consider adjusting the sliders or reviewing the uploaded resumes/JD.")

    st.markdown("---")

    # Add a 'Tag' column for quick categorization
    df['Tag'] = df.apply(lambda row: "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3 else (
        "✅ Good Fit" if row['Score (%)'] >= 75 else "⚠️ Needs Review"), axis=1)

    st.markdown("## 📋 Comprehensive Candidate Results Table")
    st.caption("Full details for all processed resumes.")
    display_df = df[['Candidate Name', 'Score (%)', 'Years Experience', 'Semantic Similarity', 'Feedback', 'Tag', 'Email']]
    st.dataframe(display_df, use_container_width=True)

    # Add download button for results
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Full Results (CSV)",
        data=csv_data,
        file_name="candidate_screening_results.csv",
        mime="text/csv",
        help="Download a CSV file containing all screening results, including detailed metrics."
    )
    st.markdown("---")
else:
    st.info("Please upload a Job Description and at least one Resume to begin the screening process.")
