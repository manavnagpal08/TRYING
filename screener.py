import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import hashlib # For hashing job descriptions
from datetime import datetime
import collections # Used by get_top_keywords
import matplotlib.pyplot as plt # For word cloud and charts
from wordcloud import WordCloud # For word cloud
import urllib.parse # For mailto links
import nltk # For stopwords
import os # For os.path.exists, os.listdir

# --- Database Configuration ---
DATABASE_FILE = "screening_data.db"
@st.cache_resource
def init_db():
    print(f"DEBUG: Attempting to initialize database at: {os.path.abspath(DATABASE_FILE)}")
    st.info(f"Attempting to initialize database at: {os.path.abspath(DATABASE_FILE)}")
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        # IMPORTANT: Column names here should match the CSV schema where applicable,
        # and other derived/app-specific columns.
        c.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                job_description_hash TEXT,
                job_description_summary TEXT,
                candidate_name TEXT,
                predicted_score REAL, -- This is the model's output, not directly from CSV 'numeric_score'
                keyword_match REAL,
                section_completeness REAL,
                semantic_similarity REAL,
                length_score REAL,
                shortlisted BOOLEAN,
                full_resume_text TEXT,
                detailed_ai_suggestion TEXT, -- Matches CSV 'detailed_ai_suggestion'
                years_exp REAL,             -- Matches CSV 'years_exp'
                email TEXT
            )
        ''')
        conn.commit()
        print("DEBUG: Database 'results' table checked/created successfully.")
        st.success("Database 'results' table checked/created successfully.")
        return True
    except sqlite3.Error as e:
        print(f"DEBUG: SQLite database initialization error: {e}")
        st.error(f"SQLite database initialization error: {e}")
        return False
    except Exception as e:
        print(f"DEBUG: General database initialization error: {e}")
        st.error(f"General database initialization error: {e}")
        return False
    finally:
        if conn:
            conn.close()

# Initialize database on app startup
db_initialized = init_db()

# --- Explicit check for database file existence ---
if not os.path.exists(DATABASE_FILE):
    print(f"DEBUG: Database file '{DATABASE_FILE}' was NOT found after initialization attempt.")
    st.error(f"Database file '{DATABASE_FILE}' was NOT found after initialization attempt. This indicates a permission issue or a critical failure during DB creation.")
    st.stop() # Stop the app if the database file isn't there
else:
    print(f"DEBUG: Database file '{DATABASE_FILE}' found. Proceeding.")
    st.info(f"Database file '{DATABASE_FILE}' found. Proceeding.")

if not db_initialized:
    st.error("Failed to initialize database. Please check logs for details.")
    st.stop() # Stop app if DB fails to init

# --- Configuration for Feature Extraction (MUST MATCH TRAIN_MODEL.PY) ---
# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

NLTK_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
CUSTOM_STOP_WORDS = set([
    "work", "experience", "years", "year", "months", "month", "day", "days", "project", "projects",
    "team", "teams", "developed", "managed", "led", "created", "implemented", "designed",
    "responsible", "proficient", "knowledge", "ability", "strong", "proven", "demonstrated",
    "solution", "solutions", "system", "systems", "platform", "platforms", "framework", "frameworks",
    "database", "databases", "server", "servers", "cloud", "computing", "machine", "learning",
    "artificial", "intelligence", "api", "apis", "rest", "graphql", "agile", "scrum", "kanban",
    "devops", "ci", "cd", "test", "testing", "qa", "quality", "assurance", "security", "data",
    "analytics", "big", "etl", "pipeline", "warehousing", "visualization", "reporting", "dashboard",
    "ui", "ux", "front-end", "backend", "full-stack", "mobile", "web", "desktop", "application",
    "applications", "software", "hardware", "firmware", "embedded", "network", "networking",
    "cybersecurity", "information", "systems", "it", "support", "consulting", "service", "services",
    "client", "customer", "business", "analysis", "analyst", "manager", "management", "director",
    "lead", "senior", "junior", "associate", "specialist", "engineer", "developer", "architect",
    "scientist", "researcher", "intern", "contractor", "freelancer", "freelance", "consultant",
    "professional", "expert", "specialist", "advisor", "adviser", "strategist", "strategical",
    "operational", "operations", "strategy", "strategic", "tactical", "tactics", "initiative",
    "initiatives", "program", "programs", "portfolio", "portfolios", "governance", "compliance",
    "regulation", "regulations", "audit", "auditing", "risk", "risks", "fraud", "forensics",
    "investigation", "investigations", "legal", "law", "attorney", "patent", "trademark", "copyright",
    "intellectual", "property", "finance", "financial", "accounting", "accountant", "bookkeeping",
    "audit", "tax", "taxation", "budget", "budgeting", "forecast", "forecasting", "treasury",
    "investment", "investments", "equity", "debt", "capital", "markets", "trading", "trader",
    "broker", "brokerage", "wealth", "management", "planner", "planning", "advisor", "adviser",
    "insurance", "underwriting", "claims", "actuarial", "actuary", "marketing", "sales", "brand",
    "branding", "product", "promotion", "advertising", "public", "relations", "pr", "communications",
    "content", "copywriting", "seo", "sem", "social", "media", "digital", "e-commerce", "retail",
    "wholesale", "distribution", "logistics", "supply", "chain", "procurement", "purchasing",
    "inventory", "warehouse", "transportation", "shipping", "freight", "customs", "export", "import",
    "international", "global", "local", "regional", "national", "country", "city", "state",
    "province", "territory", "district", "area", "zone", "region", "territory", "community",
    "communities", "public", "private", "government", "non-profit", "education", "healthcare",
    "medical", "pharmaceutical", "biotechnology", "life", "sciences", "clinical", "research",
    "development", "rd", "manufacturing", "production", "engineering", "quality", "control", "qc",
    "assurance", "qa", "safety", "environmental", "health", "safety", "ehs", "hseq", "sustainability",
    "sustainable", "green", "renewable", "energy", "oil", "gas", "mining", "metals", "minerals",
    "agriculture", "farming", "food", "beverage", "hospitality", "travel", "tourism", "leisure",
    "entertainment", "media", "publishing", "film", "television", "radio", "music", "art", "design",
    "fashion", "apparel", "textile", "jewelry", "watch", "accessories", "luxury", "goods", "sports",
    "fitness", "wellness", "personal", "care", "beauty", "cosmetics", "fragrances", "toiletries",
    "household", "cleaning", "consumer", "packaged", "goods", "cpg", "telecommunications", "broadband",
    "wireless", "wired", "fiber", "optic", "satellite", "broadcasting", "internet", "online",
    "ecommerce", "marketplace", "platform", "solutions", "services", "system", "technology",
    "innovation", "research", "development", "strategy", "strategic", "planning", "execution",
    "implementation", "optimization", "efficiency", "productivity", "performance", "growth",
    "expansion", "market", "share", "leadership", "competitive", "advantage", "differentiation",
    "value", "proposition", "customer", "satisfaction", "engagement", "loyalty", "retention",
    "acquisition", "sales", "revenue", "profit", "margin", "cost", "reduction", "efficiency",
    "productivity", "return", "investment", "roi", "compliance", "regulatory", "audit", "governance",
    "risk", "management", "security", "privacy", "ethics", "social", "responsibility", "sustainability"
])
ALL_STOP_WORDS = NLTK_STOP_WORDS.union(CUSTOM_STOP_WORDS)

# --- REQUIRED SECTIONS and PATTERN (MUST MATCH train_model.py) ---
REQUIRED_SECTIONS = [
    "experience", "education", "skills", "projects", "certifications",
    "awards", "publications", "extracurricular activities", "volunteer experience"
]

SECTION_HEADERS_PATTERN = re.compile(
    r'(?:^|\n)(?P<header>education|experience|skills|projects|certifications|awards|publications|extracurricular activities|volunteer experience|summary|about|profile|contact|interests|languages|references)\b',
    re.IGNORECASE
)

# --- Load Models (Cached for performance) ---
@st.cache_resource
def load_ml_models():
    ml_model = None
    sentence_transformer_model = None
    try:
        ml_model = joblib.load('ml_screening_model.pkl')
        st.success("ML Screening Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading ML Screening Model (ml_screening_model.pkl): {e}. Please ensure it's in your app's directory.")

    try:
        sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        st.success("SentenceTransformer model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading SentenceTransformer model: {e}")

    return ml_model, sentence_transformer_model

@st.cache_resource
def load_t5_model():
    t5_tokenizer = None
    t5_model = None
    T5_REPO_ID = "mnagpal/fine-tuned-t5-resume-screener"
    try:
        t5_tokenizer = AutoTokenizer.from_pretrained(T5_REPO_ID)
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_REPO_ID)
        st.success("T5 Model loaded successfully from Hugging Face Hub!")
    except Exception as e:
        st.error(f"Error loading T5 model from Hugging Face Hub: {e}")
    return t5_tokenizer, t5_model

# Load all models
ml_screening_model, sentence_model = load_ml_models()
t5_tokenizer, t5_model = load_t5_model()

# --- Helper Functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with PdfReader(uploaded_file) as pdf: # Using pypdf.PdfReader
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    return text

def extract_years_of_experience(text):
    total_months = 0
    text = text.lower()
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
                    if end_date > datetime.now(): # Handle future dates from parsing
                        end_date = datetime.now()
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
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def extract_name(text):
    lines = text.strip().split('\n')
    if not lines: return None
    potential_name_lines = []
    for line in lines[:3]:
        line = line.strip()
        if not re.search(r'[@\d\.\-]', line) and len(line.split()) <= 4 and (line.isupper() or (line and line[0].isupper() and all(word[0].isupper() or not word.isalpha() for word in line.split()))):
            potential_name_lines.append(line)
    if potential_name_lines:
        name = max(potential_name_lines, key=len)
        name = re.sub(r'summary|education|experience|skills|projects|certifications', '', name, flags=re.IGNORECASE).strip()
        if name: return name.title()
    return None

def get_top_keywords(text, top_n=50):
    words = clean_text(text).split()
    filtered_words = [word for word in words if word not in ALL_STOP_WORDS and len(word) > 2]
    word_counts = collections.Counter(filtered_words)

    bigrams = []
    for i in range(len(filtered_words) - 1):
        bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
        bigrams.append(bigram)
    bigram_counts = collections.Counter(bigrams)

    all_counts = word_counts + bigram_counts
    return [word for word, count in all_counts.most_common(top_n)]

def extract_sections(text):
    sections = {}
    current_section = None
    lines = text.split('\n')

    for line in lines:
        match = SECTION_HEADERS_PATTERN.match(line.strip())
        if match:
            header = match.group('header').lower()
            if header in REQUIRED_SECTIONS or header in ["summary", "about", "profile", "contact", "interests", "languages", "references"]:
                current_section = header
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line.strip())
        elif current_section is not None:
            sections[current_section].append(line.strip())

    for section, content_list in sections.items():
        sections[section] = "\n".join(content_list).strip()
    return sections

def calculate_length_score(resume_text):
    word_count = len(clean_text(resume_text).split())
    if 300 <= word_count <= 800:
        return 100
    elif 150 <= word_count < 300 or 800 < word_count <= 1200:
        return 70
    else:
        return 30

# --- Feature Calculation for ML Model (772 features) ---
def create_ml_features(jd_text, resume_text):
    if sentence_model is None:
        raise RuntimeError("SentenceTransformer model is not loaded. Cannot create ML features.")

    cleaned_jd = clean_text(jd_text)
    cleaned_resume = clean_text(resume_text)

    jd_embedding = sentence_model.encode(cleaned_jd)
    resume_embedding = sentence_model.encode(cleaned_resume)

    experience = extract_years_of_experience(resume_text)

    jd_keywords = set(get_top_keywords(cleaned_jd))
    resume_keywords = set(get_top_keywords(cleaned_resume))
    keyword_overlap_count = len(jd_keywords.intersection(resume_keywords))

    # NEW FEATURES: section completeness and length score
    resume_sections = extract_sections(cleaned_resume)
    section_completeness_score = (sum(1 for sec in REQUIRED_SECTIONS if sec in resume_sections and resume_sections[sec]) / len(REQUIRED_SECTIONS) if REQUIRED_SECTIONS else 0.0) * 100
    length_score = calculate_length_score(cleaned_resume)

    features = np.concatenate([
        jd_embedding.astype(float),
        resume_embedding.astype(float),
        np.array([float(experience)]),
        np.array([float(keyword_overlap_count)]),
        np.array([float(section_completeness_score)]), # New feature
        np.array([float(length_score)]) # New feature
    ])
    return features.reshape(1, -1)


# --- T5 Summarization Function ---
@st.cache_data(show_spinner="Generating T5 Summary...")
def generate_summary_with_t5(text, max_length=150):
    if not t5_model or not t5_tokenizer:
        return "T5 model not loaded."
    input_text = "summarize: " + text
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = t5_model.generate(
        input_ids,
        max_new_tokens=max_length,
        num_beams=4,
        early_stopping=True
    )
    summary = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

# --- AI Suggestion Function (Uses T5 for narrative, ML for score) ---
@st.cache_data(show_spinner="Generating AI Suggestion...")
def generate_ai_suggestion(candidate_name, score, years_exp, semantic_similarity, keyword_match_score, jd_text, resume_text):
    """
    Generates a concise and more realistic AI suggestion based on rules and scores,
    with less explicit numerical score mentions in the narrative.
    """
    overall_fit_phrase = ""
    recommendation_phrase = ""
    strengths = []
    gaps = []

    # Define thresholds
    HIGH_FIT_SCORE = 85
    MODERATE_FIT_SCORE = 65
    HIGH_SEMANTIC = 0.75
    MODERATE_SEMANTIC = 0.40
    HIGH_KEYWORD = 60
    MODERATE_KEYWORD = 30
    HIGH_EXP = 4
    MODERATE_EXP = 2

    # Overall Fit and Recommendation
    if score >= HIGH_FIT_SCORE and years_exp >= HIGH_EXP and semantic_similarity >= HIGH_SEMANTIC:
        overall_fit_phrase = "Exceptional Match"
        recommendation_phrase = "Strongly Recommended for Interview"
        strengths.append("Their profile demonstrates an **outstanding conceptual and practical alignment** with the job description.")
        strengths.append(f"They possess **extensive relevant experience** ({years_exp:.1f} years), aligning perfectly with the role's demands.")
        if keyword_match_score >= HIGH_KEYWORD:
            strengths.append("Their resume contains **all key industry terms and required skills**, indicating direct relevance.")
        else:
            strengths.append("While some specific keywords might not be explicit, their overall profile shows strong conceptual relevance.")

    elif score >= MODERATE_FIT_SCORE and years_exp >= MODERATE_EXP and semantic_similarity >= MODERATE_SEMANTIC:
        overall_fit_phrase = "Good Fit"
        recommendation_phrase = "Recommended for Interview"
        strengths.append("The candidate shows **good overall alignment** with the role, both conceptually and through their experience.")
        if years_exp >= HIGH_EXP:
            strengths.append(f"They bring **solid experience** ({years_exp:.1f} years) to the table.")
        else:
            gaps.append(f"Experience ({years_exp:.1f} years) is slightly below the ideal, suggesting a need to probe depth in specific areas.")

        if semantic_similarity >= HIGH_SEMANTIC:
            strengths.append("Their understanding of the domain and role responsibilities appears strong.")
        else:
            gaps.append("Their conceptual alignment with the role is fair; consider probing their approach to complex scenarios outlined in the JD.")

        if keyword_match_score >= MODERATE_KEYWORD:
            strengths.append("Many core skills and technologies mentioned in the JD are present in their resume.")
        else:
            gaps.append("Key skill mentions could be more prominent; verify specific technical proficiencies during interview.")

    else:
        overall_fit_phrase = "Lower Fit"
        recommendation_phrase = "Consider for Further Review / Likely Decline"
        gaps.append("Their overall profile indicates **significant discrepancies** with the job requirements, suggesting a lower overall fit.")

        if years_exp < MODERATE_EXP:
            gaps.append(f"Experience ({years_exp:.1f} years) is notably limited for this role.")

        if semantic_similarity < MODERATE_SEMANTIC:
            gaps.append("A **conceptual gap** exists between their profile and the job description, implying a potential mismatch in understanding or approach.")

        if keyword_match_score < MODERATE_KEYWORD:
            gaps.append("Many **critical keywords and required skills appear to be missing** from their resume.")

    summary_parts = [f"**Overall Fit:** {overall_fit_phrase}."]
    if strengths:
        summary_parts.append(f"**Strengths:** {' '.join(strengths)}")
    if gaps:
        summary_parts.append(f"**Areas for Development:** {' '.join(gaps)}")
    summary_parts.append(f"**Recommendation:** {recommendation_phrase}.")

    return " ".join(summary_parts)


# --- Database Operations ---
def get_jd_hash(jd_text):
    return hashlib.md5(jd_text.encode('utf-8')).hexdigest()

def insert_or_update_screening_result(jd_hash, jd_summary, candidate_name, scores, full_resume_text, detailed_ai_suggestion_text, years_exp, email, shortlisted=False):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()

    c.execute('''
        SELECT id FROM results
        WHERE job_description_hash = ? AND candidate_name = ?
    ''', (jd_hash, candidate_name))
    existing_id = c.fetchone()

    if existing_id:
        c.execute('''
            UPDATE results SET
                timestamp = CURRENT_TIMESTAMP,
                predicted_score = ?,
                keyword_match = ?,
                section_completeness = ?,
                semantic_similarity = ?,
                length_score = ?,
                shortlisted = ?,
                full_resume_text = ?,
                detailed_ai_suggestion = ?, -- Use detailed_ai_suggestion
                years_exp = ?,             -- Use years_exp
                email = ?
            WHERE id = ?
        ''', (scores['predicted_score'], scores['keyword_match_score'],
              scores['section_completeness_score'], scores['semantic_similarity'],
              scores['length_score'], shortlisted, full_resume_text, detailed_ai_suggestion_text, years_exp, email, existing_id[0]))
    else:
        c.execute('''
            INSERT INTO results (
                job_description_hash, job_description_summary, candidate_name,
                predicted_score, keyword_match, section_completeness,
                semantic_similarity, length_score, shortlisted, full_resume_text, detailed_ai_suggestion,
                years_exp, email
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (jd_hash, jd_summary, candidate_name, scores['predicted_score'],
              scores['keyword_match_score'], scores['section_completeness_score'],
              scores['semantic_similarity'], scores['length_score'], shortlisted, full_resume_text, detailed_ai_suggestion_text,
              years_exp, email))
    conn.commit()
    conn.close()

def get_screening_results_from_db(jd_hash=None):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    if jd_hash:
        c.execute('SELECT * FROM results WHERE job_description_hash = ? ORDER BY predicted_score DESC', (jd_hash,))
    else:
        c.execute('SELECT * FROM results ORDER BY predicted_score DESC')

    columns = [description[0] for description in c.description]
    results = c.fetchall()
    conn.close()

    df = pd.DataFrame(results, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Rename columns from DB names to consistent display names
    # Use errors='ignore' to prevent KeyError if a column is genuinely missing (e.g., in very old DBs)
    df = df.rename(columns={
        'years_exp': 'Years Experience',             # DB name to Display name
        'detailed_ai_suggestion': 'AI Suggestion',   # DB name to Display name
        'predicted_score': 'Score (%)',              # DB name to Display name
        'keyword_match': 'Keyword Match',            # DB name to Display name
        'section_completeness': 'Section Completeness', # DB name to Display name
        'semantic_similarity': 'Semantic Similarity', # DB name to Display name
        'length_score': 'Length Score',              # DB name to Display name
        'candidate_name': 'Candidate Name',          # DB name to Display name
        'email': 'Email'                             # DB name to Display name (capitalized)
    }, errors='ignore')
    return df

def update_shortlist_status_in_db(jd_hash, candidate_name, is_shortlisted):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute('''
        UPDATE results SET shortlisted = ?
        WHERE job_description_hash = ? AND candidate_name = ?
    ''', (is_shortlisted, jd_hash, candidate_name))
    conn.commit()
    conn.close()

def create_mailto_link(recipient_email, candidate_name, job_title):
    subject = urllib.parse.quote(f"Interview Invitation - {job_title}")
    body = urllib.parse.quote(
        f"Dear {candidate_name},\n\n"
        "Thank you for your interest in the position of [Job Title]. "
        "We were very impressed with your application and would like to invite you for an interview to discuss your qualifications further.\n\n"
        "Please let us know your availability for a brief chat in the coming days.\n\n"
        "Best regards,\n[Your Name/Hiring Team]"
    )
    return f"mailto:{recipient_email}?subject={subject}&body={body}"


# --- Email Sending Function ---
def send_email(recipient_email, subject, body):
    # Use Streamlit Secrets for email credentials
    try:
        sender_email = st.secrets["email"]["username"]
        sender_password = st.secrets["email"]["password"]
    except KeyError:
        st.error("Email credentials not found in Streamlit Secrets. Please configure them in .streamlit/secrets.toml.")
        return False

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
    except Exception as e:
        st.error(f"Failed to send email. Error: {e}. Check sender email/password and allow less secure apps or use App Passwords for Gmail.")
        return False
    finally:
        if 'server' in locals() and server:
            server.quit()
    return True

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="ScreenerPro - AI Resume Screener", page_icon="🧠")
st.title("🧠 ScreenerPro – AI-Powered Resume Screener")

# --- Job Description and Controls Section ---
st.markdown("## ⚙️ Define Job Requirements & Screening Criteria")
col1, col2 = st.columns([2, 1])

with col1:
    jd_text = ""
    # Populate job_roles from 'data' folder
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
            st.info("No file uploaded. You can paste the job description below.")
            jd_text = st.text_area("Or paste Job Description here:", height=200, key="manual_jd_paste")
    else:
        jd_path = job_roles[jd_option]
        if jd_path and os.path.exists(jd_path):
            with open(jd_path, "r", encoding="utf-8") as f:
                jd_text = f.read()
            # Display content of pre-loaded JD
            with st.expander(f"📝 View Content for '{jd_option}'"):
                st.text_area("Job Description Content", jd_text, height=200, disabled=True, label_visibility="collapsed")
        else:
            st.warning(f"File not found for pre-loaded JD: {jd_option}. Please ensure it exists in the 'data' folder.")

    # Existing expander for uploaded JD (if user chose to upload and text is there)
    if jd_option == "Upload my own" and jd_file and jd_text:
        with st.expander("📝 View Uploaded Job Description"):
            st.text_area("Job Description Content", jd_text, height=200, disabled=True, label_visibility="collapsed")

    jd_summary_text = ""
    if jd_text and t5_model:
        with st.expander("T5 Job Description Summary"):
            jd_summary_text = generate_summary_with_t5(jd_text)
            st.write(jd_summary_text)

    current_jd_hash = get_jd_hash(jd_text) if jd_text else None

with col2:
    cutoff = st.slider("📈 **Minimum Score Cutoff (%)**", 0, 100, 75, help="Candidates scoring below this percentage will be flagged for closer review or considered less suitable.")
    min_experience = st.slider("💼 **Minimum Experience Required (Years)**", 0, 15, 2, help="Candidates with less than this experience will be noted.")
    st.markdown("---")
    st.info("Once criteria are set, upload resumes below to begin screening.")

resume_files = st.file_uploader("📄 **Upload Resumes (PDF)**", type="pdf", accept_multiple_files=True, help="Upload one or more PDF resumes for screening.")

# Initialize an empty DataFrame for current session results
# Use session_state to persist data across reruns within a session
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

if jd_text and resume_files:
    # --- Job Description Keyword Cloud ---
    st.markdown("---")
    st.markdown("## ☁️ Job Description Keyword Cloud")
    st.caption("Visualizing the most frequent and important keywords from the Job Description.")
    jd_words_for_cloud = " ".join([word for word in re.findall(r'\b\w+\b', clean_text(jd_text)) if word not in ALL_STOP_WORDS])
    if jd_words_for_cloud:
        wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(jd_words_for_cloud)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No significant keywords to display for the Job Description. Please ensure your JD has sufficient content.")
    st.markdown("---")

    results_for_current_run = [] # To collect data for the session_state df

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(resume_files):
        status_text.text(f"Processing {file.name} ({i+1}/{len(resume_files)})...")
        progress_bar.progress((i + 1) / len(resume_files))

        full_resume_text = extract_text_from_pdf(file)
        if not full_resume_text.strip():
            st.warning(f"Could not extract text from {file.name}. Skipping analysis.")
            continue

        candidate_name = extract_name(full_resume_text) or file.name.replace(".pdf", "")
        candidate_email = extract_email(full_resume_text) or "N/A"

        scores = get_screening_features_and_score(jd_text, full_resume_text)

        # Generate AI Suggestion based on all calculated scores
        ai_suggestion_text = generate_ai_suggestion(
            candidate_name=candidate_name,
            score=scores['predicted_score'],
            years_exp=scores['years_experience'],
            semantic_similarity=scores['semantic_similarity'] / 100, # Convert back to 0-1 for AI suggestion function's thresholds
            keyword_match_score=scores['keyword_match_score'],
            jd_text=jd_text,
            resume_text=full_resume_text
        )

        # Save result to database
        insert_or_update_screening_result(
            jd_hash=current_jd_hash,
            jd_summary=jd_summary_text,
            candidate_name=candidate_name,
            scores=scores,
            full_resume_text=full_resume_text,
            detailed_ai_suggestion_text=ai_suggestion_text, # Use detailed_ai_suggestion_text here
            years_exp=scores['years_experience'], # Use years_exp here
            email=candidate_email,
            shortlisted=False # Default to not shortlisted on initial upload
        )

        results_for_current_run.append({
            "Candidate Name": candidate_name,
            "Score (%)": scores['predicted_score'], # Keep as float for sorting
            "Keyword Match (%)": scores['keyword_match_score'],
            "Section Completeness (%)": scores['section_completeness_score'],
            "Semantic Similarity (%)": scores['semantic_similarity'],
            "Length Score (%)": scores['length_score'],
            "Years Experience": scores['years_experience'], # Keep as float for sorting (display name)
            "Email": candidate_email,
            "AI Suggestion": ai_suggestion_text, # This is the display name
            "Full Resume Text": full_resume_text,
            "Shortlisted": False # Initial status
        })

    progress_bar.empty()
    status_text.empty()

    # --- Process and Display Results ---
    if results_for_current_run: # Check if the list is NOT empty
        df = pd.DataFrame(results_for_current_run)
        
        # Ensure numeric types for proper sorting and comparison
        df['Score (%)'] = pd.to_numeric(df['Score (%)'])
        df['Years Experience'] = pd.to_numeric(df['Years Experience'])

        # Sort by Score for accurate ranking
        df = df.sort_values(by="Score (%)", ascending=False).reset_index(drop=True)
        
        st.session_state.results_df = df # Store the results in session_state
        st.success("All resumes processed and results saved to database.")

        st.markdown("---")
        st.markdown("## 📊 Overall Screening Results")
        st.caption("A glance at the distribution of candidate scores.")

        # Create a histogram of scores
        fig, ax = plt.subplots(figsize=(10, 4))
        if not st.session_state.results_df.empty:
            st.session_state.results_df['Score (%)'].astype(float).hist(bins=10, ax=ax, edgecolor='black')
        else:
            ax.text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('Distribution of Candidate Scores')
        ax.set_xlabel('Score (%)')
        ax.set_ylabel('Number of Candidates')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("⚠️ No resumes could be processed successfully for the current run. This might be because the PDFs are scanned images without a text layer, are corrupted, or have complex structures that prevent text extraction.")
        st.info("Please ensure your PDF resumes are text-searchable (you should be able to select and copy text from them).")
        st.session_state.results_df = pd.DataFrame() # Ensure df in session_state is empty if no results
        # st.stop() # Removed st.stop() to allow "Past Screening Results" to load if available

st.markdown("---")

# --- Display Results from current session if available ---
if not st.session_state.results_df.empty:
    st.markdown("## 🥇 Top Candidate AI Recommendation")
    top_candidate = st.session_state.results_df.iloc[0]
    st.write(f"The **top candidate** for this Job Description is **{top_candidate['Candidate Name']}** with an overall score of **{top_candidate['Score (%)']:.1f}%**.")
    st.markdown(f"**AI Recommendation for {top_candidate['Candidate Name']}:**")
    st.info(top_candidate['AI Suggestion'])

    st.markdown("---")
    st.markdown("## ✅ Shortlisted Candidates")
    st.caption("Candidates you have marked as 'Shortlisted'.")
    # Fetch from DB for the current JD to ensure latest shortlist status
    # This will use the get_screening_results_from_db function which renames columns
    all_candidates_for_current_jd = get_screening_results_from_db(jd_hash=current_jd_hash)
    shortlisted_candidates_db = all_candidates_for_current_jd[all_candidates_for_current_jd['shortlisted'] == True]


    if not shortlisted_candidates_db.empty:
        for idx, row in shortlisted_candidates_db.iterrows():
            st.write(f"- **{row['Candidate Name']}** ({row['Score (%)']:.1f}%)")
            with st.expander(f"AI Suggestion for {row['Candidate Name']}"):
                st.info(row['AI Suggestion'])
    else:
        st.info("No candidates have been shortlisted yet for the current job description.")

    st.markdown("---")
    st.markdown("## 📋 Comprehensive Screening Table")
    st.caption("Review all candidates, their scores, and manage their status.")

    # Fetch the latest data from DB for the current JD to ensure the table reflects true DB state
    current_jd_results_from_db = get_screening_results_from_db(jd_hash=current_jd_hash)

    # Ensure numeric types for proper sorting and display
    for col in ['Score (%)', 'Keyword Match', 'Section Completeness', 'Semantic Similarity', 'Length Score', 'Years Experience']:
        if col in current_jd_results_from_db.columns: # Check if column exists before converting
            current_jd_results_from_db[col] = pd.to_numeric(current_jd_results_from_db[col], errors='coerce').fillna(0).round(1)

    # Add a 'Tag' column for quick categorization based on current run's scores
    current_jd_results_from_db['Tag'] = current_jd_results_from_db.apply(lambda row: "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3 else (
        "✅ Good Fit" if row['Score (%)'] >= 75 else "⚠️ Needs Review"), axis=1)

    # Display using st.data_editor for interactive checkboxes
    edited_df = st.data_editor(
        current_jd_results_from_db,
        column_config={
            "full_resume_text": st.column_config.Column(
                "Full Resume",
                help="Full text extracted from the resume.",
                width="small",
                display_as="hidden"
            ),
            "AI Suggestion": st.column_config.Column( # This is now AI Suggestion from DB
                "AI Suggestion",
                help="AI-generated summary and recommendation for the candidate.",
                width="large",
                display_as="expander"
            ),
            "shortlisted": st.column_config.CheckboxColumn( # Use the actual DB column name
                "Shortlisted?",
                help="Mark candidate as shortlisted",
                default=False,
            ),
            "Email": st.column_config.Column( # This is now Email from DB
                "Email",
                help="Candidate's email address.",
                width="small",
            ),
            "Score (%)": st.column_config.NumberColumn(
                "Score (%)",
                help="Predicted relevance score.",
                format="%.1f",
                width="small",
            ),
            "Years Experience": st.column_config.NumberColumn(
                "Years Experience",
                help="Extracted years of experience.",
                format="%.1f",
                width="small",
            ),
            "Keyword Match": st.column_config.NumberColumn(
                "Keywords (%)",
                help="Percentage of JD keywords found in resume.",
                format="%.1f",
                width="small",
            ),
            "Section Completeness": st.column_config.NumberColumn(
                "Sections (%)",
                help="Completeness of required resume sections.",
                format="%.1f",
                width="small",
            ),
            "Semantic Similarity": st.column_config.NumberColumn(
                "Semantic (%)",
                help="Conceptual similarity between resume and JD.",
                format="%.1f",
                width="small",
            ),
            "Length Score": st.column_config.NumberColumn(
                "Length (%)",
                help="Score based on resume length.",
                format="%.1f",
                width="small",
            ),
            # Hide raw DB columns that are renamed for display
            "job_description_hash": None,
            "job_description_summary": None,
            "timestamp": None
        },
        order=("Candidate Name", "Score (%)", "Years Experience", "Email", "AI Suggestion", "shortlisted", "Keyword Match", "Section Completeness", "Semantic Similarity", "Length Score"),
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        key="screening_results_table" # Important for unique widget key
    )

    # --- Handle updates from data_editor ---
    if st.session_state.screening_results_table.get('edited_rows') is not None:
        updated_rows_info = st.session_state.screening_results_table['edited_rows']
        if updated_rows_info:
            for index, updates in updated_rows_info.items():
                candidate_name = edited_df.loc[index, 'Candidate Name'] # Use the renamed column name
                if 'shortlisted' in updates: # Use the actual DB column name
                    is_shortlisted = updates['shortlisted']
                    update_shortlist_status_in_db(current_jd_hash, candidate_name, is_shortlisted)
                    st.toast(f"Updated {candidate_name} shortlist status to {is_shortlisted}")
                    st.rerun() # Rerun to reflect changes in the UI and shortlisted section

else:
    st.info("Upload a Job Description and Resumes to begin screening.")

# --- Past Screening Results ---
st.markdown("---")
st.markdown("## 🕰️ Past Screening Results")
st.caption("Review results from previous screening sessions for different Job Descriptions.")

# Get all unique JDs from the database
conn = sqlite3.connect(DATABASE_FILE)
c = conn.cursor()
c.execute('SELECT DISTINCT job_description_hash, job_description_summary FROM results')
past_jds = c.fetchall()
conn.close()

jd_options_for_history = {"All Past Results": None}
for jd_hash, jd_summary in past_jds:
    # Use summary if available, otherwise just hash
    display_name = jd_summary if jd_summary and len(jd_summary) > 10 else f"JD: {jd_hash[:8]}..."
    jd_options_for_history[display_name] = jd_hash

selected_past_jd = st.selectbox("Select a Past Job Description to View:", list(jd_options_for_history.keys()))

past_results_df = pd.DataFrame()
if selected_past_jd == "All Past Results":
    past_results_df = get_screening_results_from_db()
else:
    selected_hash = jd_options_for_history[selected_past_jd]
    past_results_df = get_screening_results_from_db(selected_hash)

# --- IMPORTANT: Add this check here ---
if not past_results_df.empty:
    # Column names are already renamed by get_screening_results_from_db
    # Ensure numeric types for proper sorting and display
    for col in ['Score (%)', 'Keyword Match', 'Section Completeness', 'Semantic Similarity', 'Length Score', 'Years Experience']:
        if col in past_results_df.columns:
            past_results_df[col] = pd.to_numeric(past_results_df[col], errors='coerce').fillna(0).round(1)

    # Sort by Score (%) descending
    past_results_df = past_results_df.sort_values(by='Score (%)', ascending=False).reset_index(drop=True)

    # Prepare for display, hiding large text fields by default
    st.dataframe(
        past_results_df[[
            'timestamp', 'Candidate Name', 'Score (%)', 'Keyword Match',
            'Section Completeness', 'Semantic Similarity', 'Length Score',
            'Years Experience', 'Email', 'shortlisted', 'AI Suggestion'
        ]],
        column_config={
            "timestamp": "Date",
            "Candidate Name": "Candidate",
            "AI Suggestion": st.column_config.Column(
                "AI Suggestion",
                help="AI-generated summary and recommendation.",
                display_as="expander"
            ),
            "shortlisted": st.column_config.CheckboxColumn( # Use the actual DB column name
                "Shortlisted?",
                disabled=True # Disable editing for past results view
            ),
            "Email": "Email", # Display email column
            "Score (%)": st.column_config.NumberColumn(format="%.1f"),
            "Years Experience": st.column_config.NumberColumn(format="%.1f"),
            "Keyword Match": st.column_config.NumberColumn(format="%.1f"),
            "Section Completeness": st.column_config.NumberColumn(format="%.1f"),
            "Semantic Similarity": st.column_config.NumberColumn(format="%.1f"),
            "Length Score": st.column_config.NumberColumn(format="%.1f")
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("No past screening results found for the selected Job Description or in the database.")


# --- About Section ---
st.sidebar.title("About ScreenerPro")
st.sidebar.info(
    "ScreenerPro is an AI-powered application designed to streamline the resume screening "
    "process. It leverages a custom-trained Machine Learning model, a Sentence Transformer for "
    "semantic understanding, and a fine-tuned T5 model for insightful AI suggestions.\n\n"
    "Upload job descriptions and resumes, and let AI assist you in identifying the best-fit candidates!"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Developed by [Manav Nagpal](https://www.linkedin.com/in/manav-nagpal-b03a74211/)"
)
