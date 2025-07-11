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
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    # IMPORTANT: Ensure 'years_experience' and 'email' columns are explicitly added here
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            job_description_hash TEXT,
            job_description_summary TEXT,
            candidate_name TEXT,
            predicted_score REAL,
            keyword_match REAL,
            section_completeness REAL,
            semantic_similarity REAL,
            length_score REAL,
            shortlisted BOOLEAN,
            full_resume_text TEXT,
            ai_suggestion TEXT,
            years_experience REAL, -- Explicitly added this column
            email TEXT -- Explicitly added this column
        )
    ''')
    conn.commit()
    conn.close()
    return True

# Initialize database on app startup
db_initialized = init_db()
if not db_initialized:
    st.error("Failed to initialize database.")
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


# --- Main Scoring Function ---
def get_screening_features_and_score(job_description, resume_text):
    """
    Calculates features, predicts score using ML model, and prepares display scores.
    """
    if ml_screening_model is None or sentence_model is None:
        st.error("ML models not loaded. Cannot predict score.")
        return {
            "predicted_score": 0.0, "keyword_match_score": 0.0,
            "section_completeness_score": 0.0, "semantic_similarity": 0.0,
            "length_score": 0.0, "raw_resume_sections": {}, "years_experience": 0.0
        }

    # --- 1. Create ML Features (772 features for prediction) ---
    try:
        ml_features = create_ml_features(job_description, resume_text)
    except RuntimeError as e:
        st.error(f"Error creating ML features: {e}")
        return {
            "predicted_score": 0.0, "keyword_match_score": 0.0,
            "section_completeness_score": 0.0, "semantic_similarity": 0.0,
            "length_score": 0.0, "raw_resume_sections": {}, "years_experience": 0.0
        }

    # --- 2. Predict Score using ML Model ---
    try:
        predicted_score = ml_screening_model.predict(ml_features)[0]
        predicted_score = max(0.0, min(100.0, predicted_score)) # Clamp score to 0-100
    except Exception as e:
        st.error(f"Error predicting score with ML model: {e}")
        predicted_score = 0.0

    # --- 3. Calculate Display Features (for UI and AI Suggestion) ---
    # These are the 4 human-interpretable scores, recalculated from raw texts
    cleaned_jd = clean_text(job_description)
    cleaned_resume = clean_text(resume_text)

    # Keyword Match Score (for display)
    jd_keywords_set = set(get_top_keywords(cleaned_jd))
    resume_words_set = {word for word in re.findall(r'\b\w+\b', cleaned_resume) if word not in ALL_STOP_WORDS} # Ensure resume_words_set is defined
    keyword_overlap_count_display = len(jd_keywords_set.intersection(resume_words_set))
    keyword_match_score_display = (keyword_overlap_count_display / len(jd_keywords_set)) * 100 if len(jd_keywords_set) > 0 else 0.0

    # Section Completeness Score (for display)
    resume_sections = extract_sections(cleaned_resume)
    section_completeness_score_display = (sum(1 for sec in REQUIRED_SECTIONS if sec in resume_sections and resume_sections[sec]) / len(REQUIRED_SECTIONS) if REQUIRED_SECTIONS else 0.0) * 100

    # Semantic Similarity (for display - re-calculate using the clean text embeddings)
    jd_embedding_display = sentence_model.encode(cleaned_jd)
    resume_embedding_display = sentence_model.encode(cleaned_resume)
    semantic_similarity_display = cosine_similarity(jd_embedding_display.reshape(1, -1), resume_embedding_display.reshape(1, -1))[0][0] * 100

    # Length Score (for display)
    length_score_display = calculate_length_score(cleaned_resume)

    return {
        "predicted_score": predicted_score,
        "keyword_match_score": keyword_match_score_display,
        "section_completeness_score": section_completeness_score_display,
        "semantic_similarity": semantic_similarity_display,
        "length_score": length_score_display,
        "raw_resume_sections": resume_sections,
        "years_experience": extract_years_of_experience(resume_text) # Also return years_exp for AI suggestion
    }


# --- Database Operations ---
def get_jd_hash(jd_text):
    return hashlib.md5(jd_text.encode('utf-8')).hexdigest()

def insert_or_update_screening_result(jd_hash, jd_summary, candidate_name, scores, full_resume_text, ai_suggestion_text, years_experience, email, shortlisted=False):
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
                ai_suggestion = ?,
                years_experience = ?,
                email = ?
            WHERE id = ?
        ''', (scores['predicted_score'], scores['keyword_match_score'], 
              scores['section_completeness_score'], scores['semantic_similarity'], 
              scores['length_score'], shortlisted, full_resume_text, ai_suggestion_text, years_experience, email, existing_id[0]))
    else:
        c.execute('''
            INSERT INTO results (
                job_description_hash, job_description_summary, candidate_name, 
                predicted_score, keyword_match, section_completeness, 
                semantic_similarity, length_score, shortlisted, full_resume_text, ai_suggestion,
                years_experience, email
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (jd_hash, jd_summary, candidate_name, scores['predicted_score'], 
              scores['keyword_match_score'], scores['section_completeness_score'], 
              scores['semantic_similarity'], scores['length_score'], shortlisted, full_resume_text, ai_suggestion_text,
              years_experience, email))
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
    return df

def update_shortlist_status_in_db(jd_hash, candidate_name, is_shortlisted):
    conn = sqlite3.connect(DATABASE_FILE) # Corrected this line
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
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email. Error: {e}. Check sender email/password and allow less secure apps or use App Passwords for Gmail.")
        return False

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

df = pd.DataFrame() # Initialize an empty DataFrame for current session results

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

        text = extract_text_from_pdf(file)
        if not text.strip():
            st.warning(f"Could not extract text from {file.name}. Skipping analysis.")
            continue

        exp = extract_years_of_experience(text)
        email = extract_email(text)
        candidate_name = extract_name(text) or file.name.replace('.pdf', '').replace('_', ' ').title()

        # Get all scores and features
        screening_results = get_screening_features_and_score(jd_text, text)
        predicted_score = screening_results["predicted_score"]
        keyword_match_score_display = screening_results["keyword_match_score"]
        semantic_similarity_display = screening_results["semantic_similarity"]
        section_completeness_score_display = screening_results["section_completeness_score"]
        length_score_display = screening_results["length_score"]
        years_experience_display = screening_results["years_experience"]

        # Generate the detailed AI suggestion using the T5 model
        ai_suggestion_text = generate_ai_suggestion(
            candidate_name=candidate_name,
            score=predicted_score,
            years_exp=years_experience_display,
            semantic_similarity=semantic_similarity_display/100, # Convert back to 0-1 for AI suggestion logic
            keyword_match_score=keyword_match_score_display,
            jd_text=jd_text,
            resume_text=text
        )

        # Calculate Matched Keywords and Missing Skills for display
        resume_clean_for_keywords = clean_text(text)
        jd_clean_for_keywords = clean_text(jd_text)
        resume_words_set = {word for word in re.findall(r'\b\w+\b', resume_clean_for_keywords) if word not in ALL_STOP_WORDS}
        jd_words_set = {word for word in re.findall(r'\b\w+\b', jd_clean_for_keywords) if word not in ALL_STOP_WORDS}
        matched_keywords = list(resume_words_set.intersection(jd_words_set))
        missing_skills = list(jd_words_set.difference(resume_words_set)) 

        # Store in DB
        insert_or_update_screening_result(
            jd_hash=current_jd_hash,
            jd_summary=jd_summary_text,
            candidate_name=candidate_name,
            scores={
                'predicted_score': predicted_score,
                'keyword_match_score': keyword_match_score_display,
                'section_completeness_score': section_completeness_score_display,
                'semantic_similarity': semantic_similarity_display,
                'length_score': length_score_display
            },
            full_resume_text=text,
            ai_suggestion_text=ai_suggestion_text,
            years_experience=years_experience_display,
            email=email,
            shortlisted=False # Default to not shortlisted on initial upload
        )

        results_for_current_run.append({
            "File Name": file.name,
            "Candidate Name": candidate_name,
            "Score (%)": predicted_score,
            "Years Experience": years_experience_display,
            "Email": email or "Not Found",
            "AI Suggestion": ai_suggestion_text,
            "Matched Keywords": ", ".join(matched_keywords),
            "Missing Skills": ", ".join(missing_skills),
            "Semantic Similarity": semantic_similarity_display,
            "Keyword Match Score": keyword_match_score_display,
            "Resume Raw Text": text # Stored for detailed view/email
        })

    progress_bar.empty()
    status_text.empty()

    # Update the DataFrame for current session display
    df = pd.DataFrame(results_for_current_run).sort_values(by="Score (%)", ascending=False).reset_index(drop=True)

    st.success("All resumes processed and results saved to database.")

    # --- Overall Candidate Comparison Chart ---
    st.markdown("## 📊 Candidate Score Comparison")
    st.caption("Visual overview of how each candidate ranks against the job requirements.")
    if not df.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
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
        plt.close(fig)
    else:
        st.info("Upload resumes to see a comparison chart.")

    st.markdown("---")

    # --- TOP CANDIDATE AI RECOMMENDATION ---
    st.markdown("## 👑 Top Candidate AI Recommendation")
    st.caption("A concise, AI-powered assessment for the most suitable candidate.")

    if not df.empty:
        top_candidate = df.iloc[0]
        st.markdown(f"### **{top_candidate['Candidate Name']}**")
        st.markdown(f"**Score:** {top_candidate['Score (%)']:.2f}% | **Experience:** {top_candidate['Years Experience']:.1f} years | **Semantic Similarity:** {top_candidate['Semantic Similarity']:.2f}% | **Keyword Match:** {top_candidate['Keyword Match Score']:.2f}%")
        st.markdown(f"**AI Assessment:** {top_candidate['AI Suggestion']}")

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


    # --- Shortlisted Candidates Section (from DB, for current JD) ---
    st.markdown("## 🌟 Your Shortlisted Candidates (for this Job Description)")
    # Fetch all candidates for the current JD from DB, then filter by shortlisted status
    all_candidates_for_jd = get_screening_results_from_db(jd_hash=current_jd_hash)
    shortlisted_from_db = all_candidates_for_jd[all_candidates_for_jd['shortlisted'] == True]

    if not shortlisted_from_db.empty:
        st.success(f"**{len(shortlisted_from_db)}** candidate(s) are currently shortlisted for this job description.")

        # Display a concise table for shortlisted candidates
        display_shortlisted_summary_cols = [
            'candidate_name', 'predicted_score', 'years_experience', 'semantic_similarity',
            'keyword_match', 'ai_suggestion', 'email'
        ]

        st.dataframe(
            shortlisted_from_db[display_shortlisted_summary_cols].rename(columns={
                'candidate_name': 'Candidate Name',
                'predicted_score': 'Score (%)',
                'years_experience': 'Years Experience',
                'semantic_similarity': 'Semantic Similarity (%)',
                'keyword_match': 'Keyword Match (%)',
                'ai_suggestion': 'AI Suggestion',
                'email': 'Email'
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score (%)": st.column_config.ProgressColumn("Score (%)", format="%.2f", min_value=0, max_value=100),
                "Years Experience": st.column_config.NumberColumn("Years Experience", format="%.1f years"),
                "Semantic Similarity (%)": st.column_config.NumberColumn("Semantic Similarity (%)", format="%.2f"),
                "Keyword Match (%)": st.column_config.NumberColumn("Keyword Match (%)", format="%.2f"),
                "AI Suggestion": st.column_config.Column("AI Suggestion", help="AI's concise overall assessment and recommendation", width="large"),
                "Email": st.column_config.Column("Email", help="Candidate's email address")
            }
        )

        # Email functionality for shortlisted candidates
        st.markdown("---")
        st.subheader("✉️ Send Email to Shortlisted Candidates")
        recipient_emails_input = st.text_input("Enter recipient email(s) (comma-separated):", key="recipient_emails_shortlist", value=", ".join(shortlisted_from_db['email'].tolist()))
        email_subject = st.text_input("Email Subject:", value=f"Job Application Update: {jd_option if jd_option != 'Upload my own' else 'Your Application'}", key="email_subject_shortlist")
        email_body = st.text_area("Email Body:", height=200, key="email_body_shortlist",
            value="Dear candidate,\n\nThank you for your application. We are pleased to inform you that you have been shortlisted for further consideration.\n\nBest regards,\n[Your Company Name]"
        )

        if st.button("Send Emails to Shortlisted Candidates (from this JD)"):
            if not recipient_emails_input:
                st.error("Please enter at least one recipient email address.")
            else:
                recipient_list = [e.strip() for e in recipient_emails_input.split(',')]
                all_sent = True
                for email_addr in recipient_list:
                    if send_email(email_addr, email_subject, email_body):
                        st.success(f"Email sent successfully to {email_addr}!")
                    else:
                        all_sent = False
                        st.error(f"Failed to send email to {email_addr}.")
                if all_sent:
                    st.success("All emails sent!")

    else:
        st.info("No candidates have been shortlisted yet for this job description.")

    st.markdown("---")

    # --- Comprehensive Candidate Results Table (All processed candidates for current JD) ---
    st.markdown("## 📋 Comprehensive Candidate Results Table")
    st.caption("Full details for all processed resumes for this job description. Use the checkboxes to shortlist candidates.")

    if not df.empty:
        # Add a 'Tag' column for quick categorization based on current run's scores
        df['Tag'] = df.apply(lambda row: "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3 else (
            "✅ Good Fit" if row['Score (%)'] >= 75 else "⚠️ Needs Review"), axis=1)

        # Merge with DB shortlist status to show current state
        db_results_for_merge = get_screening_results_from_db(jd_hash=current_jd_hash)[['candidate_name', 'shortlisted']]
        df_display = df.merge(db_results_for_merge, on='candidate_name', how='left')
        df_display['shortlisted'] = df_display['shortlisted'].fillna(False)

        st.markdown("---")
        st.subheader("Mark Candidates for Shortlist")
        st.write("Check the box next to a candidate to mark them as 'shortlisted' in the database.")

        # Display using st.columns for interactive checkboxes
        # Define columns for display in the interactive section
        display_cols_interactive = ["Candidate Name", "Score (%)", "Years Experience", "Email", "AI Suggestion", "Shortlist"]

        # Create header row for interactive table
        header_cols = st.columns([1.5, 0.8, 0.8, 1.5, 3, 0.5]) # Adjusted widths for better fit
        with header_cols[0]: st.markdown("**Candidate Name**")
        with header_cols[1]: st.markdown("**Score (%)**")
        with header_cols[2]: st.markdown("**Exp (Yrs)**")
        with header_cols[3]: st.markdown("**Email**")
        with header_cols[4]: st.markdown("**AI Suggestion**")
        with header_cols[5]: st.markdown("**Shortlist**")

        st.markdown("---") # Separator for header

        for i, row_data in df_display.iterrows():
            col_name, col_score, col_exp, col_email, col_ai, col_shortlist = st.columns([1.5, 0.8, 0.8, 1.5, 3, 0.5])

            with col_name: st.write(row_data['Candidate Name'])
            with col_score: st.write(f"{row_data['Score (%)']:.2f}%")
            with col_exp: st.write(f"{row_data['Years Experience']:.1f}")
            with col_email: st.write(row_data['Email'])
            with col_ai: st.write(row_data['AI Suggestion'])

            with col_shortlist:
                current_shortlisted_status = bool(row_data['shortlisted'])
                new_shortlisted_status = st.checkbox(
                    "", # No label for cleaner look
                    value=current_shortlisted_status,
                    key=f"shortlist_checkbox_{row_data['Candidate Name']}_{i}" # Unique key
                )
                if new_shortlisted_status != current_shortlisted_status:
                    update_shortlist_status_in_db(current_jd_hash, row_data['Candidate Name'], new_shortlisted_status)
                    st.success(f"Updated shortlist status for {row_data['Candidate Name']} to {new_shortlisted_status}")
                    st.rerun() # Rerun to reflect changes in the UI and shortlisted section

    else:
        st.info("No candidates processed for this job description yet.")

else:
    st.info("Upload a Job Description and Resumes to begin screening.")

# --- About Section ---
st.markdown("---")
st.header("About This App")
st.info(
    "This AI Resume Screener Pro helps streamline your hiring process by analyzing "
    "job descriptions and candidate resumes. It uses TF-IDF for keyword matching, "
    "SentenceTransformers for semantic similarity, and a trained Machine Learning model "
    "for overall fit prediction. "
    "\n\n**New:** Integrated with a fine-tuned T5 model for concise job description and resume summaries, "
    "now supports direct PDF text extraction, and persists screening data using SQLite!"
)

st.markdown("---")
st.header("📊 Analytics Dashboard (View Historical Data)")
st.info("This section allows you to view all historical screening data stored in the database.")
if st.button("View All Historical Screening Data"):
    all_data = get_screening_results_from_db()
    if not all_data.empty:
        st.dataframe(all_data)
    else:
        st.info("No historical data found in the database.")
