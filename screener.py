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
# import sqlite3 # REMOVED
# import hashlib # REMOVED (no more JD hashing for DB)
from datetime import datetime
import collections
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import urllib.parse
import nltk
import os

# --- Configuration for Feature Extraction ---
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
        with PdfReader(uploaded_file) as pdf:
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
                    if end_date > datetime.now():
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

# --- Feature Creation for ML Model (772 features) ---
def create_ml_features(job_description, resume_text):
    # Ensure models are loaded
    if sentence_model is None:
        raise RuntimeError("SentenceTransformer model not loaded. Cannot create semantic features.")

    # Semantic features
    jd_embedding = sentence_model.encode(clean_text(job_description))
    resume_embedding = sentence_model.encode(clean_text(resume_text))
    semantic_similarity_vector = cosine_similarity(jd_embedding.reshape(1, -1), resume_embedding.reshape(1, -1))[0]

    # Keyword features (simple presence/absence or count in a vector)
    # For a fixed 768-dim feature vector, we might use a predefined list of 4 keywords
    # and check their presence, concatenating with semantic_similarity_vector.
    # For this example, let's keep it aligned with previous discussions and assume a fixed size.
    # A robust solution involves a TF-IDF vectorizer fitted on a large corpus during training.
    # For now, we'll use a placeholder for keyword features if they were to contribute to the 772.
    # Given the original model was 768 (sentence embedding) + 4 (other features), let's ensure we match.

    # Placeholder for a few "hand-crafted" features if needed for 772 total
    # Example:
    # 1. Years of Experience
    # 2. Length Score
    # 3. Section Completeness (as a percentage)
    # 4. Keyword Match (as a percentage)

    years_exp = extract_years_of_experience(resume_text)
    length_score = calculate_length_score(resume_text)

    # Re-calculate keyword match for feature vector (if ml model expects it as a number)
    cleaned_jd = clean_text(job_description)
    cleaned_resume = clean_text(resume_text)
    jd_keywords_set = set(get_top_keywords(cleaned_jd))
    resume_words_set = {word for word in re.findall(r'\b\w+\b', cleaned_resume) if word not in ALL_STOP_WORDS}
    keyword_overlap_count = len(jd_keywords_set.intersection(resume_words_set))
    keyword_match_percentage = (keyword_overlap_count / len(jd_keywords_set)) * 100 if len(jd_keywords_set) > 0 else 0.0

    # Re-calculate section completeness for feature vector (if ml model expects it as a number)
    resume_sections = extract_sections(cleaned_resume)
    section_completeness_percentage = (sum(1 for sec in REQUIRED_SECTIONS if sec in resume_sections and resume_sections[sec]) / len(REQUIRED_SECTIONS) if REQUIRED_SECTIONS else 0.0) * 100

    # Combine all features into a single array for the ML model
    # Ensure the order and number of features matches what the ML model was trained on.
    # Assuming 768 (semantic) + 4 (other derived features) = 772 features.
    other_features = np.array([years_exp, length_score, section_completeness_percentage, keyword_match_percentage])

    # Concatenate the semantic similarity vector with the other features
    # Ensure both are float types and correctly shaped
    ml_features = np.concatenate((semantic_similarity_vector.astype(float), other_features.astype(float))).reshape(1, -1)

    return ml_features

# --- Feature Calculation for ML Model (772 features) ---
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
    resume_words_set = {word for word in re.findall(r'\b\w+\b', cleaned_resume) if word not in ALL_STOP_WORDS}
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
        "years_experience": extract_years_of_experience(resume_text)
    }

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


# --- Email Utilities (No DB operations here) ---
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

def send_email(recipient_email, subject, body):
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
            with st.expander(f"📝 View Content for '{jd_option}'"):
                st.text_area("Job Description Content", jd_text, height=200, disabled=True, label_visibility="collapsed")
        else:
            st.warning(f"File not found for pre-loaded JD: {jd_option}. Please ensure it exists in the 'data' folder.")

    if jd_option == "Upload my own" and jd_file and jd_text:
        with st.expander("📝 View Uploaded Job Description"):
            st.text_area("Job Description Content", jd_text, height=200, disabled=True, label_visibility="collapsed")

    jd_summary_text = ""
    if jd_text and t5_model:
        with st.expander("T5 Job Description Summary"):
            jd_summary_text = generate_summary_with_t5(jd_text)
            st.write(jd_summary_text)

    # current_jd_hash = get_jd_hash(jd_text) if jd_text else None # REMOVED

with col2:
    cutoff = st.slider("📈 **Minimum Score Cutoff (%)**", 0, 100, 75, help="Candidates scoring below this percentage will be flagged for closer review or considered less suitable.")
    min_experience = st.slider("💼 **Minimum Experience Required (Years)**", 0, 15, 2, help="Candidates with less than this experience will be noted.")
    st.markdown("---")
    st.info("Once criteria are set, upload resumes below to begin screening.")

resume_files = st.file_uploader("📄 **Upload Resumes (PDF)**", type="pdf", accept_multiple_files=True, help="Upload one or more PDF resumes for screening.")

# Initialize an empty DataFrame for current session results
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

    results_for_current_run = []

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

        ai_suggestion_text = generate_ai_suggestion(
            candidate_name=candidate_name,
            score=scores['predicted_score'],
            years_exp=scores['years_experience'],
            semantic_similarity=scores['semantic_similarity'] / 100,
            keyword_match_score=scores['keyword_match_score'],
            jd_text=jd_text,
            resume_text=full_resume_text
        )

        # Removed database insertion here

        results_for_current_run.append({
            "Candidate Name": candidate_name,
            "Score (%)": scores['predicted_score'],
            "Keyword Match (%)": scores['keyword_match_score'],
            "Section Completeness (%)": scores['section_completeness_score'],
            "Semantic Similarity (%)": scores['semantic_similarity'],
            "Length Score (%)": scores['length_score'],
            "Years Experience": scores['years_experience'],
            "Email": candidate_email,
            "AI Suggestion": ai_suggestion_text,
            "Full Resume Text": full_resume_text,
            "Shortlisted": False # Initial status
        })

    progress_bar.empty()
    status_text.empty()

    # --- Process and Display Results ---
    if results_for_current_run:
        df = pd.DataFrame(results_for_current_run)

        df['Score (%)'] = pd.to_numeric(df['Score (%)'])
        df['Years Experience'] = pd.to_numeric(df['Years Experience'])

        df = df.sort_values(by="Score (%)", ascending=False).reset_index(drop=True)

        st.session_state.results_df = df
        st.success("All resumes processed and results ready for review.")

        st.markdown("---")
        st.markdown("## 📊 Overall Screening Results")
        st.caption("A glance at the distribution of candidate scores.")

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
        st.session_state.results_df = pd.DataFrame()

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
    st.caption("Candidates you have marked as 'Shortlisted'. (Note: Shortlist status is only for the current session without a database)")

    shortlisted_candidates_session = st.session_state.results_df[st.session_state.results_df['Shortlisted'] == True]

    if not shortlisted_candidates_session.empty:
        for idx, row in shortlisted_candidates_session.iterrows():
            st.write(f"- **{row['Candidate Name']}** ({row['Score (%)']:.1f}%)")
            with st.expander(f"AI Suggestion for {row['Candidate Name']}"):
                st.info(row['AI Suggestion'])
    else:
        st.info("No candidates have been shortlisted yet for the current session.")

    st.markdown("---")
    st.markdown("## 📋 Comprehensive Screening Table")
    st.caption("Review all candidates, their scores, and manage their status.")

    # Data to display will be from st.session_state.results_df (current session only)
    current_session_results_df = st.session_state.results_df.copy()

    for col in ['Score (%)', 'Keyword Match (%)', 'Section Completeness (%)', 'Semantic Similarity (%)', 'Length Score (%)', 'Years Experience']:
        if col in current_session_results_df.columns:
            current_session_results_df[col] = pd.to_numeric(current_session_results_df[col], errors='coerce').fillna(0).round(1)

    current_session_results_df['Tag'] = current_session_results_df.apply(lambda row: "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3 else (
        "✅ Good Fit" if row['Score (%)'] >= 75 else "⚠️ Needs Review"), axis=1)

    edited_df = st.data_editor(
        current_session_results_df,
        column_config={
            "Full Resume Text": st.column_config.Column(
                "Full Resume",
                help="Full text extracted from the resume.",
                width="small",
                display_as="hidden"
            ),
            "AI Suggestion": st.column_config.Column(
                "AI Suggestion",
                help="AI-generated summary and recommendation for the candidate.",
                width="large",
                display_as="expander"
            ),
            "Shortlisted": st.column_config.CheckboxColumn(
                "Shortlisted?",
                help="Mark candidate as shortlisted",
                default=False,
            ),
            "Email": st.column_config.Column(
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
            "Keyword Match (%)": st.column_config.NumberColumn(
                "Keywords (%)",
                help="Percentage of JD keywords found in resume.",
                format="%.1f",
                width="small",
            ),
            "Section Completeness (%)": st.column_config.NumberColumn(
                "Sections (%)",
                help="Completeness of required resume sections.",
                format="%.1f",
                width="small",
            ),
            "Semantic Similarity (%)": st.column_config.NumberColumn(
                "Semantic (%)",
                help="Conceptual similarity between resume and JD.",
                format="%.1f",
                width="small",
            ),
            "Length Score (%)": st.column_config.NumberColumn(
                "Length (%)",
                help="Score based on resume length.",
                format="%.1f",
                width="small",
            ),
        },
        order=("Candidate Name", "Score (%)", "Years Experience", "Email", "AI Suggestion", "Shortlisted", "Keyword Match (%)", "Section Completeness (%)", "Semantic Similarity (%)", "Length Score (%)"),
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        key="screening_results_table"
    )

    # Handle updates from data_editor for session_state only
    if st.session_state.screening_results_table.get('edited_rows') is not None:
        updated_rows_info = st.session_state.screening_results_table['edited_rows']
        if updated_rows_info:
            for index, updates in updated_rows_info.items():
                if 'Shortlisted' in updates:
                    st.session_state.results_df.loc[index, 'Shortlisted'] = updates['Shortlisted']
                    st.toast(f"Updated {st.session_state.results_df.loc[index, 'Candidate Name']} shortlist status to {updates['Shortlisted']}")
                    st.rerun() # Rerun to reflect changes in the UI and shortlisted section

else:
    st.info("Upload a Job Description and Resumes to begin screening.")

# --- Past Screening Results section is REMOVED ---
st.markdown("---")
st.markdown("## ⚠️ Important Note")
st.warning("Database functionality has been removed. Screening results will **not be saved** and 'Past Screening Results' is no longer available. All results are for the current session only.")

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
