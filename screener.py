import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import hashlib # For hashing job descriptions

# --- Database Configuration ---
DATABASE_FILE = "screening_data.db"
# Ensure the database is initialized on app startup
@st.cache_resource
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
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
            full_resume_text TEXT
        )
    ''')
    conn.commit()
    conn.close()
    return True

# Initialize database
db_initialized = init_db()
if db_initialized:
    st.sidebar.success("Database initialized and ready.")
else:
    st.sidebar.error("Failed to initialize database.")


# --- 1. Load Models and Resources ---
# Load the ML Screening Model
try:
    ml_screening_model = joblib.load('ml_screening_model.pkl')
except Exception as e:
    st.error(f"Error loading ML Screening Model (ml_screening_model.pkl): {e}. Please ensure it's in your app's directory.")
    ml_screening_model = None

# Load the Sentence Transformer model for semantic similarity
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.error(f"Error loading SentenceTransformer model: {e}")
    st.stop()

# --- Load the Fine-tuned T5 Model from Hugging Face Hub ---
T5_REPO_ID = "mnagpal/fine-tuned-t5-resume-screener"
try:
    t5_tokenizer = AutoTokenizer.from_pretrained(T5_REPO_ID)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_REPO_ID)
except Exception as e:
    st.error(f"Error loading T5 model from Hugging Face Hub: {e}")
    t5_tokenizer = None
    t5_model = None

# --- Configuration ---
REQUIRED_SECTIONS = [
    "experience", "education", "skills", "projects", "certifications",
    "awards", "publications", "extracurricular activities", "volunteer experience"
]

SECTION_HEADERS_PATTERN = re.compile(
    r'(?:^|\n)(?P<header>education|experience|skills|projects|certifications|awards|publications|extracurricular activities|volunteer experience|summary|about|profile|contact|interests|languages|references)\b',
    re.IGNORECASE
)

tfidf_vectorizer = None

# --- Helper Functions (Same as before) ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    return text

def extract_sections(text):
    sections = {}
    matches = list(SECTION_HEADERS_PATTERN.finditer(text))
    for i, match in enumerate(matches):
        header = match.group('header').lower()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        sections[header] = text[start:end].strip()
    return sections

def calculate_keyword_match_score(job_description, resume_text):
    global tfidf_vectorizer
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    if not job_description.strip() or not resume_text.strip():
        return 0.0
    documents = [clean_text(job_description), clean_text(resume_text)]
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    except ValueError:
        return 0.0
    if tfidf_matrix.shape[0] < 2:
        return 0.0
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(cosine_sim) * 100

def calculate_section_completeness(resume_sections):
    completeness_score = sum(1 for sec in REQUIRED_SECTIONS if sec in resume_sections and resume_sections[sec])
    return (completeness_score / len(REQUIRED_SECTIONS) if REQUIRED_SECTIONS else 0.0) * 100

def calculate_semantic_similarity(job_description, resume_text):
    jd_embedding = sentence_model.encode(clean_text(job_description))
    resume_embedding = sentence_model.encode(clean_text(resume_text))
    similarity = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
    return float(similarity) * 100

def calculate_length_score(resume_text):
    word_count = len(resume_text.split())
    if word_count < 200: return 20
    if word_count < 400: return 50
    if word_count < 800: return 100
    if word_count < 1200: return 70
    return 30

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

# --- Main Scoring Function ---
def get_screening_features_and_score(job_description, resume_text):
    if ml_screening_model is None:
        st.error("ML model not loaded. Cannot predict score.")
        return {
            "keyword_match_score": 0.0, "section_completeness_score": 0.0,
            "semantic_score": 0.0, "length_score": 0.0,
            "predicted_score": 0.0, "raw_resume_sections": {}
        }

    cleaned_jd = clean_text(job_description)
    cleaned_resume = clean_text(resume_text)

    keyword_match_score = calculate_keyword_match_score(cleaned_jd, cleaned_resume)
    resume_sections = extract_sections(cleaned_resume)
    section_completeness_score = calculate_section_completeness(resume_sections)
    semantic_score = calculate_semantic_similarity(cleaned_jd, cleaned_resume)
    length_score = calculate_length_score(cleaned_resume)
    
    features = pd.DataFrame([[
        keyword_match_score,
        section_completeness_score,
        semantic_score,
        length_score
    ]], columns=['keyword_match_score', 'section_completeness_score', 'semantic_score', 'length_score'])

    try:
        predicted_score = ml_screening_model.predict(features)[0]
        predicted_score = max(0.0, min(100.0, predicted_score))
    except Exception as e:
        st.error(f"Error predicting score with ML model: {e}")
        predicted_score = 0.0

    return {
        "keyword_match_score": keyword_match_score,
        "section_completeness_score": section_completeness_score,
        "semantic_score": semantic_score,
        "length_score": length_score,
        "predicted_score": predicted_score,
        "raw_resume_sections": resume_sections
    }

# --- Database Operations ---
def get_jd_hash(jd_text):
    return hashlib.md5(jd_text.encode('utf-8')).hexdigest()

def insert_or_update_screening_result(jd_hash, jd_summary, candidate_name, scores, full_resume_text, shortlisted=False):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    
    # Check if a record for this candidate under this JD already exists
    c.execute('''
        SELECT id FROM results 
        WHERE job_description_hash = ? AND candidate_name = ?
    ''', (jd_hash, candidate_name))
    existing_id = c.fetchone()

    if existing_id:
        # Update existing record
        c.execute('''
            UPDATE results SET
                timestamp = CURRENT_TIMESTAMP,
                predicted_score = ?,
                keyword_match = ?,
                section_completeness = ?,
                semantic_similarity = ?,
                length_score = ?,
                shortlisted = ?,
                full_resume_text = ?
            WHERE id = ?
        ''', (scores['predicted_score'], scores['keyword_match_score'], 
              scores['section_completeness_score'], scores['semantic_score'], 
              scores['length_score'], shortlisted, full_resume_text, existing_id[0]))
    else:
        # Insert new record
        c.execute('''
            INSERT INTO results (
                job_description_hash, job_description_summary, candidate_name, 
                predicted_score, keyword_match, section_completeness, 
                semantic_similarity, length_score, shortlisted, full_resume_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (jd_hash, jd_summary, candidate_name, scores['predicted_score'], 
              scores['keyword_match_score'], scores['section_completeness_score'], 
              scores['semantic_score'], scores['length_score'], shortlisted, full_resume_text))
    conn.commit()
    conn.close()

def get_screening_results_from_db(jd_hash=None):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    if jd_hash:
        c.execute('SELECT * FROM results WHERE job_description_hash = ? ORDER BY predicted_score DESC', (jd_hash,))
    else:
        c.execute('SELECT * FROM results ORDER BY predicted_score DESC') # Get all historical data
    
    columns = [description[0] for description in c.description]
    results = c.fetchall()
    conn.close()
    
    df = pd.DataFrame(results, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp']) # Convert timestamp to datetime objects
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


# --- Email Sending Function ---
def send_email(recipient_email, subject, body):
    sender_email = st.secrets["email"]["username"]  # Use Streamlit Secrets
    sender_password = st.secrets["email"]["password"] # Use Streamlit Secrets
    
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
        st.error(f"Failed to send email. Error: {e}. Please check your Streamlit Secrets for email credentials and ensure correct app password/less secure app access for your email provider.")
        return False

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="AI Resume Screener Pro")

st.title("🤖 AI Resume Screener Pro")
st.markdown("Upload a Job Description and Candidate Resumes to get instant screening insights!")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Job Description")
    job_description_file = st.file_uploader("Upload Job Description (TXT, PDF)", type=["txt", "pdf"])
    job_description_text = ""
    jd_summary_text = "" # To store JD summary for DB

    if job_description_file:
        if job_description_file.type == "text/plain":
            job_description_text = job_description_file.read().decode("utf-8")
        elif job_description_file.type == "application/pdf":
            job_description_text = extract_text_from_pdf(job_description_file)
            if not job_description_text.strip():
                st.warning("Could not extract text from Job Description PDF. Please try pasting its content manually.")
        
        # Use text_area with value from file if uploaded, otherwise allow pasting
        job_description_text = st.text_area("Or paste Job Description here:", value=job_description_text, height=300, key="jd_upload_text_area")
    else:
        # If no file uploaded, allow direct pasting
        job_description_text = st.text_area("Paste Job Description here:", height=300, key="jd_text_area")

    if job_description_text and t5_model:
        with st.expander("T5 Job Description Summary"):
            jd_summary_text = generate_summary_with_t5(job_description_text)
            st.write(jd_summary_text)
    
    # Generate JD hash after text is finalized
    current_jd_hash = get_jd_hash(job_description_text) if job_description_text else None

with col2:
    st.header("Candidate Resumes")
    resume_files = st.file_uploader("Upload Candidate Resumes (TXT, PDF)", type=["txt", "pdf"], accept_multiple_files=True)


if job_description_text and resume_files:
    st.subheader("Screening Results for Current JD")

    # Process and save each resume to DB
    for i, resume_file in enumerate(resume_files):
        resume_text = ""
        if resume_file.type == "text/plain":
            resume_text = resume_file.read().decode("utf-8")
        elif resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        
        if not resume_text.strip():
            st.info(f"No content extracted/pasted for {resume_file.name}. Skipping analysis.")
            continue
        
        screening_results = get_screening_features_and_score(job_description_text, resume_text)
        
        # Save or update result in DB
        insert_or_update_screening_result(
            jd_hash=current_jd_hash,
            jd_summary=jd_summary_text,
            candidate_name=resume_file.name,
            scores=screening_results,
            full_resume_text=resume_text,
            shortlisted=False # Default to not shortlisted on initial upload
        )
    
    st.success("All resumes processed and results saved to database.")

    # --- Display Results Table for CURRENT JD ---
    current_jd_results_df = get_screening_results_from_db(jd_hash=current_jd_hash)

    if not current_jd_results_df.empty:
        st.subheader("Summary of Candidates for This Job Description")
        # Select and rename columns for display
        df_display = current_jd_results_df[[
            'candidate_name', 'predicted_score', 'keyword_match', 
            'section_completeness', 'semantic_similarity', 'length_score', 'shortlisted'
        ]].rename(columns={
            'candidate_name': 'Candidate Name',
            'predicted_score': 'Predicted Score',
            'keyword_match': 'Keyword Match',
            'section_completeness': 'Section Completeness',
            'semantic_similarity': 'Semantic Similarity',
            'length_score': 'Length Score',
            'shortlisted': 'Shortlisted?'
        })
        
        # Format scores to 2 decimal places
        for col in ['Predicted Score', 'Keyword Match', 'Section Completeness', 'Semantic Similarity', 'Length Score']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")

        st.dataframe(df_display.style.highlight_max(subset=['Predicted Score'], axis=0, color='lightgreen'))

        # --- Top Candidate Justification ---
        if not current_jd_results_df.empty:
            top_candidate_data = current_jd_results_df.iloc[0] # Already sorted by score
            st.subheader(f"🏆 Top Candidate: {top_candidate_data['candidate_name']} (Score: {top_candidate_data['predicted_score']:.2f}/100)")
            
            justification = f"**{top_candidate_data['candidate_name']}** is the top candidate with a score of **{top_candidate_data['predicted_score']:.2f}**."
            justification += f" This is primarily due to their strong **semantic similarity ({top_candidate_data['semantic_similarity']:.2f})** and **keyword match ({top_candidate_data['keyword_match']:.2f})** with the job description,"
            justification += f" indicating a deep relevance of their skills and experience. "
            if top_candidate_data['section_completeness'] > 70:
                justification += f"Their resume also shows good **section completeness ({top_candidate_data['section_completeness']:.2f})**, suggesting a comprehensive profile."
            else:
                justification += f"While section completeness is {top_candidate_data['section_completeness']:.2f}, their high content match outweighs this."
            st.markdown(justification)

        st.markdown("---") # Separator

        # --- Detailed Candidate Views & Shortlisting ---
        st.subheader("Detailed Candidate Insights")
        for idx, candidate_row in current_jd_results_df.iterrows():
            candidate_name = candidate_row['candidate_name']
            predicted_score = candidate_row['predicted_score']
            full_resume_text = candidate_row['full_resume_text']

            st.markdown(f"**{candidate_name} (Score: {predicted_score:.2f}/100)**")
            
            col_detail1, col_detail2 = st.columns([1,1])
            with col_detail1:
                if t5_model:
                    with st.expander(f"T5 Summary for {candidate_name}"):
                        resume_summary = generate_summary_with_t5(full_resume_text)
                        st.write(resume_summary)

                with st.expander("Detailed Feature Scores"):
                    st.write(f"Keyword Match Score: {candidate_row['keyword_match']:.2f}")
                    st.write(f"Section Completeness Score: {candidate_row['section_completeness']:.2f}")
                    st.write(f"Semantic Similarity Score: {candidate_row['semantic_similarity']:.2f}")
                    st.write(f"Resume Length Score: {candidate_row['length_score']:.2f}")
            
            with col_detail2:
                current_shortlisted_status = candidate_row['shortlisted']
                
                # Checkbox for shortlisting
                if st.checkbox(f"Shortlist {candidate_name}", value=current_shortlisted_status, key=f"shortlist_cb_{candidate_name}_{current_jd_hash}"):
                    if not current_shortlisted_status: # Only update if status changed
                        update_shortlist_status_in_db(current_jd_hash, candidate_name, True)
                        st.success(f"{candidate_name} shortlisted!")
                        # Rerun to update the display if needed, or rely on next interaction
                else: # Checkbox is unchecked
                    if current_shortlisted_status: # Only update if status changed
                        update_shortlist_status_in_db(current_jd_hash, candidate_name, False)
                        st.warning(f"{candidate_name} unshortlisted.")
                        # Rerun to update the display if needed, or rely on next interaction


            st.markdown("---") # Separator after each candidate's detailed view

    # --- Shortlisted Candidates Section (from DB) ---
    st.subheader("🌟 Your Shortlisted Candidates (Persistent)")
    shortlisted_from_db = get_screening_results_from_db(jd_hash=current_jd_hash)
    shortlisted_from_db = shortlisted_from_db[shortlisted_from_db['shortlisted'] == True]

    if not shortlisted_from_db.empty:
        for idx, row in shortlisted_from_db.iterrows():
            st.write(f"- {row['candidate_name']} (Score: {row['predicted_score']:.2f})")
        
        # Email functionality
        st.markdown("---")
        st.subheader("✉️ Send Email to Shortlisted Candidates")
        recipient_emails_input = st.text_input("Enter recipient email(s) (comma-separated):", key="recipient_emails")
        email_subject = st.text_input("Email Subject:", value=f"Job Application Update: {job_description_text.splitlines()[0][:50] if job_description_text else 'Your Application'}", key="email_subject")
        email_body = st.text_area("Email Body:", height=200, key="email_body",
            value="Dear candidate,\n\nThank you for your application. We are pleased to inform you that you have been shortlisted for further consideration.\n\nBest regards,\n[Your Company Name]"
        )
        
        if st.button("Send Emails to Shortlisted Candidates"):
            if not recipient_emails_input:
                st.error("Please enter at least one recipient email address.")
            else:
                recipient_list = [e.strip() for e in recipient_emails_input.split(',')]
                all_sent = True
                for email in recipient_list:
                    if send_email(email, email_subject, email_body):
                        st.success(f"Email sent successfully to {email}!")
                    else:
                        all_sent = False
                        st.error(f"Failed to send email to {email}.")
                if all_sent:
                    st.success("All emails sent!")
    else:
        st.info("No candidates have been shortlisted yet for this job description.")

elif not job_description_text and resume_files:
    st.warning("Please provide a Job Description to start screening resumes.")
elif job_description_text and not resume_files:
    st.info("Upload candidate resumes to begin the screening process.")
else:
    st.info("Upload a Job Description and Resumes to begin!")

# --- About Section (Moved to main content area) ---
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
st.header("📊 Analytics Dashboard (Coming Soon!)")
st.info("This section will eventually display historical screening data and insights from the database.")
if st.button("View All Historical Screening Data"):
    all_data = get_screening_results_from_db()
    if not all_data.empty:
        st.dataframe(all_data)
    else:
        st.info("No historical data found in the database.")
