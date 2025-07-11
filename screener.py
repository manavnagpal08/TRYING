import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib # Import joblib for loading the ML model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader

# --- 1. Load Models and Resources ---
# Load the ML Screening Model
try:
    ml_screening_model = joblib.load('ml_screening_model.pkl')
    st.sidebar.success("ML Screening Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading ML Screening Model (ml_screening_model.pkl): {e}")
    st.sidebar.warning("Please ensure 'ml_screening_model.pkl' is in your app's directory.")
    ml_screening_model = None # Set to None if loading fails

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
    st.sidebar.success("T5 Model loaded successfully from Hugging Face Hub!")
except Exception as e:
    st.sidebar.error(f"Error loading T5 model from Hugging Face Hub: {e}")
    t5_tokenizer = None
    t5_model = None

# --- Configuration ---
# Define required sections in a resume
REQUIRED_SECTIONS = [
    "experience", "education", "skills", "projects", "certifications",
    "awards", "publications", "extracurricular activities", "volunteer experience"
]

# Define common resume section headers for parsing
SECTION_HEADERS_PATTERN = re.compile(
    r'(?:^|\n)(?P<header>education|experience|skills|projects|certifications|awards|publications|extracurricular activities|volunteer experience|summary|about|profile|contact|interests|languages|references)\b',
    re.IGNORECASE
)

# Placeholder for a TF-IDF vectorizer (will be fitted dynamically)
tfidf_vectorizer = None

# --- Helper Functions ---

def clean_text(text):
    """
    Cleans text by converting to lowercase, removing non-alphanumeric characters,
    and normalizing whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with a single space
    return text

# Function to extract text from PDF using pypdf
def extract_text_from_pdf(pdf_file):
    """
    Extracts text content from an uploaded PDF file.
    """
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
    """
    Extracts common sections from a resume text based on predefined headers.
    """
    sections = {}
    matches = list(SECTION_HEADERS_PATTERN.finditer(text))
    
    for i, match in enumerate(matches):
        header = match.group('header').lower()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        sections[header] = text[start:end].strip()
    return sections

def calculate_keyword_match_score(job_description, resume_text):
    """
    Calculates a TF-IDF based cosine similarity score between job description and resume.
    """
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
    return float(cosine_sim) * 100 # Scale to 0-100

def calculate_section_completeness(resume_sections):
    """
    Calculates a score based on the presence of required sections in the resume.
    """
    completeness_score = sum(1 for sec in REQUIRED_SECTIONS if sec in resume_sections and resume_sections[sec])
    return (completeness_score / len(REQUIRED_SECTIONS) if REQUIRED_SECTIONS else 0.0) * 100 # Scale to 0-100

def calculate_semantic_similarity(job_description, resume_text):
    """
    Calculates semantic similarity using SentenceTransformer embeddings.
    """
    jd_embedding = sentence_model.encode(clean_text(job_description))
    resume_embedding = sentence_model.encode(clean_text(resume_text))
    similarity = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
    return float(similarity) * 100 # Scale to 0-100

def calculate_length_score(resume_text):
    """
    Assigns a score based on the word count of the resume, favoring a moderate length.
    """
    word_count = len(resume_text.split())
    if word_count < 200: return 20 # Too short
    if word_count < 400: return 50 # A bit short
    if word_count < 800: return 100 # Ideal (1-2 pages)
    if word_count < 1200: return 70 # A bit long
    return 30 # Too long, might indicate verbosity

# --- T5 Summarization Function ---
def generate_summary_with_t5(text, max_length=150):
    """
    Generates a summary of the given text using the loaded T5 model.
    """
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

# --- Main Scoring Function (Now uses the ML Model) ---
def get_screening_features_and_score(job_description, resume_text):
    """
    Calculates features and uses the ML model to predict the score.
    """
    if ml_screening_model is None:
        st.error("ML model not loaded. Cannot predict score.")
        return {
            "keyword_match_score": 0.0,
            "section_completeness_score": 0.0,
            "semantic_score": 0.0,
            "length_score": 0.0,
            "predicted_score": 0.0,
            "raw_resume_sections": {}
        }

    cleaned_jd = clean_text(job_description)
    cleaned_resume = clean_text(resume_text)

    # Calculate features (all scaled 0-100)
    keyword_match_score = calculate_keyword_match_score(cleaned_jd, cleaned_resume)
    
    resume_sections = extract_sections(cleaned_resume)
    section_completeness_score = calculate_section_completeness(resume_sections)
    
    semantic_score = calculate_semantic_similarity(cleaned_jd, cleaned_resume)
    
    length_score = calculate_length_score(cleaned_resume)
    
    # Create a DataFrame for prediction (ensure column names match training)
    features = pd.DataFrame([[
        keyword_match_score,
        section_completeness_score,
        semantic_score,
        length_score
    ]], columns=['keyword_match_score', 'section_completeness_score', 'semantic_score', 'length_score'])

    try:
        predicted_score = ml_screening_model.predict(features)[0]
        # Ensure score is within 0-100 range, as the trained model might output outside this range
        predicted_score = max(0.0, min(100.0, predicted_score))
    except Exception as e:
        st.error(f"Error predicting score with ML model: {e}")
        predicted_score = 0.0 # Default to 0 if prediction fails

    return {
        "keyword_match_score": keyword_match_score,
        "section_completeness_score": section_completeness_score,
        "semantic_score": semantic_score,
        "length_score": length_score,
        "predicted_score": predicted_score,
        "raw_resume_sections": resume_sections
    }

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="AI Resume Screener Pro")

st.title("🤖 AI Resume Screener Pro")
st.markdown("Upload a Job Description and Candidate Resumes to get instant screening insights!")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Job Description")
    job_description_file = st.file_uploader("Upload Job Description (TXT, PDF)", type=["txt", "pdf"])
    job_description_text = ""
    if job_description_file:
        if job_description_file.type == "text/plain":
            job_description_text = job_description_file.read().decode("utf-8")
        elif job_description_file.type == "application/pdf":
            job_description_text = extract_text_from_pdf(job_description_file)
            if not job_description_text.strip():
                st.warning("Could not extract text from Job Description PDF. Please try pasting its content manually.")
        st.text_area("Or paste Job Description here:", value=job_description_text, height=300, key="jd_upload_text_area")
    else:
        job_description_text = st.text_area("Paste Job Description here:", height=300, key="jd_text_area")

    # Display T5 Summary for Job Description
    if job_description_text and t5_model:
        with st.expander("T5 Job Description Summary"):
            jd_summary = generate_summary_with_t5(job_description_text)
            st.write(jd_summary)

with col2:
    st.header("Candidate Resumes")
    resume_files = st.file_uploader("Upload Candidate Resumes (TXT, PDF)", type=["txt", "pdf"], accept_multiple_files=True)

if job_description_text and resume_files:
    st.subheader("Screening Results")

    for i, resume_file in enumerate(resume_files):
        st.markdown(f"---")
        st.subheader(f"Candidate {i+1}: {resume_file.name}")

        resume_text = ""
        if resume_file.type == "text/plain":
            resume_text = resume_file.read().decode("utf-8")
        elif resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
            if not resume_text.strip():
                st.warning(f"Could not extract text from PDF {resume_file.name}. Please try pasting its content manually.")
        
        if not resume_text.strip():
            st.info("No content extracted/pasted for this resume. Skipping analysis.")
            continue
        
        # Display T5 Summary for Resume
        if t5_model:
            with st.expander(f"T5 Summary for {resume_file.name}"):
                resume_summary = generate_summary_with_t5(resume_text)
                st.write(resume_summary)

        # Get features and score
        screening_results = get_screening_features_and_score(job_description_text, resume_text)
        
        predicted_score = screening_results["predicted_score"]
        
        # Display scores with color coding
        st.markdown(f"**Predicted Fit Score:** {predicted_score:.2f}/100")
        
        # AI Suggestion based on thresholds
        if predicted_score >= 80:
            st.success("⭐ AI Suggestion: Highly Recommended - Excellent Fit!")
        elif predicted_score >= 60:
            st.info("👍 AI Suggestion: Recommended - Good Fit, Consider for Interview.")
        elif predicted_score >= 40:
            st.warning("⚠️ AI Suggestion: Moderate Fit - Potential, but might need further review.")
        else:
            st.error("❌ AI Suggestion: Low Fit - Not Recommended at this time.")

        # Display detailed features in an expander
        with st.expander("Detailed Feature Scores (0-100 Scale)"):
            st.write(f"Keyword Match Score (TF-IDF): {screening_results['keyword_match_score']:.2f}")
            st.write(f"Section Completeness Score: {screening_results['section_completeness_score']:.2f}")
            st.write(f"Semantic Similarity Score (SentenceTransformer): {screening_results['semantic_score']:.2f}")
            st.write(f"Resume Length Score: {screening_results['length_score']:.2f}")

        # Display extracted sections for debugging/review
        with st.expander("Extracted Resume Sections"):
            if screening_results["raw_resume_sections"]:
                for header, content in screening_results["raw_resume_sections"].items():
                    st.markdown(f"**{header.upper()}:**")
                    st.write(content[:200] + "..." if len(content) > 200 else content)
            else:
                st.write("No distinct sections found.")
    
    st.markdown("---")
    st.success("Screening complete for all uploaded resumes!")

elif not job_description_text and resume_files:
    st.warning("Please provide a Job Description to start screening resumes.")
elif job_description_text and not resume_files:
    st.info("Upload candidate resumes to begin the screening process.")
else:
    st.info("Upload a Job Description and Resumes to begin!")

st.sidebar.header("About")
st.sidebar.info(
    "This AI Resume Screener Pro helps streamline your hiring process by analyzing "
    "job descriptions and candidate resumes. It uses TF-IDF for keyword matching, "
    "SentenceTransformers for semantic similarity, and a trained Machine Learning model "
    "for overall fit prediction. "
    "\n\n**New:** Integrated with a fine-tuned T5 model for concise job description and resume summaries, "
    "and now supports direct PDF text extraction!"
)
