import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
import joblib
import numpy as np # Ensure numpy is imported for np.nan
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
import collections
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse # For encoding mailto links

# Import T5 specific libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Import skills data from a separate file ---
# Ensure 'skills_data.py' is in the same directory as this script
from skills_data import ALL_SKILLS_MASTER_SET, SORTED_MASTER_SKILLS, CUSTOM_STOP_WORDS

# --- Load Embedding + ML Model ---
@st.cache_resource
def load_ml_model():
    """Loads the SentenceTransformer model for embeddings and a pre-trained ML screening model."""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Ensure ml_screening_model.pkl is trained with predict_proba capability (e.g., RandomForestClassifier, XGBClassifier)
        ml_model = joblib.load("ml_screening_model.pkl")
        
        # --- IMPORTANT CHECK FOR predict_proba ---
        if not hasattr(ml_model, 'predict_proba'):
            st.error(f"❌ Loaded ML model ({type(ml_model)}) does not have 'predict_proba' method. Please ensure 'ml_screening_model.pkl' is a classifier trained to output probabilities (e.g., RandomForestClassifier, XGBClassifier).")
            return None, None
        # --- End IMPORTANT CHECK ---

        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory and network is available for SentenceTransformer.")
        return None, None

# --- Load T5 Model ---
@st.cache_resource
def load_t5_model():
    """Loads a pre-trained T5 model for resume summarization from Hugging Face Hub."""
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
model, ml_model = load_ml_model()
t5_tokenizer, t5_model = load_t5_model()

# --- Helper Functions ---

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None
    return text

def clean_text(text):
    """Cleans text by removing special characters, extra spaces, and converting to lowercase."""
    if text is None:
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space
    return text.lower() # Convert to lowercase

def extract_skills(text, job_description_skills=None):
    """
    Extracts skills from text using a master list and job description skills for prioritization.
    Prioritizes multi-word skills and uses a custom stop word list.
    """
    if text is None:
        return set()

    extracted = set()
    text_lower = text.lower()

    # Create a combined set of skills from master list and JD for efficient lookup
    # Prioritize skills from JD if provided, by adding them to the front of the sorted list
    search_skills = SORTED_MASTER_SKILLS
    if job_description_skills:
        jd_skills_lower = {s.lower() for s in job_description_skills}
        # Add JD skills to the search list, ensuring they are also in the master set
        # and sort them by length to prioritize multi-word matches
        jd_specific_sorted_skills = sorted([s for s in jd_skills_lower if s in ALL_SKILLS_MASTER_SET], key=len, reverse=True)
        # Combine, ensuring no duplicates and maintaining priority for JD skills
        search_skills = sorted(list(set(jd_specific_sorted_skills + SORTED_MASTER_SKILLS)), key=len, reverse=True)


    # First pass: Look for exact matches of multi-word skills (longest first)
    # This helps prevent matching "java" if "javascript" is present
    for skill in search_skills:
        if len(skill.split()) > 1: # Only consider multi-word skills here
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                extracted.add(skill)

    # Second pass: Look for single-word skills, avoiding those already found as part of multi-word skills
    # and filtering out common stop words
    words = set(re.findall(r'\b\w+\b', text_lower))
    for word in words:
        if word in ALL_SKILLS_MASTER_SET and word not in CUSTOM_STOP_WORDS:
            is_part_of_multi_word = False
            for multi_word_skill in extracted:
                if len(multi_word_skill.split()) > 1 and word in multi_word_skill.split():
                    is_part_of_multi_word = True
                    break
            if not is_part_of_multi_word:
                extracted.add(word)

    return extracted


def calculate_skill_match(job_skills, resume_skills):
    """Calculates matched and missing skills."""
    matched_skills = job_skills.intersection(resume_skills)
    missing_skills = job_skills.difference(resume_skills)
    return list(matched_skills), list(missing_skills)

def get_ai_suggestion(resume_text, job_description_text):
    """Generates an AI suggestion using the loaded T5 model."""
    if t5_tokenizer is None or t5_model is None:
        return "AI suggestion model not loaded."

    # Ensure texts are not None before cleaning
    resume_text = resume_text if resume_text is not None else ""
    job_description_text = job_description_text if job_description_text is not None else ""

    # Concatenate texts for the model input
    input_text = f"summarize resume for job description: {job_description_text} \\n resume: {resume_text}"

    # Tokenize and generate
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = t5_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def get_ai_overall_assessment(resume_text, job_description_text):
    """Generates an overall AI assessment and recommendation using the loaded T5 model."""
    if t5_tokenizer is None or t5_model is None:
        return "AI assessment model not loaded."

    # Ensure texts are not None before cleaning
    resume_text = resume_text if resume_text is not None else ""
    job_description_text = job_description_text if job_description_text is not None else ""

    # Concatenate texts for the model input
    input_text = f"assess resume suitability for job description: {job_description_text} \\n resume: {resume_text}"

    # Tokenize and generate
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    assessment_ids = t5_model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    assessment = t5_tokenizer.decode(assessment_ids[0], skip_special_tokens=True)
    return assessment

# --- Experience Extraction (Improved Regex) ---
def extract_experience(text):
    """
    Extracts total years of experience from text using improved regex patterns.
    Looks for "X years", "X+ years", "X-Y years", "X years of experience", etc.
    """
    if text is None:
        return 0

    text_lower = text.lower()
    total_experience = 0

    # Pattern 1: "X years" or "X+ years" or "X-Y years" where X, Y are numbers
    # Prioritize patterns that explicitly mention "years of experience" or similar
    patterns = [
        r'(\d+\.?\d*)\s*(?:plus|\+)?\s*years\s+of\s+experience',
        r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*years', # e.g., "5-7 years"
        r'(\d+\.?\d*)\s*(?:to|-)\s*(\d+\.?\d*)\s*yrs', # e.g., "5 to 7 yrs"
        r'(\d+\.?\d*)\s*(?:plus|\+)?\s*years?', # e.g., "5 years", "5+ years", "5 yrs"
        r'(\d+\.?\d*)\s*yrs?'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple): # For patterns with groups like "X-Y years"
                try:
                    # Take the upper bound or average for a range
                    exp_start = float(match[0])
                    exp_end = float(match[1]) if len(match) > 1 and match[1] else exp_start
                    total_experience = max(total_experience, exp_end) # Take the higher end of the range
                except ValueError:
                    continue
            else: # For single number patterns
                try:
                    total_experience = max(total_experience, float(match))
                except ValueError:
                    continue
    
    # Heuristic: If "experience" is mentioned multiple times, and a number is near it,
    # but no clear "X years" pattern, this might catch it.
    # This is a fallback and can be less accurate.
    if total_experience == 0:
        exp_keywords = ["experience", "exp"]
        for keyword in exp_keywords:
            # Look for numbers preceding or following the keyword within a small window
            # e.g., "5 years experience", "experience of 3 years"
            match = re.search(r'(\d+\.?\d*)\s*(?:year|yr)s?\s+' + re.escape(keyword), text_lower)
            if match:
                try:
                    total_experience = max(total_experience, float(match.group(1)))
                except ValueError:
                    pass
            match = re.search(re.escape(keyword) + r'\s+(?:of\s+)?(\d+\.?\d*)\s*(?:year|yr)s?', text_lower)
            if match:
                try:
                    total_experience = max(total_experience, float(match.group(1)))
                except ValueError:
                    pass

    return total_experience


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="ScreenerPro - AI Resume Screener")

st.title("🚀 ScreenerPro: AI-Powered Resume Screener")
st.markdown("""
Welcome to ScreenerPro! Upload a Job Description (JD) and multiple resumes to get AI-driven insights,
skill matching, and overall suitability assessments for each candidate.
""")

# --- Main Content Area - Uploads and Settings ---
st.header("Upload Files")
job_description_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
resume_files = st.file_uploader("Upload Resumes (PDFs)", type=["pdf"], accept_multiple_files=True)

st.header("Settings")
# Adjusted tagging thresholds
tag_threshold_highly_suitable = st.slider("Score Threshold for 'Highly Suitable'", 0.7, 1.0, 0.85, 0.01)
tag_threshold_suitable = st.slider("Score Threshold for 'Suitable'", 0.5, 0.8, 0.70, 0.01)
tag_threshold_needs_review = st.slider("Score Threshold for 'Needs Review'", 0.3, 0.6, 0.50, 0.01)

# Ensure thresholds are in logical order
if tag_threshold_suitable >= tag_threshold_highly_suitable:
    st.warning(" 'Suitable' threshold should be less than 'Highly Suitable' threshold.")
    tag_threshold_suitable = tag_threshold_highly_suitable - 0.05
if tag_threshold_needs_review >= tag_threshold_suitable:
    st.warning(" 'Needs Review' threshold should be less than 'Suitable' threshold.")
    tag_threshold_needs_review = tag_threshold_suitable - 0.05

# --- Main Content Area ---
if job_description_file and resume_files:
    st.subheader("Processing Files...")

    # Process Job Description
    job_description_text = extract_text_from_pdf(job_description_file)
    if job_description_text is None:
        st.error("Could not process Job Description. Please try another file.")
        st.stop()
    cleaned_jd_text = clean_text(job_description_text)
    jd_skills = extract_skills(cleaned_jd_text)

    # Generate JD embedding
    jd_embedding = None
    if model:
        jd_embedding = model.encode([cleaned_jd_text])
    else:
        st.warning("Sentence Transformer model not loaded. Similarity scores will not be calculated.")

    st.success(f"✅ Job Description processed. Found {len(jd_skills)} key skills.")
    with st.expander("View Extracted Job Description Skills"):
        st.write(", ".join(jd_skills) if jd_skills else "No skills extracted from JD.")

    # Process Resumes
    results = []
    for i, resume_file in enumerate(resume_files):
        with st.spinner(f"Processing {resume_file.name}..."):
            resume_text = extract_text_from_pdf(resume_file)
            if resume_text is None:
                st.warning(f"Skipping {resume_file.name} due to processing error.")
                continue

            cleaned_resume_text = clean_text(resume_text)
            resume_skills = extract_skills(cleaned_resume_text, jd_skills)
            matched_skills, missing_skills = calculate_skill_match(jd_skills, resume_skills)
            experience = extract_experience(resume_text) # Use raw text for experience extraction

            # Calculate similarity score
            similarity_score = 0.0
            if model and jd_embedding is not None:
                resume_embedding = model.encode([cleaned_resume_text])
                similarity_score = cosine_similarity(jd_embedding, resume_embedding)[0][0]
                # Ensure score is between 0 and 1, sometimes cosine_similarity can slightly exceed 1 due to floating point
                similarity_score = max(0.0, min(1.0, similarity_score))

            # Predict suitability using ML model
            ml_prediction_proba = None
            ml_tag = "N/A"
            if ml_model and model and jd_embedding is not None:
                try:
                    # Original combined embedding from JD and resume text
                    base_combined_embedding = np.concatenate((jd_embedding[0], resume_embedding[0]))
                    
                    # Additional scalar features
                    additional_features = np.array([
                        experience,
                        len(matched_skills),
                        len(missing_skills),
                        similarity_score
                    ])
                    
                    # Concatenate all features
                    # Ensure additional_features are reshaped to (1, N) if they are 1D
                    # and base_combined_embedding is already (1, M) or can be flattened
                    # Here, base_combined_embedding is (768,) and additional_features is (4,)
                    # So, we concatenate them to get (772,) and then reshape to (1, 772)
                    final_features_for_ml = np.concatenate((base_combined_embedding, additional_features)).reshape(1, -1)

                    # Get prediction probabilities for "suitable" class (assuming 1 is suitable)
                    if 1 in ml_model.classes_:
                        proba_index = list(ml_model.classes_).index(1)
                        ml_prediction_proba = ml_model.predict_proba(final_features_for_ml)[0][proba_index]
                        
                        # Tagging logic based on ML prediction probability and experience
                        if ml_prediction_proba >= tag_threshold_highly_suitable and experience >= 3: # Example: 3+ years for highly suitable
                            ml_tag = "Highly Suitable"
                        elif ml_prediction_proba >= tag_threshold_suitable and experience >= 1: # Example: 1+ year for suitable
                            ml_tag = "Suitable"
                        elif ml_prediction_proba >= tag_threshold_needs_review:
                            ml_tag = "Needs Review"
                        else:
                            ml_tag = "Not Suitable"
                    else:
                        st.warning("ML model does not have a '1' class for suitability prediction. Tagging will be limited.")
                        ml_tag = "Cannot Tag (Model Class Missing)"
                except Exception as e:
                    st.error(f"Error during ML prediction for {resume_file.name}: {e}")
                    ml_tag = "Error in ML Prediction"
            else:
                ml_tag = "ML Model Not Loaded"


            # Generate AI suggestions and overall assessment
            ai_suggestion = get_ai_suggestion(resume_text, job_description_text)
            ai_overall_assessment = get_ai_overall_assessment(resume_text, job_description_text)

            results.append({
                "Resume Name": resume_file.name,
                "Experience (Years)": experience,
                "Similarity Score": similarity_score, # Changed to float directly
                "ML Prediction Probability": ml_prediction_proba if ml_prediction_proba is not None else np.nan, # Changed to np.nan for ProgressColumn
                "Tag": ml_tag,
                "AI Suggestion": ai_suggestion,
                "Overall Assessment and Recommendation": ai_overall_assessment,
                "Matched Keywords": ", ".join(matched_skills),
                "Missing Skills": ", ".join(missing_skills),
                "Raw Resume Text": resume_text # Keep raw text for full view
            })

    st.subheader("Screening Results")

    if results:
        df = pd.DataFrame(results)
        
        # Add a download button for the results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        # Display results in an interactive table
        st.dataframe(
            df[[
                "Resume Name", "Experience (Years)", "Similarity Score", "ML Prediction Probability", "Tag",
                "AI Suggestion", "Overall Assessment and Recommendation", "Matched Keywords", "Missing Skills"
            ]],
            hide_row_index=True,
            column_config={
                "Resume Name": st.column_config.Column(
                    "Resume Name",
                    help="Name of the uploaded resume file",
                    width="small"
                ),
                "Experience (Years)": st.column_config.NumberColumn(
                    "Experience (Years)",
                    help="Total years of experience extracted from the resume",
                    format="%d",
                    width="small"
                ),
                "Similarity Score": st.column_config.ProgressColumn(
                    "Similarity Score",
                    help="Cosine similarity between resume and JD embeddings (0.00 - 1.00)",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                    width="small"
                ),
                "ML Prediction Probability": st.column_config.ProgressColumn(
                    "ML Prediction Probability",
                    help="Probability of suitability predicted by the ML model (0.00 - 1.00)",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                    width="small"
                ),
                "AI Suggestion": st.column_config.Column(
                    "AI Suggestion",
                    help="AI-generated summary/suggestion for the resume based on JD",
                    width="large"
                ),
                "Overall Assessment and Recommendation": st.column_config.Column(
                    "Overall Assessment and Recommendation",
                    help="AI-generated overall assessment and recommendation",
                    width="large" # Make it wider to show more text
                ),
                "Matched Keywords": st.column_config.Column(
                    "Matched Keywords",
                    help="Keywords from JD found in resume",
                    width="medium"
                ),
                "Missing Skills": st.column_config.Column(
                    "Missing Skills",
                    help="Keywords from JD not found in resume",
                    width="medium"
                ),
                "Tag": st.column_config.Column(
                    "Tag",
                    help="Categorization based on score and experience",
                    width="small"
                )
            }
        )

        st.subheader("Detailed Resume Views")
        for idx, row in df.iterrows():
            with st.expander(f"View Details for: {row['Resume Name']}"):
                st.write(f"**Experience (Years):** {row['Experience (Years)']}")
                st.write(f"**Similarity Score:** {row['Similarity Score']:.2f}") # Format for display
                st.write(f"**ML Prediction Probability:** {row['ML Prediction Probability']:.2f}" if not np.isnan(row['ML Prediction Probability']) else "N/A") # Format for display
                st.write(f"**Tag:** {row['Tag']}")
                st.write(f"**AI Suggestion:** {row['AI Suggestion']}")
                st.write(f"**Overall Assessment and Recommendation:** {row['Overall Assessment and Recommendation']}")
                st.write(f"**Matched Keywords:** {row['Matched Keywords']}")
                st.write(f"**Missing Skills:** {row['Missing Skills']}")
                st.markdown("---")
                st.subheader("Raw Resume Text")
                st.text_area(f"Full Text of {row['Resume Name']}", row['Raw Resume Text'], height=300)

                # Generate Word Cloud for the resume
                st.subheader(f"Word Cloud for {row['Resume Name']}")
                words_for_cloud = " ".join([word for word in row['Raw Resume Text'].lower().split() if word.isalnum() and word not in CUSTOM_STOP_WORDS])
                if words_for_cloud:
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words_for_cloud)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("Not enough text to generate a word cloud.")

                # Option to send email (mailto link)
                st.subheader("Candidate Actions")
                candidate_name = os.path.splitext(row['Resume Name'])[0] # Simple extraction
                email_subject = urllib.parse.quote(f"Regarding your application for [Job Title] - {candidate_name}")
                email_body = urllib.parse.quote(f"Dear {candidate_name},\n\nThank you for your application for the [Job Title] position. We have reviewed your resume and would like to provide some feedback.\n\n[Insert AI Suggestion/Assessment here, or custom message]\n\nBest regards,\n[Your Name/Company]")
                
                st.markdown(f"**Contact {candidate_name}:** [Send Email](mailto:?subject={email_subject}&body={email_body})")
                
                # Placeholder for "Move to Next Stage" button
                st.button(f"Move {candidate_name} to Next Stage", key=f"next_stage_{idx}")

    else:
        st.info("No resumes were successfully processed or no results to display.")
else:
    st.info("Upload a Job Description and Resumes to begin screening.")

# --- About Section (Moved back to sidebar) ---
st.sidebar.title("About ScreenerPro")
st.sidebar.info(
    "ScreenerPro is an AI-powered application designed to streamline the resume screening "
    "process. It leverages a custom-trained Machine Learning model, a Sentence Transformer for "
    "semantic understanding, and a fine-tuned T5 model for insightful AI suggestions and summarization.\n\n"
    "Upload job descriptions and resumes, and let AI assist you in identifying the best-fit candidates!"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Developed by [Manav Nagpal](https://www.linkedin.com/in/manav-nagpal-b03a743b/)"
)
