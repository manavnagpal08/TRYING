# train_model.py

import joblib
import numpy as np
import pandas as pd
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# --- Configuration ---
MODEL_SAVE_PATH = "ml_screening_model.pkl"

# Ensure NLTK stopwords are downloaded (if you are using NLTK stopwords in clean_text)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Feature Calculation Helper Functions (Copied from screener.py) ---
# Define required sections (must match screener.py)
REQUIRED_SECTIONS = [
    "experience", "education", "skills", "projects", "certifications",
    "awards", "publications", "extracurricular activities", "volunteer experience"
]

# Define common resume section headers for parsing (must match screener.py)
SECTION_HEADERS_PATTERN = re.compile(
    r'(?:^|\n)(?P<header>education|experience|skills|projects|certifications|awards|publications|extracurricular activities|volunteer experience|summary|about|profile|contact|interests|languages|references)\b',
    re.IGNORECASE
)

# Initialize SentenceTransformer for semantic similarity (must be loaded here too)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Global TF-IDF vectorizer for keyword match score (will be fitted dynamically)
tfidf_vectorizer_global = None

def clean_text(text):
    """
    Cleans text by converting to lowercase, removing non-alphanumeric characters,
    and normalizing whitespace.
    """
    if not isinstance(text, str): # Handle non-string inputs gracefully
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with a single space
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
    global tfidf_vectorizer_global
    if tfidf_vectorizer_global is None:
        tfidf_vectorizer_global = TfidfVectorizer(stop_words='english')

    if not job_description.strip() or not resume_text.strip():
        return 0.0

    documents = [clean_text(job_description), clean_text(resume_text)]
    try:
        tfidf_matrix = tfidf_vectorizer_global.fit_transform(documents)
    except ValueError:
        return 0.0

    if tfidf_matrix.shape[0] < 2:
        return 0.0

    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(cosine_sim) * 100 # Scale to 0-100 for consistency if target is 0-100

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
    if word_count < 200: return 20
    if word_count < 400: return 50
    if word_count < 800: return 100
    if word_count < 1200: return 70
    return 30

# --- Helper to extract score from detailed_ai_suggestion ---
def extract_score_from_suggestion(suggestion_text):
    """Extracts the numerical score from 'detailed_ai_suggestion' string."""
    match = re.search(r'Score:\s*(\d+)%', suggestion_text)
    if match:
        return int(match.group(1))
    return np.nan # Return NaN if no score found


# --- Main Training Script Logic ---

if __name__ == "__main__":
    # --- Load your training data ---
    try:
        df = pd.read_csv(r"C:\Users\manav\Downloads\ai_resume_training_data_450_diverse.csv")
        print(f"Loaded training data with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: 'ai_resume_training_data_450_diverse.csv' not found. Please place your training data CSV in the same directory or provide the correct path.")
        exit()
    except Exception as e:
        print(f"Error loading training data: {e}")
        exit()

    # --- Extract Target Score ---
    print("Extracting target scores from 'detailed_ai_suggestion' column...")
    df['target_score'] = df['detailed_ai_suggestion'].apply(extract_score_from_suggestion)
    df.dropna(subset=['target_score'], inplace=True) # Remove rows where score couldn't be extracted
    
    if df.empty:
        print("No valid target scores found in the dataset. Exiting.")
        exit()
    print(f"Extracted {len(df)} valid target scores.")


    # --- Feature Engineering ---
    print("Extracting features from training data (this may take a while)...")
    features_list = []
    target_scores = []

    for index, row in df.iterrows():
        jd_text = str(row['jd_text']) if not pd.isna(row['jd_text']) else ""
        resume_text = str(row['resume_text']) if not pd.isna(row['resume_text']) else ""
        
        target_score = row['target_score'] # Use the extracted target score
        
        if not jd_text.strip() or not resume_text.strip() or pd.isna(target_score):
            # print(f"Skipping row {index} due to missing job description, resume text, or target score.")
            continue # Skip rows with incomplete data

        # Calculate the 4 features using the functions from screener.py
        keyword_score = calculate_keyword_match_score(jd_text, resume_text)
        cleaned_res_text_for_sections = clean_text(resume_text)
        sections = extract_sections(cleaned_res_text_for_sections)
        completeness_score = calculate_section_completeness(sections)
        semantic_score = calculate_semantic_similarity(jd_text, resume_text)
        length_score = calculate_length_score(resume_text)
        
        features_list.append([keyword_score, completeness_score, semantic_score, length_score])
        target_scores.append(target_score)

    if not features_list:
        print("No valid features extracted from the training data. Check your CSV and data parsing.")
        exit()

    # Convert features and target scores to pandas objects
    X = pd.DataFrame(features_list, columns=['keyword_match_score', 'section_completeness_score', 'semantic_score', 'length_score'])
    y = pd.Series(target_scores)

    print(f"Successfully extracted {X.shape[1]} features for {X.shape[0]} training samples.")
    
    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Hyperparameter Tuning with GridSearchCV ---
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')

    print("\nStarting GridSearchCV for hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    print("RandomForestRegressor trained with best hyperparameters.")
    print(f"Best parameters found: {grid_search.best_params_}")

    # --- Model Evaluation ---
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation on Test Set:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print(f"Features used for training: {model.n_features_in_}")
    if hasattr(model, 'feature_names_in_'):
        print(f"Feature names: {model.feature_names_in_}")

    # --- Save the Trained Model ---
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\nNew trained model saved to {MODEL_SAVE_PATH}")
    print(f"This model was trained on {model.n_features_in_} features. This should be 4.")