import joblib
import numpy as np
import pandas as pd
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import nltk
import collections

# --- Configuration ---
MODEL_SAVE_PATH = "ml_screening_model.pkl"

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Stop Words List (MUST MATCH screener.py) ---
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

# --- REQUIRED SECTIONS and PATTERN (MUST MATCH screener.py) ---
REQUIRED_SECTIONS = [
    "experience", "education", "skills", "projects", "certifications",
    "awards", "publications", "extracurricular activities", "volunteer experience"
]

SECTION_HEADERS_PATTERN = re.compile(
    r'(?:^|\n)(?P<header>education|experience|skills|projects|certifications|awards|publications|extracurricular activities|volunteer experience|summary|about|profile|contact|interests|languages|references)\b',
    re.IGNORECASE
)

# --- Helper Functions (Copied from ScreenerPro for consistency) ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
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


# --- Main Training Script ---
if __name__ == "__main__":
    print("Loading SentenceTransformer model...")
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded successfully!")
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Please ensure you have an active internet connection and the model can be downloaded.")
        exit()

    print("Loading training data...")
    # IMPORTANT: Ensure 'ai_resume_training_data_450_diverse.csv' is in the same directory
    try:
        df = pd.read_csv('ai_resume_training_data_450_final.csv')
        print("Training data loaded successfully!")
    except FileNotFoundError:
        print("Error: 'ai_resume_training_data_450_diverse.csv' not found. Please ensure it's in the same directory.")
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()

    # Extract target score from 'detailed_ai_suggestion' column
    def extract_score_from_suggestion(suggestion_text):
        match = re.search(r'Score:\s*(\d+)%', suggestion_text)
        if match:
            return int(match.group(1))
        return np.nan

    df['target_score'] = df['detailed_ai_suggestion'].apply(extract_score_from_suggestion)
    df.dropna(subset=['target_score'], inplace=True)
    
    if df.empty:
        print("No valid target scores found in the dataset after extraction. Exiting.")
        exit()
    print(f"Extracted {len(df)} valid target scores.")


    # Pre-process and extract features
    print("Extracting features from training data...")
    X = [] # Features for the ML model
    y = [] # Target variable (Score)

    for index, row in df.iterrows():
        jd_text = str(row['jd_text']) if not pd.isna(row['jd_text']) else ""
        resume_text = str(row['resume_text']) if not pd.isna(row['resume_text']) else ""
        score = float(row['target_score'])

        if not jd_text.strip() or not resume_text.strip():
            continue

        cleaned_jd = clean_text(jd_text)
        cleaned_resume = clean_text(resume_text)

        jd_embedding = np.array(sentence_model.encode(cleaned_jd))
        resume_embedding = np.array(sentence_model.encode(cleaned_resume))

        experience = extract_years_of_experience(resume_text)

        jd_keywords = set(get_top_keywords(cleaned_jd))
        resume_keywords = set(get_top_keywords(cleaned_resume))
        keyword_overlap_count = len(jd_keywords.intersection(resume_keywords))

        # NEW: Calculate Section Completeness and Length Score for ML features
        resume_sections = extract_sections(cleaned_resume)
        section_completeness_score = (sum(1 for sec in REQUIRED_SECTIONS if sec in resume_sections and resume_sections[sec]) / len(REQUIRED_SECTIONS) if REQUIRED_SECTIONS else 0.0) * 100
        length_score = calculate_length_score(cleaned_resume)

        # Combine all features into a single array
        # This will now result in 384 (JD embed) + 384 (Resume embed) + 1 (exp) + 1 (keyword overlap) + 1 (section completeness) + 1 (length) = 772 features
        features = np.concatenate([
            jd_embedding.astype(float),
            resume_embedding.astype(float),
            np.array([float(experience)]),
            np.array([float(keyword_overlap_count)]),
            np.array([float(section_completeness_score)]), # New feature
            np.array([float(length_score)]) # New feature
        ])
        X.append(features)
        y.append(score)

    X = np.array(X)
    y = np.array(y)

    if X.shape[0] == 0:
        print("No valid data points to train the model. Exiting.")
        exit()

    print(f"Shape of features (X): {X.shape}")
    print(f"Shape of target (y): {y.shape}")
    print(f"Number of features generated: {X.shape[1]}") # This should now print 772

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Defining the parameter grid for GridSearchCV...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')

    print("Starting GridSearchCV for hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    print("RandomForestRegressor trained with best hyperparameters.")
    print(f"Best parameters found: {grid_search.best_params_}")

    # Evaluate the best model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2) Score: {r2:.2f}")
    print(f"Features the model was trained on: {model.n_features_in_}") # This will confirm 772

    # Save the trained model
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Trained ML model saved to {MODEL_SAVE_PATH}")