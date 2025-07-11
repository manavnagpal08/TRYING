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
import collections
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

import nltk # Keep this import
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- REMOVE all the nltk.download() blocks from here ---
# st.info("Ensuring NLTK data is available...")
# try:
#     nltk.data.find('corpora/stopwords')
# except:
#     st.info("Downloading NLTK stopwords...")
#     nltk.download('stopwords')
# ... and so on for punkt, wordnet, averaged_perceptron_tagger

# --- Start directly with loading models ---
# Load Embedding + ML Model
@st.cache_resource
def load_ml_model():
    """Loads the SentenceTransformer model for embeddings and a pre-trained ML screening model."""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ml_model = joblib.load("ml_screening_model.pkl") # Ensure this file exists in the same directory
        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory and network is available for SentenceTransformer.")
        return None, None

# ... rest of your code ...

# --- Load T5 Model ---
@st.cache_resource
def load_t5_model():
    """Loads a pre-trained T5 model for resume summarization from Hugging Face Hub."""
    t5_tokenizer = None
    t5_model = None
    T5_REPO_ID = "mnagpal/fine-tuned-t5-resume-screener" # Specific T5 model for resume summarization
    try:
        t5_tokenizer = AutoTokenizer.from_pretrained(T5_REPO_ID)
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_REPO_ID)
        st.success("T5 Model loaded successfully from Hugging Face Hub!")
    except Exception as e:
        st.error(f"Error loading T5 model from Hugging Face Hub: {e}. Check network connection and model availability.")
    return t5_tokenizer, t5_model

# Load all models at startup
model, ml_model = load_ml_model()
t5_tokenizer, t5_model = load_t5_model()

# Initialize Lemmatizer for NLTK
lemmatizer = WordNetLemmatizer()

# --- Helper function to map NLTK POS tags to WordNetLemmatizer tags ---
def get_wordnet_pos(treebank_tag):
    """Maps NLTK's Part-of-Speech tags to WordNet's POS tags for lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # Default to noun if no clear tag


# --- Stop Words List (Using NLTK + Custom) ---
NLTK_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
# Comprehensive list of words to be filtered out because they are typically not skills
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
    "six", "sigma", "black", "belt", "green", "yellow", "qms", "9001", "27001",
]) # Make sure this list is truly comprehensive as per our prior discussions

ALL_STOP_WORDS = NLTK_STOP_WORDS.union(CUSTOM_STOP_WORDS)

# --- MASTER SKILLS DICTIONARY ---
# This is your comprehensive list of all potential skills.
all_skills_master = {
    # Product & Project Management
    "Product Strategy", "Roadmap Development", "Agile Methodologies", "Scrum", "Kanban", "Jira", "Trello",
    "Feature Prioritization", "OKRs", "KPIs", "Stakeholder Management", "A/B Testing", "User Stories", "Epics",
    "Product Lifecycle", "Sprint Planning", "Project Charter", "Gantt Charts", "MVP", "Backlog Grooming",

    # Software Development & Engineering
    "Python", "Java", "JavaScript", "C++", "C#", "HTML", "CSS", "React", "Angular", "Vue",
    "Git", "GitHub", "GitLab", "REST APIs", "GraphQL", "DevOps", "CI/CD", "Docker", "Kubernetes",
    "Microservices Architecture", "Object-Oriented Programming", "System Design", "Unit Testing", "Integration Testing",
    "Software Architecture",

    # Data Science & AI/ML
    "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "Scikit-learn", "TensorFlow",
    "PyTorch", "Data Cleaning", "Feature Engineering", "Regression", "Classification", "Clustering", "Neural Networks",
    "Time Series Analysis", "Data Visualization", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Model Evaluation Metrics",
    "Prompt Engineering", "LLMs", "GPT-4", "Claude", "Gemini",

    # Data Analytics & BI
    "SQL", "MySQL", "PostgreSQL", "Power BI", "Tableau", "Looker", "Google Data Studio", "Excel", "ETL", "ELT",
    "Data Warehousing", "Snowflake", "Redshift", "Data Modeling", "Business Analysis", "Statistical Analysis",
    "Google Analytics", "BigQuery",

    # Cloud & Infrastructure
    "AWS", "EC2", "S3", "Lambda", "RDS", "CloudWatch", "Azure", "GCP", "Terraform", "Linux", "Shell Scripting",
    "Monitoring", "Datadog", "Prometheus", "Grafana", "Load Balancing", "Serverless Architecture",

    # UI/UX & Design
    "Figma", "Adobe XD", "Sketch", "Illustrator", "Design Thinking", "Wireframing", "UX Research",
    "Prototyping", "Accessibility", "Responsive Design", "Material UI", "Bootstrap",

    # Marketing & Sales
    "Digital Marketing", "SEO", "SEM", "Google Ads", "Facebook Ads", "Email Marketing", "Growth Hacking",
    "Marketing Analytics", "Content Strategy", "HubSpot", "Salesforce CRM", "Lead Generation", "Campaign Management",
    "Conversion Optimization", "CRM", "Sales Strategy", "Negotiation", "Cold Calling", "Brand Management",

    # Finance & Accounting
    "Financial Analysis", "Budgeting", "Forecasting", "Bookkeeping", "Auditing", "Taxation", "Accounts Payable",
    "Accounts Receivable", "QuickBooks", "GAAP", "Financial Reporting", "Cost Accounting", "ERP",

    # Business & Management
    "Strategic Planning", "Operations Management", "Process Improvement", "Risk Management",
    "Change Management", "Business Development", "Vendor Management",

    # Human Resources (HR)
    "Recruitment", "Onboarding", "Employee Relations", "Payroll", "Compensation and Benefits", "HRIS",
    "HR Policies", "Performance Management", "Talent Acquisition", "Workforce Planning", "Labor Law Compliance",
    "Training & Development",

    # Customer Service & Support
    "Customer Relationship Management", "Call Handling", "Issue Resolution", "Zendesk", "Live Chat Support",
    "Ticketing Systems", "Upselling", "Escalation Handling",

    # Admin & Operations
    "Office Management", "Calendar Management", "Travel Coordination", "Procurement", "Inventory Management",
    "Document Preparation", "Data Entry", "MS Office", "Email Communication", "Scheduling", "Record Keeping",

    # Education & Training
    "Curriculum Development", "Lesson Planning", "Classroom Management", "Assessment Design", "Instructional Design",
    "LMS", "Training Delivery", "Student Engagement", "E-learning",

    # Healthcare & Medical
    "Patient Care", "Medical Terminology", "EMR Systems", "Vital Signs Monitoring", "Clinical Documentation",
    "Medical Coding", "Healthcare Compliance", "HIPAA", "Medical Billing", "Diagnosis Support",

    # Logistics & Supply Chain
    "Warehouse Operations", "Supply Chain Planning", "Fleet Management", "Demand Forecasting",
    "Shipping & Receiving", "Order Fulfillment", "SAP Logistics",

    # Legal & Compliance
    "Legal Research", "Contract Drafting", "Case Management", "Document Review", "Regulatory Compliance",
    "Litigation Support", "Intellectual Property", "Risk Mitigation", "Corporate Law", "Legal Writing",

    # Cybersecurity & IT Support
    "Network Security", "Vulnerability Scanning", "Penetration Testing", "IAM", "SIEM", "Splunk", "QRadar",
    "Encryption", "Firewalls", "Incident Response", "Active Directory", "Technical Troubleshooting", "ITIL",

    # General Tech & Productivity
    "Microsoft Office", "Google Workspace", "Zoom", "Slack", "WordPress", "Basic HTML", "Email Automation"
}

# Convert all_skills_master to a set for faster lookup and uniform case
ALL_SKILLS_MASTER_SET = {skill.lower() for skill in all_skills_master}


# --- JOB-SPECIFIC REQUIRED SKILLS DICTIONARIES ---
# Define required skills for various job roles using the ALL_SKILLS_MASTER_SET
# These are the skills we EXPECT to see for a good match.

JOB_SKILLS = {
    "Software Engineer": {
        "Python", "Java", "JavaScript", "React", "Angular", "Vue", "Git", "GitHub", "REST APIs", "DevOps",
        "Docker", "Kubernetes", "Microservices Architecture", "Object-Oriented Programming", "System Design",
        "Unit Testing", "Software Architecture", "CI/CD", "C++", "C#", "HTML", "CSS", "SQL", "PostgreSQL", "MySQL"
    },
    "Data Scientist": {
        "Python", "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision",
        "Scikit-learn", "TensorFlow", "PyTorch", "Data Cleaning", "Feature Engineering", "Regression",
        "Classification", "Clustering", "Neural Networks", "Time Series Analysis", "Data Visualization",
        "Pandas", "NumPy", "Matplotlib", "Seaborn", "SQL", "BigQuery", "LLMs", "Prompt Engineering"
    },
    "Product Manager": {
        "Product Strategy", "Roadmap Development", "Agile Methodologies", "Scrum", "Kanban", "Jira",
        "Feature Prioritization", "OKRs", "KPIs", "Stakeholder Management", "A/B Testing", "User Stories",
        "Product Lifecycle", "MVP", "Backlog Grooming", "UX Research", "Business Analysis", "Gantt Charts"
    },
    "Data Analyst": {
        "SQL", "Excel", "Power BI", "Tableau", "Looker", "Google Data Studio", "Data Cleaning", "Data Visualization",
        "Statistical Analysis", "Google Analytics", "Business Analysis", "ETL", "Data Modeling"
    },
    "DevOps Engineer": {
        "DevOps", "CI/CD", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Terraform", "Linux",
        "Shell Scripting", "Monitoring", "Datadog", "Prometheus", "Grafana", "Git", "GitHub", "GitLab",
        "Serverless Architecture", "Ansible", "Jenkins"
    },
    "Marketing Specialist": {
        "Digital Marketing", "SEO", "SEM", "Google Ads", "Facebook Ads", "Email Marketing", "Growth Hacking",
        "Marketing Analytics", "Content Strategy", "HubSpot", "Salesforce CRM", "Lead Generation", "Campaign Management",
        "Conversion Optimization", "CRM", "Sales Strategy", "Negotiation", "Cold Calling", "Brand Management"
    },
    "HR Manager": {
        "Recruitment", "Onboarding", "Employee Relations", "Payroll", "Compensation and Benefits", "HRIS",
        "HR Policies", "Performance Management", "Talent Acquisition", "Workforce Planning", "Labor Law Compliance",
        "Training & Development", "Strategic Planning", "Change Management"
    },
    "UI/UX Designer": {
        "Figma", "Adobe XD", "Sketch", "Illustrator", "Design Thinking", "Wireframing", "UX Research",
        "Prototyping", "Accessibility", "Responsive Design", "Material UI", "Bootstrap", "User Stories"
    },
    "Cybersecurity Analyst": {
        "Network Security", "Vulnerability Scanning", "Penetration Testing", "IAM", "SIEM", "Splunk",
        "QRadar", "Encryption", "Firewalls", "Incident Response", "Active Directory", "ITIL", "Healthcare Compliance", "HIPAA"
    },
    "Financial Analyst": {
        "Financial Analysis", "Budgeting", "Forecasting", "Excel", "Financial Reporting", "Cost Accounting",
        "ERP", "GAAP", "Auditing", "Statistical Analysis"
    },
    "Customer Service Rep": {
        "Customer Relationship Management", "Call Handling", "Issue Resolution", "Zendesk",
        "Live Chat Support", "Ticketing Systems", "Email Communication", "Scheduling"
    }
}

# Convert all skills in JOB_SKILLS to lowercase for robust matching
JOB_SKILLS_LOWER = {
    job_role: {skill.lower() for skill in skills_set}
    for job_role, skills_set in JOB_SKILLS.items()
}


# --- Keyword Extraction Function (Enhanced) ---
def extract_skills_from_text(text):
    """
    Extracts relevant skills from a given text using multi-word matching,
    lemmatization, POS tagging, and stop word filtering.
    """
    text = text.lower()
    
    # Use NLTK for tokenization and POS tagging
    tokens = nltk.word_tokenize(text) # This is the line that caused LookupError
    tagged_tokens = nltk.pos_tag(tokens)

    extracted_skills = set()
    
    # First, try to match multi-word skills (bi-grams, tri-grams from all_skills_master)
    # Iterate through skills_master sorted by length (descending) to prioritize longer matches
    sorted_master_skills = sorted(list(ALL_SKILLS_MASTER_SET), key=len, reverse=True)

    # Use a copy of the text to mark out matched skills, preventing partial matches
    processed_text = text 
    for skill in sorted_master_skills:
        # Create a regex pattern for the multi-word skill to find it in the text
        # Use word boundaries (\b) to match whole words and re.escape for special characters
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, processed_text):
            extracted_skills.add(skill)
            # Replace matched skill with placeholders to avoid re-matching its components later
            # Replace with spaces to maintain word boundaries for subsequent single-word tokenization
            processed_text = re.sub(pattern, ' ' * len(skill), processed_text) # Replace with spaces of same length

    # Now process the remaining text for single words, after multi-words are extracted
    # Re-tokenize and tag the processed_text
    tokens_remaining = nltk.word_tokenize(processed_text)
    tagged_tokens_remaining = nltk.pos_tag(tokens_remaining)
    
    for word, tag in tagged_tokens_remaining:
        # Lemmatize the word
        pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos)

        # Basic filtering: numeric, short, stop words
        if not lemma.isalpha() or len(lemma) < 2 or lemma in ALL_STOP_WORDS:
            continue

        # Check if the lemmatized word is in our master skill list
        if lemma in ALL_SKILLS_MASTER_SET:
            extracted_skills.add(lemma)
            
    return list(extracted_skills)


# --- Core PDF Text Extraction ---
def extract_text_from_pdf(pdf_file):
    """Extracts all text content from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

# --- Word Cloud Generation ---
def generate_wordcloud(skills_list):
    """Generates and displays a word cloud from a list of skills."""
    if skills_list:
        skills_text = " ".join(skills_list)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(skills_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.info("No skills to display in word cloud.")

# --- Skills Bar Chart Generation ---
def generate_skills_chart(skills_list):
    """Generates and displays a bar chart of skill frequencies."""
    if skills_list:
        skill_counts = collections.Counter(skills_list)
        df_skills = pd.DataFrame(skill_counts.items(), columns=['Skill', 'Count']).sort_values(by='Count', ascending=False)
        st.bar_chart(df_skills.set_index('Skill'))
    else:
        st.info("No skills to display in chart.")

# --- Resume Summarization using T5 ---
def summarize_resume_t5(resume_text, t5_tokenizer, t5_model):
    """Summarizes a resume using the loaded T5 model."""
    if t5_tokenizer and t5_model:
        prompt = "summarize the following resume: " + resume_text
        inputs = t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        # Generate summary with specified parameters
        outputs = t5_model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            length_penalty=2.0, # Encourages longer summaries
            num_beams=4,        # Wider search for better summary
            early_stopping=True
        )
        summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    return "T5 model not loaded. Cannot summarize."

# --- Embedding Generation for ML Model ---
def get_job_description_embedding(job_description, model):
    """Generates a numerical embedding for a job description using SentenceTransformer."""
    if model:
        return model.encode(job_description)
    return None

# --- ML Match Prediction ---
def predict_match_ml(resume_embedding, job_description_embedding, ml_model):
    """
    Predicts the match probability between a resume and job description
    using a pre-trained machine learning model.
    """
    if ml_model and resume_embedding is not None and job_description_embedding is not None:
        # Concatenate embeddings to form the feature vector for the ML model
        combined_features = np.concatenate((resume_embedding, job_description_embedding)).reshape(1, -1)
        # Get probability of being a match (class 1)
        prediction_proba = ml_model.predict_proba(combined_features)[0][1] 
        return prediction_proba
    return 0.0 # Return 0 if models/embeddings are not available

# --- Streamlit UI Configuration ---
st.set_page_config(layout="wide", page_title="ScreenerPro – AI Hiring Assistant 🧠")
st.title("🧠 ScreenerPro – AI Hiring Assistant")

# --- UI Elements - NO SIDEBAR ---
st.header("1. Upload Job Description (PDF/TXT)")
job_description_file = st.file_uploader("Upload Job Description", type=["pdf", "txt"], key="jd_upload")

st.header("2. Upload Resumes (PDF)")
uploaded_resumes = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True, key="resume_upload")

st.header("3. Select Target Job Role for Skill Matching")
selected_job_role = st.selectbox(
    "Select a Job Role for Skill Matching:",
    list(JOB_SKILLS.keys())
)


job_description_text = ""
jd_embedding = None
jd_skills = []

if job_description_file:
    # Extract text from JD file
    if job_description_file.type == "application/pdf":
        job_description_text = extract_text_from_pdf(job_description_file)
    else: # text file
        job_description_text = job_description_file.read().decode("utf-8")
    
    st.subheader("Job Description Content (Excerpt)")
    st.text(job_description_text[:500] + "...") # Show first 500 chars

    # Process JD for skills and embeddings if models loaded
    if model:
        jd_skills = extract_skills_from_text(job_description_text)
        jd_embedding = get_job_description_embedding(job_description_text, model)
else:
    st.warning("Please upload a Job Description to proceed.")


st.markdown("---") # Visual separator in the main content area

# --- Main Content Area for Resume Analysis ---
if uploaded_resumes and job_description_file:
    st.subheader("Uploaded Resumes Analysis")
    results = [] # To store results for all resumes

    if model and ml_model and t5_tokenizer and t5_model:
        for i, resume_file in enumerate(uploaded_resumes):
            st.markdown(f"#### Analyzing: {resume_file.name}")
            resume_text = extract_text_from_pdf(resume_file)
            
            # --- Extract Skills from Resume ---
            resume_skills = extract_skills_from_text(resume_text)
            st.write(f"**Extracted Skills:** {', '. join(resume_skills) if resume_skills else 'No specific skills found.'}")

            # --- Calculate Skill Match based on Selected Job Role ---
            target_skills_for_role = JOB_SKILLS_LOWER.get(selected_job_role, set())
            
            skill_match_percentage = 0
            matched_skills = set()
            missing_skills = set()

            if target_skills_for_role:
                resume_skills_lower = {skill.lower() for skill in resume_skills} # Convert extracted skills to lowercase
                
                matched_skills = resume_skills_lower.intersection(target_skills_for_role)
                missing_skills = target_skills_for_role.difference(resume_skills_lower)
                
                if target_skills_for_role: # Avoid division by zero
                    skill_match_percentage = (len(matched_skills) / len(target_skills_for_role)) * 100
                
                st.write(f"**Skill Match for '{selected_job_role}':** {skill_match_percentage:.2f}%")
                if matched_skills:
                    st.write(f"**Matched Skills:** {', '.join(sorted(list(matched_skills)))}")
                if missing_skills:
                    st.write(f"**Missing Skills for Role:** {', '.join(sorted(list(missing_skills)))}")
            else:
                st.info(f"No specific skill requirements defined for '{selected_job_role}'.")
            
            # --- Generate Embedding for Resume and Predict ML Match ---
            resume_embedding = model.encode(resume_text)
            match_proba = predict_match_ml(resume_embedding, jd_embedding, ml_model)
            st.write(f"**AI Similarity Score (ML Match Probability):** {match_proba:.2%}")

            # --- Summarize Resume using T5 Model ---
            summary = summarize_resume_t5(resume_text, t5_tokenizer, t5_model)
            st.write(f"**Resume Summary:** {summary}")

            # --- Display Visualizations (Word Cloud & Bar Chart) ---
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Skills Word Cloud")
                generate_wordcloud(resume_skills)
            with col2:
                st.subheader("Skills Bar Chart")
                generate_skills_chart(resume_skills)

            # Store results for the final DataFrame
            results.append({
                "Resume Name": resume_file.name,
                "AI Similarity Score": f"{match_proba:.2%}",
                "Skill Match (%)": f"{skill_match_percentage:.2f}%",
                "Extracted Skills": ", ".join(resume_skills),
                "Matched Skills": ", ".join(matched_skills) if matched_skills else "N/A",
                "Missing Skills": ", ".join(missing_skills) if missing_skills else "None",
                "Resume Summary": summary,
                "Resume Text": resume_text # Keep full text for potential further use
            })
            st.markdown("---") # Separator for each resume in the UI

        st.subheader("Overall Resume Screening Results")
        # Display results in a DataFrame
        results_df = pd.DataFrame(results)
        st.dataframe(results_df[[
            "Resume Name",
            "AI Similarity Score",
            "Skill Match (%)",
            "Extracted Skills",
            "Matched Skills",
            "Missing Skills",
            "Resume Summary"
        ]])

        # Download results as CSV
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="resume_screening_results.csv",
            mime="text/csv",
        )

    # Error messages if models fail to load
    elif not (model and ml_model):
        st.error("ML model or SentenceTransformer failed to load. Please check the `ml_screening_model.pkl` file and network connection for Hugging Face models.")
    elif not (t5_tokenizer and t5_model):
        st.error("T5 model failed to load. Please check your network connection for Hugging Face models.")
elif not job_description_file:
    st.info("Please upload a Job Description to start screening resumes.")
elif not uploaded_resumes:
    st.info("Please upload at least one resume to analyze.")
