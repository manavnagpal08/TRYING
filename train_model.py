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

# --- Stop Words List (Leave these empty for you to paste your custom lists) ---
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

# Combine all stop words
ALL_STOP_WORDS = NLTK_STOP_WORDS.union(CUSTOM_STOP_WORDS)

# --- Text Preprocessing Functions ---
def clean_text(text):
    """
    Cleans the input text by converting to lowercase, removing punctuation,
    and removing extra whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def get_top_keywords(text, num_keywords=10):
    """
    Extracts the most frequent non-stopwords from the text.
    Considers both single words and common two-word phrases (bigrams)
    for better keyword representation.
    """
    words = clean_text(text).split()
    filtered_words = [word for word in words if word not in ALL_STOP_WORDS and len(word) > 2]

    # Count single words
    word_counts = collections.Counter(filtered_words)

    # Count bigrams (two-word phrases)
    bigrams = list(nltk.ngrams(filtered_words, 2))
    bigram_counts = collections.Counter(bigrams)

    # Combine counts, prioritizing bigrams if they are highly frequent
    combined_counts = collections.Counter()
    for word, count in word_counts.items():
        combined_counts[word] = count
    for bigram, count in bigram_counts.items():
        combined_counts[" ".join(bigram)] = count * 1.5 # Give bigrams a slight boost

    # Return top N keywords
    return [keyword for keyword, _ in combined_counts.most_common(num_keywords)]

def extract_experience(text):
    """
    Extracts total years of experience from a resume text.
    Looks for patterns like 'X years', 'X+ years', 'X-Y years'.
    If multiple found, takes the maximum. If ranges, averages them.
    Also calculates experience based on start/end dates if available.
    """
    years = []

    # Regex for "X years", "X+ years"
    matches_single = re.findall(r'(\d+)\s*year(?:s)?(?:\s*\+)?(?: of experience)?', text, re.IGNORECASE)
    years.extend([int(m) for m in matches_single])

    # Regex for "X-Y years"
    matches_range = re.findall(r'(\d+)\s*-\s*(\d+)\s*year(?:s)?', text, re.IGNORECASE)
    for start, end in matches_range:
        years.append((int(start) + int(end)) / 2)

    # Regex for date ranges (e.g., "Jan 2020 - Dec 2023", "2020 - Present")
    date_patterns = [
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{4})\s*-\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{4})',
        r'(\d{4})\s*-\s*(\d{4})',
        r'(\d{4})\s*-\s*(?:Present|Current|Now)',
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{4})\s*-\s*(?:Present|Current|Now)'
    ]

    current_year = datetime.now().year
    for pattern in date_patterns:
        date_matches = re.findall(pattern, text, re.IGNORECASE)
        for match in date_matches:
            try:
                start_year = int(match[0])
                end_year = current_year if 'present' in match[-1].lower() or 'current' in match[-1].lower() or 'now' in match[-1].lower() else int(match[-1])
                if end_year >= start_year: # Ensure valid range
                    years.append(end_year - start_year)
            except ValueError:
                continue # Skip if year conversion fails

    if years:
        # Return the maximum experience found from all methods
        return max(years)
    return 0.0

# --- Feature Creation Function ---
def create_features(jd_text, resume_text, jd_model, resume_model):
    """
    Creates feature vector for a given JD and resume pair.
    Features include:
    - JD and Resume sentence embeddings (384 dimensions each for all-MiniLM-L6-v2)
    - Experience from resume
    - Keyword overlap count
    """
    # Generate embeddings
    jd_embedding = jd_model.encode(clean_text(jd_text))
    resume_embedding = resume_model.encode(clean_text(resume_text))

    # Extract experience
    experience = extract_experience(resume_text)

    # Get keywords and calculate overlap
    jd_keywords = set(get_top_keywords(jd_text, num_keywords=30)) # Increased keywords for better coverage
    resume_keywords = set(get_top_keywords(resume_text, num_keywords=50)) # More keywords from resume

    keyword_overlap = len(jd_keywords.intersection(resume_keywords))

    # Combine all features into a single numpy array
    features = np.concatenate([jd_embedding, resume_embedding, [experience], [keyword_overlap]])
    return features

# --- Main Training Script ---
if __name__ == "__main__":
    print("Starting model training process...")

    # Load pre-trained SentenceTransformer models
    # Using 'all-MiniLM-L6-v2' for efficiency and good performance (384 dimensions per embedding)
    jd_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model loaded.")

    # --- Synthetic Data (Leave this empty for you to paste your data) ---
    synthetic_data = [        # --- Additional 100 Data Points for Expanded Diversity ---
       {
            "jd_text": "Data Scientist. Skills: Python, R, SQL, Machine Learning, Deep Learning, NLP, Tableau, Spark. Experience: 4+ years.",
            "resume_text": "Experienced Nurse Practitioner, 8 years. Provided patient care and managed medical records. Strong in diagnostics. No programming or data science background.",
            "relevance_score": 5
        },
        {
            "jd_text": "Data Scientist. Skills: Python, R, SQL, Machine Learning, Deep Learning, NLP, Tableau, Spark. Experience: 4+ years.",
            "resume_text": "Senior Accountant, 10 years. Managed general ledger and prepared financial statements. Proficient in SAP. No statistical modeling or coding.",
            "relevance_score": 5
        },
        {
            "jd_text": "Data Scientist. Skills: Python, R, SQL, Machine Learning, Deep Learning, NLP, Tableau, Spark. Experience: 4+ years.",
            "resume_text": "Construction Project Manager, 12 years. Oversaw large construction projects. Managed budgets and schedules. No analytical or technical skills.",
            "relevance_score": 5
        },

        # Example: UX/UI Designer (low relevance)
        {
            "jd_text": "UX/UI Designer. Skills: User Research, Wireframing, Prototyping, Figma, Adobe XD, Usability Testing. Experience: 2+ years.",
            "resume_text": "Experienced Backend Developer, 6 years. Built APIs with Java and Spring Boot. Managed databases. No design tools or user research experience.",
            "relevance_score": 10
        },
        {
            "jd_text": "UX/UI Designer. Skills: User Research, Wireframing, Prototyping, Figma, Adobe XD, Usability Testing. Experience: 2+ years.",
            "resume_text": "Financial Analyst, 5 years. Built financial models in Excel. Strong in valuation. No design or creative skills.",
            "relevance_score": 5
        },
        {
            "jd_text": "UX/UI Designer. Skills: User Research, Wireframing, Prototyping, Figma, Adobe XD, Usability Testing. Experience: 2+ years.",
            "resume_text": "Operations Manager, 9 years. Optimized logistics processes. Managed inventory. No design or user-centric focus.",
            "relevance_score": 5
        },

        # Example: Cloud Architect (low relevance)
        {
            "jd_text": "Cloud Architect. Skills: AWS, Azure, GCP, Cloud Architecture, Terraform, Kubernetes, Network Security. Experience: 7+ years.",
            "resume_text": "Senior Marketing Director, 15 years. Developed global marketing strategies. Managed large teams. No IT or cloud infrastructure experience.",
            "relevance_score": 10
        },
        {
            "jd_text": "Cloud Architect. Skills: AWS, Azure, GCP, Cloud Architecture, Terraform, Kubernetes, Network Security. Experience: 7+ years.",
            "resume_text": "Chief Financial Officer, 20 years. Oversaw financial operations. Managed investor relations. No technical or cloud skills.",
            "relevance_score": 5
        },
        {
            "jd_text": "Cloud Architect. Skills: AWS, Azure, GCP, Cloud Architecture, Terraform, Kubernetes, Network Security. Experience: 7+ years.",
            "resume_text": "Human Resources Director, 18 years. Led HR strategy and talent management. No technical background.",
            "relevance_score": 5
        },

        # Example: Cybersecurity Engineer (low relevance)
        {
            "jd_text": "Cybersecurity Engineer. Skills: Incident Response, Vulnerability Management, SIEM, Firewalls, Python Scripting. Experience: 4+ years.",
            "resume_text": "Senior Product Manager, 8 years. Defined product roadmaps and user stories. Managed product lifecycle. No cybersecurity or IT operations.",
            "relevance_score": 10
        },
        {
            "jd_text": "Cybersecurity Engineer. Skills: Incident Response, Vulnerability Management, SIEM, Firewalls, Python Scripting. Experience: 4+ years.",
            "resume_text": "Content Strategist, 6 years. Developed content strategies and managed editorial calendars. No technical or security skills.",
            "relevance_score": 5
        },
        {
            "jd_text": "Cybersecurity Engineer. Skills: Incident Response, Vulnerability Management, SIEM, Firewalls, Python Scripting. Experience: 4+ years.",
            "resume_text": "E-commerce Manager, 7 years. Managed online store operations and digital marketing. No IT security.",
            "relevance_score": 5
        },

        # Example: DevOps Specialist (low relevance)
        {
            "jd_text": "DevOps Specialist. Skills: CI/CD, Docker, Kubernetes, Jenkins, Terraform, Ansible, AWS, Python. Experience: 4+ years.",
            "resume_text": "Senior Data Scientist, 7 years. Developed machine learning models. Strong in Python. No infrastructure or CI/CD experience.",
            "relevance_score": 10
        },
        {
            "jd_text": "DevOps Specialist. Skills: CI/CD, Docker, Kubernetes, Jenkins, Terraform, Ansible, AWS, Python. Experience: 4+ years.",
            "resume_text": "Financial Analyst, 6 years. Built financial models. Proficient in Excel. No technical operations.",
            "relevance_score": 5
        },
        {
            "jd_text": "DevOps Specialist. Skills: CI/CD, Docker, Kubernetes, Jenkins, Terraform, Ansible, AWS, Python. Experience: 4+ years.",
            "resume_text": "Marketing Director, 10 years. Led marketing teams. No technical or operations background.",
            "relevance_score": 5
        },

        # Example: Mobile App Developer (iOS) (low relevance)
        {
            "jd_text": "Mobile App Developer (iOS). Skills: Swift, SwiftUI, iOS SDK, Xcode, RESTful APIs, UI/UX. Experience: 3+ years.",
            "resume_text": "Experienced Android Developer, 5 years. Built native Android apps with Kotlin. Strong in Android SDK. No iOS experience.",
            "relevance_score": 15 # Some transferable skills, but not direct match
        },
        {
            "jd_text": "Mobile App Developer (iOS). Skills: Swift, SwiftUI, iOS SDK, Xcode, RESTful APIs, UI/UX. Experience: 3+ years.",
            "resume_text": "Backend Java Developer, 8 years. Built enterprise applications. No mobile development.",
            "relevance_score": 5
        },
        {
            "jd_text": "Mobile App Developer (iOS). Skills: Swift, SwiftUI, iOS SDK, Xcode, RESTful APIs, UI/UX. Experience: 3+ years.",
            "resume_text": "IT Support Specialist, 6 years. Provided technical support. No programming or mobile dev.",
            "relevance_score": 5
        },

        # Example: Business Development Manager (low relevance)
        {
            "jd_text": "Business Development Manager. Skills: Sales Strategy, Lead Generation, Negotiation, CRM (Salesforce), Client Relationships. Experience: 5+ years.",
            "resume_text": "Senior Software Engineer, 10 years. Built scalable software systems. No sales or business development.",
            "relevance_score": 10
        },
        {
            "jd_text": "Business Development Manager. Skills: Sales Strategy, Lead Generation, Negotiation, CRM (Salesforce), Client Relationships. Experience: 5+ years.",
            "resume_text": "Data Analyst, 7 years. Performed data analysis. No sales or client-facing role.",
            "relevance_score": 5
        },
        {
            "jd_text": "Business Development Manager. Skills: Sales Strategy, Lead Generation, Negotiation, CRM (Salesforce), Client Relationships. Experience: 5+ years.",
            "resume_text": "HR Manager, 8 years. Managed employee relations. No sales or business development.",
            "relevance_score": 5
        },

        # Example: Project Coordinator (low relevance)
        {
            "jd_text": "Project Coordinator. Skills: Project Planning, Scheduling, Communication, Microsoft Office, Jira. Experience: 2+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built web applications. No project coordination.",
            "relevance_score": 10
        },
        {
            "jd_text": "Project Coordinator. Skills: Project Planning, Scheduling, Communication, Microsoft Office, Jira. Experience: 2+ years.",
            "resume_text": "Data Scientist, 7 years. Developed ML models. No project coordination.",
            "relevance_score": 5
        },
        {
            "jd_text": "Project Coordinator. Skills: Project Planning, Scheduling, Communication, Microsoft Office, Jira. Experience: 2+ years.",
            "resume_text": "Financial Auditor, 10 years. Conducted financial audits. No project coordination.",
            "relevance_score": 5
        },

        # Example: Technical Writer (low relevance)
        {
            "jd_text": "Technical Writer. Skills: Technical Documentation, API Documentation, Content Management Systems, Markdown, XML. Experience: 3+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built enterprise applications. No technical writing.",
            "relevance_score": 10
        },
        {
            "jd_text": "Technical Writer. Skills: Technical Documentation, API Documentation, Content Management Systems, Markdown, XML. Experience: 3+ years.",
            "resume_text": "Data Scientist, 7 years. Developed ML models. No technical writing.",
            "relevance_score": 5
        },
        {
            "jd_text": "Technical Writer. Skills: Technical Documentation, API Documentation, Content Management Systems, Markdown, XML. Experience: 3+ years.",
            "resume_text": "Financial Analyst, 10 years. Built financial models. No technical writing.",
            "relevance_score": 5
        },

        # Example: Sales Representative (low relevance)
        {
            "jd_text": "Sales Representative. Skills: Sales, Lead Generation, Cold Calling, Negotiation, CRM. Experience: 1+ years.",
            "resume_text": "Software Engineer, 5 years. Built web applications. No sales experience.",
            "relevance_score": 10
        },
        {
            "jd_text": "Sales Representative. Skills: Sales, Lead Generation, Cold Calling, Negotiation, CRM. Experience: 1+ years.",
            "resume_text": "Data Analyst, 4 years. Performed data analysis. No sales experience.",
            "relevance_score": 5
        },
        {
            "jd_text": "Sales Representative. Skills: Sales, Lead Generation, Cold Calling, Negotiation, CRM. Experience: 1+ years.",
            "resume_text": "HR Generalist, 6 years. Managed employee relations. No sales experience.",
            "relevance_score": 5
        },

        # Example: Customer Support Specialist (low relevance)
        {
            "jd_text": "Customer Support Specialist. Skills: Customer Service, Problem Solving, Troubleshooting, CRM, Communication. Experience: 2+ years.",
            "resume_text": "Software Engineer, 5 years. Built web applications. No customer service.",
            "relevance_score": 10
        },
        {
            "jd_text": "Customer Support Specialist. Skills: Customer Service, Problem Solving, Troubleshooting, CRM, Communication. Experience: 2+ years.",
            "resume_text": "Data Scientist, 4 years. Developed ML models. No customer service.",
            "relevance_score": 5
        },
        {
            "jd_text": "Customer Support Specialist. Skills: Customer Service, Problem Solving, Troubleshooting, CRM, Communication. Experience: 2+ years.",
            "resume_text": "Financial Analyst, 6 years. Built financial models. No customer service.",
            "relevance_score": 5
        },

        # Example: Operations Coordinator (low relevance)
        {
            "jd_text": "Operations Coordinator. Skills: Process Improvement, Logistics, Inventory Management, Data Entry, Excel. Experience: 2+ years.",
            "resume_text": "Software Engineer, 5 years. Built web applications. No operations.",
            "relevance_score": 10
        },
        {
            "jd_text": "Operations Coordinator. Skills: Process Improvement, Logistics, Inventory Management, Data Entry, Excel. Experience: 2+ years.",
            "resume_text": "Data Analyst, 4 years. Performed data analysis. No operations.",
            "relevance_score": 5
        },
        {
            "jd_text": "Operations Coordinator. Skills: Process Improvement, Logistics, Inventory Management, Data Entry, Excel. Experience: 2+ years.",
            "resume_text": "HR Generalist, 6 years. Managed employee relations. No operations.",
            "relevance_score": 5
        },
        # Example: Software Quality Assurance Engineer (low relevance)
        {
            "jd_text": "Software Quality Assurance Engineer. Skills: Test Automation, Selenium, JUnit/Pytest, CI/CD, Agile, Bug Tracking. Experience: 3+ years.",
            "resume_text": "Financial Analyst, 5 years. Built financial models. No QA or software testing.",
            "relevance_score": 10
        },
        {
            "jd_text": "Software Quality Assurance Engineer. Skills: Test Automation, Selenium, JUnit/Pytest, CI/CD, Agile, Bug Tracking. Experience: 3+ years.",
            "resume_text": "Marketing Specialist, 4 years. Managed digital campaigns. No QA or software testing.",
            "relevance_score": 5
        },
        {
            "jd_text": "Software Quality Assurance Engineer. Skills: Test Automation, Selenium, JUnit/Pytest, CI/CD, Agile, Bug Tracking. Experience: 3+ years.",
            "resume_text": "HR Generalist, 6 years. Managed employee relations. No QA or software testing.",
            "relevance_score": 5
        },# Example: Network Engineer (low relevance)
        {
            "jd_text": "Network Engineer. Skills: Cisco, Juniper, Routing, Switching, Firewalls, VPN, Network Monitoring. Experience: 4+ years.",
            "resume_text": "Marketing Specialist, 5 years. Managed digital campaigns. No IT or networking.",
            "relevance_score": 10
        },
        {
            "jd_text": "Network Engineer. Skills: Cisco, Juniper, Routing, Switching, Firewalls, VPN, Network Monitoring. Experience: 4+ years.",
            "resume_text": "Financial Analyst, 4 years. Built financial models. No IT or networking.",
            "relevance_score": 5
        },
        {
            "jd_text": "Network Engineer. Skills: Cisco, Juniper, Routing, Switching, Firewalls, VPN, Network Monitoring. Experience: 4+ years.",
            "resume_text": "HR Generalist, 6 years. Managed employee relations. No IT or networking.",
            "relevance_score": 5
        },

        # Example: Data Analyst (Entry Level) (low relevance)
        {
            "jd_text": "Data Analyst (Entry Level). Skills: Excel, SQL, Data Cleaning, Basic Statistics, Communication. Experience: 0-2 years.",
            "resume_text": "Retail Sales Associate, 3 years. Assisted customers and managed sales floor. No data analysis.",
            "relevance_score": 10
        },
        {
            "jd_text": "Data Analyst (Entry Level). Skills: Excel, SQL, Data Cleaning, Basic Statistics, Communication. Experience: 0-2 years.",
            "resume_text": "Customer Service Representative, 2 years. Handled customer inquiries. No data analysis.",
            "relevance_score": 5
        },
        {
            "jd_text": "Data Analyst (Entry Level). Skills: Excel, SQL, Data Cleaning, Basic Statistics, Communication. Experience: 0-2 years.",
            "resume_text": "Administrative Assistant, 1 year. Managed office tasks. No data analysis.",
            "relevance_score": 5
        },
        # Example: Marketing Coordinator (low relevance)
        {
            "jd_text": "Marketing Coordinator. Skills: Event Planning, Social Media, Email Marketing, Content Support, CRM. Experience: 1+ years.",
            "resume_text": "Software Engineer, 3 years. Built web applications. No marketing.",
            "relevance_score": 10
        },
        {
            "jd_text": "Marketing Coordinator. Skills: Event Planning, Social Media, Email Marketing, Content Support, CRM. Experience: 1+ years.",
            "resume_text": "Data Analyst, 2 years. Performed data analysis. No marketing.",
            "relevance_score": 5
        },
        {
            "jd_text": "Marketing Coordinator. Skills: Event Planning, Social Media, Email Marketing, Content Support, CRM. Experience: 1+ years.",
            "resume_text": "Financial Accountant, 3 years. Managed financial records. No marketing.",
            "relevance_score": 5
        },

        # Example: Business Analyst (IT Focus) (low relevance)
        {
            "jd_text": "Business Analyst (IT). Skills: Requirements Gathering, Process Mapping, SQL, Agile, JIRA, Stakeholder Management. Experience: 3+ years.",
            "resume_text": "Marketing Manager, 5 years. Led marketing campaigns. No IT or business analysis.",
            "relevance_score": 10
        },
        {
            "jd_text": "Business Analyst (IT). Skills: Requirements Gathering, Process Mapping, SQL, Agile, JIRA, Stakeholder Management. Experience: 3+ years.",
            "resume_text": "Financial Analyst, 4 years. Built financial models. No IT or business analysis.",
            "relevance_score": 5
        },
        {
            "jd_text": "Business Analyst (IT). Skills: Requirements Gathering, Process Mapping, SQL, Agile, JIRA, Stakeholder Management. Experience: 3+ years.",
            "resume_text": "HR Generalist, 6 years. Managed employee relations. No IT or business analysis.",
            "relevance_score": 5
        },

        # Example: Mechanical Engineer (Entry Level) (low relevance)
        {
            "jd_text": "Mechanical Engineer (Entry Level). Skills: CAD (SolidWorks), Thermodynamics, Materials Science, Prototyping. Experience: 0-2 years.",
            "resume_text": "Software Engineer, 1 year. Built web applications. No mechanical engineering.",
            "relevance_score": 10
        },
        {
            "jd_text": "Mechanical Engineer (Entry Level). Skills: CAD (SolidWorks), Thermodynamics, Materials Science, Prototyping. Experience: 0-2 years.",
            "resume_text": "Data Analyst, 1 year. Performed data analysis. No mechanical engineering.",
            "relevance_score": 5
        },
        {
            "jd_text": "Mechanical Engineer (Entry Level). Skills: CAD (SolidWorks), Thermodynamics, Materials Science, Prototyping. Experience: 0-2 years.",
            "resume_text": "HR Intern. Assisted with recruitment. No mechanical engineering.",
            "relevance_score": 5
        },# Example: Registered Nurse (General) (low relevance)
        {
            "jd_text": "Registered Nurse. Skills: Patient Care, Medication Administration, Electronic Health Records, Communication, Critical Thinking. Experience: 1+ years.",
            "resume_text": "Software Engineer, 3 years. Built web applications. No medical experience.",
            "relevance_score": 10
        },
        {
            "jd_text": "Registered Nurse. Skills: Patient Care, Medication Administration, Electronic Health Records, Communication, Critical Thinking. Experience: 1+ years.",
            "resume_text": "Data Analyst, 2 years. Performed data analysis. No medical experience.",
            "relevance_score": 5
        },
        {
            "jd_text": "Registered Nurse. Skills: Patient Care, Medication Administration, Electronic Health Records, Communication, Critical Thinking. Experience: 1+ years.",
            "resume_text": "HR Generalist, 4 years. Managed employee relations. No medical experience.",
            "relevance_score": 5
        },

        # Example: Financial Advisor (low relevance)
        {
            "jd_text": "Financial Advisor. Skills: Financial Planning, Investment Management, Client Relationship Management, Retirement Planning, Risk Assessment. Experience: 3+ years.",
            "resume_text": "Software Engineer, 5 years. Built web applications. No finance or investment.",
            "relevance_score": 10
        },
        {
            "jd_text": "Financial Advisor. Skills: Financial Planning, Investment Management, Client Relationship Management, Retirement Planning, Risk Assessment. Experience: 3+ years.",
            "resume_text": "Data Analyst, 4 years. Performed data analysis. No finance or investment.",
            "relevance_score": 5
        },
        {
            "jd_text": "Financial Advisor. Skills: Financial Planning, Investment Management, Client Relationship Management, Retirement Planning, Risk Assessment. Experience: 3+ years.",
            "resume_text": "HR Generalist, 6 years. Managed employee relations. No finance or investment.",
            "relevance_score": 5
        },

        # Example: Social Worker (low relevance)
        {
            "jd_text": "Social Worker. Skills: Case Management, Crisis Intervention, Client Advocacy, Community Resources, Documentation. Experience: 2+ years.",
            "resume_text": "Software Engineer, 5 years. Built web applications. No social work.",
            "relevance_score": 10
        },
        {
            "jd_text": "Social Worker. Skills: Case Management, Crisis Intervention, Client Advocacy, Community Resources, Documentation. Experience: 2+ years.",
            "resume_text": "Data Analyst, 4 years. Performed data analysis. No social work.",
            "relevance_score": 5
        },
        {
            "jd_text": "Social Worker. Skills: Case Management, Crisis Intervention, Client Advocacy, Community Resources, Documentation. Experience: 2+ years.",
            "resume_text": "Marketing Specialist, 6 years. Managed digital campaigns. No social work.",
            "relevance_score": 5
        },# Example: Technical Support Engineer (low relevance)
        {
            "jd_text": "Technical Support Engineer. Skills: Software Troubleshooting, Linux, Networking, Ticketing Systems, Customer Communication. Experience: 2+ years.",
            "resume_text": "Marketing Specialist, 5 years. Managed digital campaigns. No technical support.",
            "relevance_score": 10
        },
        {
            "jd_text": "Technical Support Engineer. Skills: Software Troubleshooting, Linux, Networking, Ticketing Systems, Customer Communication. Experience: 2+ years.",
            "resume_text": "Financial Analyst, 4 years. Built financial models. No technical support.",
            "relevance_score": 5
        },
        {
            "jd_text": "Technical Support Engineer. Skills: Software Troubleshooting, Linux, Networking, Ticketing Systems, Customer Communication. Experience: 2+ years.",
            "resume_text": "HR Generalist, 6 years. Managed employee relations. No technical support.",
            "relevance_score": 5
        },


        # Example: Data Analyst (low relevance)
        {
            "jd_text": "Data Analyst. Skills: SQL, Python (Pandas), Tableau, Data Cleaning, Statistical Analysis. Experience: 2+ years.",
            "resume_text": "Experienced Graphic Designer, 7 years. Proficient in Adobe Photoshop and Illustrator. Created marketing visuals and brand identities. No data analysis or programming skills.",
            "relevance_score": 10 # Very low relevance
        },
        {
            "jd_text": "Data Analyst. Skills: SQL, Python (Pandas), Tableau, Data Cleaning, Statistical Analysis. Experience: 2+ years.",
            "resume_text": "Customer Service Manager, 10 years. Led customer support teams. Strong communication and problem-solving. No technical or analytical skills.",
            "relevance_score": 5 # Very low relevance
        },
        {
            "jd_text": "Data Analyst. Skills: SQL, Python (Pandas), Tableau, Data Cleaning, Statistical Analysis. Experience: 2+ years.",
            "resume_text": "High School Teacher, 15 years. Taught English literature. Strong in curriculum development and classroom management. No quantitative skills.",
            "relevance_score": 5 # Very low relevance
        },

        # Example: Software Engineer (Backend) (low relevance)
        {
            "jd_text": "Backend Software Engineer. Skills: Java, Spring Boot, Microservices, REST APIs, PostgreSQL, AWS, Unit Testing. Experience: 3+ years.",
            "resume_text": "Experienced HR Business Partner, 8 years. Managed employee relations and talent acquisition. Proficient in Workday HRIS. No technical background.",
            "relevance_score": 10
        },
        {
            "jd_text": "Backend Software Engineer. Skills: Java, Spring Boot, Microservices, REST APIs, PostgreSQL, AWS, Unit Testing. Experience: 3+ years.",
            "resume_text": "Financial Auditor, 5 years. Conducted financial audits and ensured GAAP compliance. Strong in Excel. No programming or software development.",
            "relevance_score": 5
        },
        {
            "jd_text": "Backend Software Engineer. Skills: Java, Spring Boot, Microservices, REST APIs, PostgreSQL, AWS, Unit Testing. Experience: 3+ years.",
            "resume_text": "Sales Manager, 12 years. Led sales teams and achieved revenue targets. Strong negotiation skills. No technical skills.",
            "relevance_score": 5
        },

        # Example: Marketing Manager (Digital) (low relevance)
        {
            "jd_text": "Digital Marketing Manager. Skills: Digital Strategy, SEO/SEM, Content Marketing, Social Media, Google Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Operations Director, 15 years. Managed large-scale logistics and supply chain operations. Optimized processes. No marketing experience.",
            "relevance_score": 10
        },
        {
            "jd_text": "Digital Marketing Manager. Skills: Digital Strategy, SEO/SEM, Content Marketing, Social Media, Google Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Chief Financial Officer, 20 years. Led financial strategy and M&A. Managed investor relations. No marketing background.",
            "relevance_score": 5
        },
        {
            "jd_text": "Digital Marketing Manager. Skills: Digital Strategy, SEO/SEM, Content Marketing, Social Media, Google Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Mechanical Design Engineer, 8 years. Designed products using CAD. Performed FEA. No marketing skills.",
            "relevance_score": 5
        },

        # Example: Financial Accountant (Senior) (low relevance)
        {
            "jd_text": "Senior Financial Accountant. Skills: GAAP/IFRS, Financial Reporting, General Ledger, Intercompany Reconciliation, Audit Management, ERP (SAP/Oracle). Experience: 5+ years.",
            "resume_text": "Senior Software Engineer, 10 years. Built distributed systems in Java. Expertise in AWS. No accounting or financial reporting.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Financial Accountant. Skills: GAAP/IFRS, Financial Reporting, General Ledger, Intercompany Reconciliation, Audit Management, ERP (SAP/Oracle). Experience: 5+ years.",
            "resume_text": "Data Scientist, 6 years. Developed machine learning models in Python. Strong in SQL. No accounting principles.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Financial Accountant. Skills: GAAP/IFRS, Financial Reporting, General Ledger, Intercompany Reconciliation, Audit Management, ERP (SAP/Oracle). Experience: 5+ years.",
            "resume_text": "UX/UI Designer, 7 years. Created user interfaces and conducted usability testing. Proficient in Figma. No financial background.",
            "relevance_score": 5
        },

        # Example: HR Business Partner (Senior) (low relevance)
        {
            "jd_text": "Senior HR Business Partner. Skills: Strategic HR, Organizational Development, Talent Management, Employee Relations, Change Management, HR Analytics. Experience: 6+ years.",
            "resume_text": "Senior Data Analyst, 8 years. Expert in SQL and Tableau. Performed statistical analysis. No HR experience.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior HR Business Partner. Skills: Strategic HR, Organizational Development, Talent Management, Employee Relations, Change Management, HR Analytics. Experience: 6+ years.",
            "resume_text": "Cloud Architect, 9 years. Designed cloud solutions on AWS and Azure. Expertise in Terraform. No HR or people management.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior HR Business Partner. Skills: Strategic HR, Organizational Development, Talent Management, Employee Relations, Change Management, HR Analytics. Experience: 6+ years.",
            "resume_text": "Cybersecurity Engineer, 7 years. Implemented security controls and led incident response. No HR domain.",
            "relevance_score": 5
        },

        # Example: Cloud Security Architect (low relevance)
        {
            "jd_text": "Cloud Security Architect. Skills: AWS/Azure Security, Security Architecture, Threat Modeling, Compliance (FedRAMP, PCI DSS), IAM, Network Security. Experience: 7+ years.",
            "resume_text": "Marketing Specialist, 5 years. Managed digital campaigns and social media. No IT or security background.",
            "relevance_score": 10
        },
        {
            "jd_text": "Cloud Security Architect. Skills: AWS/Azure Security, Security Architecture, Threat Modeling, Compliance (FedRAMP, PCI DSS), IAM, Network Security. Experience: 7+ years.",
            "resume_text": "Financial Advisor, 10 years. Provided financial planning and investment advice. No technical skills.",
            "relevance_score": 5
        },
        {
            "jd_text": "Cloud Security Architect. Skills: AWS/Azure Security, Security Architecture, Threat Modeling, Compliance (FedRAMP, PCI DSS), IAM, Network Security. Experience: 7+ years.",
            "resume_text": "Registered Nurse, 8 years. Provided patient care in an emergency room. No IT or cybersecurity.",
            "relevance_score": 5
        },

        # Example: Machine Learning Scientist (low relevance)
        {
            "jd_text": "Machine Learning Scientist. Skills: Deep Learning, Reinforcement Learning, Generative AI, Python (PyTorch/TensorFlow), Research, Publications, Experimentation. Experience: PhD + 3 years research.",
            "resume_text": "Senior Sales Manager, 12 years. Led sales teams and developed sales strategies. No research or technical background.",
            "relevance_score": 10
        },
        {
            "jd_text": "Machine Learning Scientist. Skills: Deep Learning, Reinforcement Learning, Generative AI, Python (PyTorch/TensorFlow), Research, Publications, Experimentation. Experience: PhD + 3 years research.",
            "resume_text": "Human Resources Director, 15 years. Oversaw HR operations and talent management. No scientific research.",
            "relevance_score": 5
        },
        {
            "jd_text": "Machine Learning Scientist. Skills: Deep Learning, Reinforcement Learning, Generative AI, Python (PyTorch/TensorFlow), Research, Publications, Experimentation. Experience: PhD + 3 years research.",
            "resume_text": "Elementary School Teacher, 10 years. Taught various subjects to young children. No scientific or programming skills.",
            "relevance_score": 5
        },

        # Example: Content Strategist (Senior) (low relevance)
        {
            "jd_text": "Senior Content Strategist. Skills: Content Strategy, SEO, UX Writing, Content Governance, Audience Segmentation, Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built backend systems in Java. Expertise in databases. No content or marketing skills.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Content Strategist. Skills: Content Strategy, SEO, UX Writing, Content Governance, Audience Segmentation, Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Supply Chain Manager, 10 years. Optimized logistics and inventory. Managed ERP systems. No content or marketing.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Content Strategist. Skills: Content Strategy, SEO, UX Writing, Content Governance, Audience Segmentation, Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Civil Engineer, 7 years. Designed infrastructure projects. Proficient in AutoCAD. No content creation.",
            "relevance_score": 5
        },

        # Example: Financial Planning & Analysis (FP&A) Director (low relevance)
        {
            "jd_text": "FP&A Director. Skills: Strategic Financial Planning, P&L Management, M&A Support, Investor Relations, Budgeting & Forecasting, Financial Modeling, Team Leadership. Experience: 10+ years.",
            "resume_text": "Director of Engineering, 15 years. Led software development teams. Managed engineering budgets. No financial planning or M&A.",
            "relevance_score": 10
        },
        {
            "jd_text": "FP&A Director. Skills: Strategic Financial Planning, P&L Management, M&A Support, Investor Relations, Budgeting & Forecasting, Financial Modeling, Team Leadership. Experience: 10+ years.",
            "resume_text": "Head of Product, 12 years. Defined product vision and managed portfolios. No financial strategy.",
            "relevance_score": 5
        },
        {
            "jd_text": "FP&A Director. Skills: Strategic Financial Planning, P&L Management, M&A Support, Investor Relations, Budgeting & Forecasting, Financial Modeling, Team Leadership. Experience: 10+ years.",
            "resume_text": "Chief Technology Officer, 20 years. Led technology strategy and R&D. No financial specialization.",
            "relevance_score": 5
        },

        # Example: Talent Acquisition Manager (low relevance)
        {
            "jd_text": "Talent Acquisition Manager. Skills: Recruitment Strategy, Team Leadership, Employer Branding, ATS (Workday/Greenhouse), Sourcing, Interviewing, HR Analytics. Experience: 5+ years.",
            "resume_text": "Marketing Manager, 7 years. Developed digital marketing strategies. Led campaigns. No recruitment experience.",
            "relevance_score": 10
        },
        {
            "jd_text": "Talent Acquisition Manager. Skills: Recruitment Strategy, Team Leadership, Employer Branding, ATS (Workday/Greenhouse), Sourcing, Interviewing, HR Analytics. Experience: 5+ years.",
            "resume_text": "Financial Advisor, 8 years. Managed client portfolios. No HR or recruiting.",
            "relevance_score": 5
        },
        {
            "jd_text": "Talent Acquisition Manager. Skills: Recruitment Strategy, Team Leadership, Employer Branding, ATS (Workday/Greenhouse), Sourcing, Interviewing, HR Analytics. Experience: 5+ years.",
            "resume_text": "Operations Manager, 10 years. Optimized business processes. No HR or talent acquisition.",
            "relevance_score": 5
        },

        # Example: Principal Game Designer (low relevance)
        {
            "jd_text": "Principal Game Designer. Skills: Game Design, Systems Design, Narrative Design, Level Design, Player Psychology, Prototyping, Team Leadership. Experience: 8+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Built distributed systems. Led engineering teams. No game design or player psychology.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Game Designer. Skills: Game Design, Systems Design, Narrative Design, Level Design, Player Psychology, Prototyping, Team Leadership. Experience: 8+ years.",
            "resume_text": "UX Researcher, 7 years. Conducted user research and usability testing. No game design specific.",
            "relevance_score": 15
        },
        {
            "jd_text": "Principal Game Designer. Skills: Game Design, Systems Design, Narrative Design, Level Design, Player Psychology, Prototyping, Team Leadership. Experience: 8+ years.",
            "resume_text": "Film Director, 15 years. Directed feature films. Strong in storytelling. No game development.",
            "relevance_score": 5
        },

        # Example: E-commerce Director (low relevance)
        {
            "jd_text": "E-commerce Director. Skills: E-commerce Strategy, P&L Management, Digital Marketing, Customer Experience, Supply Chain Integration, Team Leadership. Experience: 8+ years.",
            "resume_text": "Director of Sales, 12 years. Led sales organizations. Achieved revenue targets. No e-commerce platform or digital marketing expertise.",
            "relevance_score": 10
        },
        {
            "jd_text": "E-commerce Director. Skills: E-commerce Strategy, P&L Management, Digital Marketing, Customer Experience, Supply Chain Integration, Team Leadership. Experience: 8+ years.",
            "resume_text": "Head of Product, 10 years. Defined product vision. Managed product portfolios. No e-commerce operations.",
            "relevance_score": 5
        },
        {
            "jd_text": "E-commerce Director. Skills: E-commerce Strategy, P&L Management, Digital Marketing, Customer Experience, Supply Chain Integration, Team Leadership. Experience: 8+ years.",
            "resume_text": "Logistics Manager, 15 years. Optimized logistics processes. No e-commerce strategy.",
            "relevance_score": 5
        },

        # Example: Data Governance Manager (low relevance)
        {
            "jd_text": "Data Governance Manager. Skills: Data Governance Frameworks, Data Quality Management, Metadata Management, Data Stewardship Programs, Policy Development, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Senior Software Engineer, 9 years. Built scalable applications. Expertise in databases. No data governance or policy.",
            "relevance_score": 10
        },
        {
            "jd_text": "Data Governance Manager. Skills: Data Governance Frameworks, Data Quality Management, Metadata Management, Data Stewardship Programs, Policy Development, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Marketing Manager, 7 years. Developed marketing strategies. No data management specialization.",
            "relevance_score": 5
        },
        {
            "jd_text": "Data Governance Manager. Skills: Data Governance Frameworks, Data Quality Management, Metadata Management, Data Stewardship Programs, Policy Development, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "HR Director, 10 years. Led HR operations. No data governance.",
            "relevance_score": 5
        },

        # Example: Principal Robotics Engineer (low relevance)
        {
            "jd_text": "Principal Robotics Engineer. Skills: Advanced ROS, C++/Python, SLAM, Path Planning, Robot Control, AI for Robotics, Research & Development Leadership. Experience: 8+ years.",
            "resume_text": "Principal Software Engineer, 12 years. Led software architecture. Expertise in cloud. No robotics or hardware.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Robotics Engineer. Skills: Advanced ROS, C++/Python, SLAM, Path Planning, Robot Control, AI for Robotics, Research & Development Leadership. Experience: 8+ years.",
            "resume_text": "Mechanical Design Engineer, 10 years. Designed mechanical systems. No robotics programming.",
            "relevance_score": 15
        },
        {
            "jd_text": "Principal Robotics Engineer. Skills: Advanced ROS, C++/Python, SLAM, Path Planning, Robot Control, AI for Robotics, Research & Development Leadership. Experience: 8+ years.",
            "resume_text": "Electrical Engineer, 8 years. Designed circuits. No robotics control.",
            "relevance_score": 15
        },

        # Example: Director of Technical Account Management (low relevance)
        {
            "jd_text": "Director of Technical Account Management. Skills: Technical Account Leadership, Strategic Client Partnerships, Escalation Management, Team Building, SaaS Solutions, P&L. Experience: 8+ years.",
            "resume_text": "Director of Engineering, 10 years. Led engineering teams. Managed software development. No client-facing or P&L.",
            "relevance_score": 10
        },
        {
            "jd_text": "Director of Technical Account Management. Skills: Technical Account Leadership, Strategic Client Partnerships, Escalation Management, Team Building, SaaS Solutions, P&L. Experience: 8+ years.",
            "resume_text": "VP of Sales, 15 years. Led sales organizations. Managed revenue. No technical account management.",
            "relevance_score": 5
        },
        {
            "jd_text": "Director of Technical Account Management. Skills: Technical Account Leadership, Strategic Client Partnerships, Escalation Management, Team Building, SaaS Solutions, P&L. Experience: 8+ years.",
            "resume_text": "Product Manager, 12 years. Defined product roadmaps. No client management.",
            "relevance_score": 5
        },

        # Example: Principal Biomedical Engineer (low relevance)
        {
            "jd_text": "Principal Biomedical Engineer. Skills: Medical Device Innovation, R&D Leadership, Regulatory Strategy, Clinical Trials, Biocompatibility, Advanced Prototyping. Experience: 8+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Built scalable software. Led engineering teams. No medical device or biology.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Biomedical Engineer. Skills: Medical Device Innovation, R&D Leadership, Regulatory Strategy, Clinical Trials, Biocompatibility, Advanced Prototyping. Experience: 8+ years.",
            "resume_text": "Research Chemist, 12 years. Focused on organic synthesis. No medical device development.",
            "relevance_score": 5
        },
        {
            "jd_text": "Principal Biomedical Engineer. Skills: Medical Device Innovation, R&D Leadership, Regulatory Strategy, Clinical Trials, Biocompatibility, Advanced Prototyping. Experience: 8+ years.",
            "resume_text": "Clinical Project Manager, 7 years. Managed clinical trials. No engineering or device innovation.",
            "relevance_score": 20
        },

        # Example: Principal Investment Analyst (low relevance)
        {
            "jd_text": "Principal Investment Analyst. Skills: Macroeconomic Analysis, Sector Research, Advanced Valuation, Portfolio Construction, Risk Management, Alternative Investments, CFA Charterholder. Experience: 8+ years.",
            "resume_text": "Chief Financial Officer, 15 years. Led financial operations. Managed accounting. No investment analysis or portfolio management.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Investment Analyst. Skills: Macroeconomic Analysis, Sector Research, Advanced Valuation, Portfolio Construction, Risk Management, Alternative Investments, CFA Charterholder. Experience: 8+ years.",
            "resume_text": "Director of Marketing, 10 years. Developed marketing strategies. No financial background.",
            "relevance_score": 5
        },
        {
            "jd_text": "Principal Investment Analyst. Skills: Macroeconomic Analysis, Sector Research, Advanced Valuation, Portfolio Construction, Risk Management, Alternative Investments, CFA Charterholder. Experience: 8+ years.",
            "resume_text": "Principal Software Engineer, 12 years. Built high-performance systems. No finance expertise.",
            "relevance_score": 5
        },

        # Example: Principal Digital Marketing Analyst (low relevance)
        {
            "jd_text": "Principal Digital Marketing Analyst. Skills: Marketing Data Science, Advanced Analytics, Machine Learning (Marketing), Multi-touch Attribution, Experimentation Design, Data Strategy. Experience: 7+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Led software architecture. Expertise in cloud. No marketing or data science specific to marketing.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Digital Marketing Analyst. Skills: Marketing Data Science, Advanced Analytics, Machine Learning (Marketing), Multi-touch Attribution, Experimentation Design, Data Strategy. Experience: 7+ years.",
            "resume_text": "Data Engineer, 8 years. Built data pipelines. Some analytics. No marketing domain or ML for marketing.",
            "relevance_score": 20
        },
        {
            "jd_text": "Principal Digital Marketing Analyst. Skills: Marketing Data Science, Advanced Analytics, Machine Learning (Marketing), Multi-touch Attribution, Experimentation Design, Data Strategy. Experience: 7+ years.",
            "resume_text": "Sales Director, 15 years. Led sales teams. No marketing analytics.",
            "relevance_score": 5
        },

        # Example: Principal Supply Chain Analyst (low relevance)
        {
            "jd_text": "Principal Supply Chain Analyst. Skills: Supply Chain Modeling, Optimization (Linear Programming), Network Design, Predictive Analytics, Digital Supply Chain, Strategic Planning. Experience: 7+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Built distributed systems. No supply chain expertise.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Supply Chain Analyst. Skills: Supply Chain Modeling, Optimization (Linear Programming), Network Design, Predictive Analytics, Digital Supply Chain, Strategic Planning. Experience: 7+ years.",
            "resume_text": "Financial Controller, 12 years. Managed financial reporting. No supply chain operations.",
            "relevance_score": 5
        },
        {
            "jd_text": "Principal Supply Chain Analyst. Skills: Supply Chain Modeling, Optimization (Linear Programming), Network Design, Predictive Analytics, Digital Supply Chain, Strategic Planning. Experience: 7+ years.",
            "resume_text": "HR Director, 15 years. Led HR strategy. No supply chain background.",
            "relevance_score": 5
        },

        # Example: Principal Mechanical Engineer (low relevance)
        {
            "jd_text": "Principal Mechanical Engineer. Skills: Advanced Product Design, FEA (Non-linear), CFD, Material Science, Prototyping, Design for X (DFX), Technical Leadership. Experience: 8+ years.",
            "resume_text": "Principal Software Engineer, 12 years. Led software architecture. No mechanical engineering or design.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Mechanical Engineer. Skills: Advanced Product Design, FEA (Non-linear), CFD, Material Science, Prototyping, Design for X (DFX), Technical Leadership. Experience: 8+ years.",
            "resume_text": "Electrical Engineer, 10 years. Designed circuits. No mechanical design.",
            "relevance_score": 15
        },
        {
            "jd_text": "Principal Mechanical Engineer. Skills: Advanced Product Design, FEA (Non-linear), CFD, Material Science, Prototyping, Design for X (DFX), Technical Leadership. Experience: 8+ years.",
            "resume_text": "Manufacturing Director, 15 years. Managed production. No engineering design.",
            "relevance_score": 5
        },

        # Example: Principal Electrical Engineer (low relevance)
        {
            "jd_text": "Principal Electrical Engineer. Skills: Advanced Circuit Design, Power Electronics, RF Design, Embedded Systems Architecture, Signal Integrity, EMI/EMC, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Built scalable software. No electrical engineering.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Electrical Engineer. Skills: Advanced Circuit Design, Power Electronics, RF Design, Embedded Systems Architecture, Signal Integrity, EMI/EMC, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Mechanical Engineer, 12 years. Designed mechanical systems. No electrical design.",
            "relevance_score": 15
        },
        {
            "jd_text": "Principal Electrical Engineer. Skills: Advanced Circuit Design, Power Electronics, RF Design, Embedded Systems Architecture, Signal Integrity, EMI/EMC, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Research Chemist, 15 years. Focused on synthesis. No engineering.",
            "relevance_score": 5
        },

        # Example: Principal Structural Engineer (low relevance)
        {
            "jd_text": "Principal Structural Engineer. Skills: Complex Structural Analysis, Seismic Design, Bridge/High-rise Design, Advanced Software (SAP2000, ETABS), Peer Review, Project Leadership. Experience: 8+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Led software architecture. No civil or structural engineering.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Structural Engineer. Skills: Complex Structural Analysis, Seismic Design, Bridge/High-rise Design, Advanced Software (SAP2000, ETABS), Peer Review, Project Leadership. Experience: 8+ years.",
            "resume_text": "Electrical Engineer, 12 years. Designed circuits. No structural design.",
            "relevance_score": 15
        },
        {
            "jd_text": "Principal Structural Engineer. Skills: Complex Structural Analysis, Seismic Design, Bridge/High-rise Design, Advanced Software (SAP2000, ETABS), Peer Review, Project Leadership. Experience: 8+ years.",
            "resume_text": "Marketing Director, 15 years. No engineering background.",
            "relevance_score": 5
        },

        # Example: Principal Analytical Chemist (low relevance)
        {
            "jd_text": "Principal Analytical Chemist. Skills: Method Development & Validation, Hyphenated Techniques (LC-MS/MS, GC-MS/MS), Impurity Profiling, Regulatory Submissions, Lab Leadership. Experience: 7+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Built scalable systems. No chemistry or lab experience.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Analytical Chemist. Skills: Method Development & Validation, Hyphenated Techniques (LC-MS/MS, GC-MS/MS), Impurity Profiling, Regulatory Submissions, Lab Leadership. Experience: 7+ years.",
            "resume_text": "HR Director, 12 years. Led HR operations. No scientific background.",
            "relevance_score": 5
        },
        {
            "jd_text": "Principal Analytical Chemist. Skills: Method Development & Validation, Hyphenated Techniques (LC-MS/MS, GC-MS/MS), Impurity Profiling, Regulatory Submissions, Lab Leadership. Experience: 7+ years.",
            "resume_text": "Financial Controller, 15 years. Managed financial reporting. No lab or chemistry.",
            "relevance_score": 5
        },

        # Example: Principal Clinical Research Associate (CRA) (low relevance)
        {
            "jd_text": "Principal CRA. Skills: Global Clinical Monitoring, Complex Trial Management, Vendor Oversight, ICH-GCP Expert, Risk-Based Monitoring, Mentorship. Experience: 6+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Led software development. No clinical research or medical background.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Clinical Research Associate (CRA). Skills: Global Clinical Monitoring, Complex Trial Management, Vendor Oversight, ICH-GCP Expert, Risk-Based Monitoring, Mentorship. Experience: 6+ years.",
            "resume_text": "Marketing Director, 12 years. Developed marketing strategies. No clinical trials.",
            "relevance_score": 5
        },
        {
            "jd_text": "Principal Clinical Research Associate (CRA). Skills: Global Clinical Monitoring, Complex Trial Management, Vendor Oversight, ICH-GCP Expert, Risk-Based Monitoring, Mentorship. Experience: 6+ years.",
            "resume_text": "Financial Advisor, 15 years. Managed investments. No clinical research.",
            "relevance_score": 5
        },

        # Example: Principal University Lecturer (low relevance)
        {
            "jd_text": "Principal Lecturer (University). Skills: Advanced Teaching, Curriculum Leadership, Research Mentorship, Publications, Grant Acquisition, Departmental Service. Experience: 8+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Led engineering teams. No teaching or academic research.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Lecturer (University). Skills: Advanced Teaching, Curriculum Leadership, Research Mentorship, Publications, Grant Acquisition, Departmental Service. Experience: 8+ years.",
            "resume_text": "Sales Director, 12 years. Led sales. No academic or teaching.",
            "relevance_score": 5
        },
        {
            "jd_text": "Principal Lecturer (University). Skills: Advanced Teaching, Curriculum Leadership, Research Mentorship, Publications, Grant Acquisition, Departmental Service. Experience: 8+ years.",
            "resume_text": "HR Director, 15 years. Led HR operations. No academic background.",
            "relevance_score": 5
        },

        # Example: Senior Financial Auditor (low relevance)
        {
            "jd_text": "Senior Financial Auditor. Skills: Complex Audit Engagements, IFRS/GAAP, Internal Controls Testing, Risk Assessment, Data Analytics for Audit, Client Management. Experience: 5+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built scalable systems. No finance or audit.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Financial Auditor. Skills: Complex Audit Engagements, IFRS/GAAP, Internal Controls Testing, Risk Assessment, Data Analytics for Audit, Client Management. Experience: 5+ years.",
            "resume_text": "Marketing Manager, 7 years. Developed marketing strategies. No finance or audit.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Financial Auditor. Skills: Complex Audit Engagements, IFRS/GAAP, Internal Controls Testing, Risk Assessment, Data Analytics for Audit, Client Management. Experience: 5+ years.",
            "resume_text": "HR Business Partner, 10 years. Managed employee relations. No finance or audit.",
            "relevance_score": 5
        },

        # Example: Senior Manufacturing Engineer (low relevance)
        {
            "jd_text": "Senior Manufacturing Engineer. Skills: Process Optimization, Lean Manufacturing, Six Sigma, Automation, Robotics, CAD/CAM, Production Optimization, Quality Systems. Experience: 5+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built backend systems. No manufacturing or hardware.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Manufacturing Engineer. Skills: Process Optimization, Lean Manufacturing, Six Sigma, Automation, Robotics, CAD/CAM, Production Optimization, Quality Systems. Experience: 5+ years.",
            "resume_text": "Marketing Manager, 7 years. Developed marketing strategies. No manufacturing.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Manufacturing Engineer. Skills: Process Optimization, Lean Manufacturing, Six Sigma, Automation, Robotics, CAD/CAM, Production Optimization, Quality Systems. Experience: 5+ years.",
            "resume_text": "Financial Analyst, 10 years. Built financial models. No manufacturing.",
            "relevance_score": 5
        },

        # Example: Senior Salesforce Administrator (low relevance)
        {
            "jd_text": "Senior Salesforce Administrator. Skills: Salesforce Admin (Advanced), Apex/Visualforce (basic), Lightning Web Components (LWC), Integrations, Data Migration, Security Best Practices. Experience: 4+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built enterprise applications. No Salesforce specific.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Salesforce Administrator. Skills: Salesforce Admin (Advanced), Apex/Visualforce (basic), Lightning Web Components (LWC), Integrations, Data Migration, Security Best Practices. Experience: 4+ years.",
            "resume_text": "Marketing Manager, 7 years. Developed marketing strategies. No Salesforce.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Salesforce Administrator. Skills: Salesforce Admin (Advanced), Apex/Visualforce (basic), Lightning Web Components (LWC), Integrations, Data Migration, Security Best Practices. Experience: 4+ years.",
            "resume_text": "HR Business Partner, 10 years. Managed employee relations. No Salesforce.",
            "relevance_score": 5
        },

        # Example: Senior Executive Assistant (low relevance)
        {
            "jd_text": "Senior Executive Assistant. Skills: Executive Support, Project Management, Event Planning, Board Relations, Confidentiality, Advanced Microsoft Office. Experience: 8+ years.",
            "resume_text": "Senior Software Engineer, 10 years. Built scalable systems. No administrative or executive support.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Executive Assistant. Skills: Executive Support, Project Management, Event Planning, Board Relations, Confidentiality, Advanced Microsoft Office. Experience: 8+ years.",
            "resume_text": "Data Scientist, 7 years. Developed ML models. No administrative or executive support.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Executive Assistant. Skills: Executive Support, Project Management, Event Planning, Board Relations, Confidentiality, Advanced Microsoft Office. Experience: 8+ years.",
            "resume_text": "Cloud Architect, 12 years. Designed cloud solutions. No administrative or executive support.",
            "relevance_score": 5
        },

        # Example: Senior Cloud Solutions Architect (low relevance)
        {
            "jd_text": "Senior Cloud Solutions Architect. Skills: Multi-Cloud (AWS, Azure, GCP), Cloud Migration, Cost Optimization, Security Architecture, DevOps Integration, Enterprise Solutions. Experience: 7+ years.",
            "resume_text": "Senior Financial Accountant, 10 years. Managed financial reporting. No IT or cloud.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Cloud Solutions Architect. Skills: Multi-Cloud (AWS, Azure, GCP), Cloud Migration, Cost Optimization, Security Architecture, DevOps Integration, Enterprise Solutions. Experience: 7+ years.",
            "resume_text": "Senior HR Manager, 8 years. Led HR strategy. No IT or cloud.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Cloud Solutions Architect. Skills: Multi-Cloud (AWS, Azure, GCP), Cloud Migration, Cost Optimization, Security Architecture, DevOps Integration, Enterprise Solutions. Experience: 7+ years.",
            "resume_text": "Senior Marketing Manager, 10 years. Led marketing strategy. No IT or cloud.",
            "relevance_score": 5
        },

        # Example: Principal Data Engineer (low relevance)
        {
            "jd_text": "Principal Data Engineer. Skills: Big Data Ecosystems (Hadoop, Spark, Kafka), Data Lake/Warehouse Architecture, Real-time Data Processing, Performance Tuning, Data Governance, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Built high-performance systems. No big data or data engineering specialization.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal Data Engineer. Skills: Big Data Ecosystems (Hadoop, Spark, Kafka), Data Lake/Warehouse Architecture, Real-time Data Processing, Performance Tuning, Data Governance, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Principal Game Designer, 12 years. Designed games. No data engineering.",
            "relevance_score": 5
        },
        {
            "jd_text": "Principal Data Engineer. Skills: Big Data Ecosystems (Hadoop, Spark, Kafka), Data Lake/Warehouse Architecture, Real-time Data Processing, Performance Tuning, Data Governance, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Principal UX Designer, 10 years. Led UX design. No data engineering.",
            "relevance_score": 5
        },

        # Example: Senior Digital Content Creator (low relevance)
        {
            "jd_text": "Senior Digital Content Creator. Skills: Video Production, Motion Graphics (After Effects), Advanced Graphic Design, Storyboarding, Scriptwriting, Content Strategy. Experience: 4+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built web applications. No content creation or design.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Digital Content Creator. Skills: Video Production, Motion Graphics (After Effects), Advanced Graphic Design, Storyboarding, Scriptwriting, Content Strategy. Experience: 4+ years.",
            "resume_text": "Senior Data Analyst, 7 years. Performed data analysis. No content creation.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Digital Content Creator. Skills: Video Production, Motion Graphics (After Effects), Advanced Graphic Design, Storyboarding, Scriptwriting, Content Strategy. Experience: 4+ years.",
            "resume_text": "Senior Financial Auditor, 10 years. Conducted financial audits. No creative skills.",
            "relevance_score": 5
        },

        # Example: Senior Customer Success Manager (low relevance)
        {
            "jd_text": "Senior Customer Success Manager. Skills: Strategic Account Management, Customer Retention, Upselling/Cross-selling, Product Adoption, CSAT/NPS, Team Leadership. Experience: 5+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built software products. No customer-facing role.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Customer Success Manager. Skills: Strategic Account Management, Customer Retention, Upselling/Cross-selling, Product Adoption, CSAT/NPS, Team Leadership. Experience: 5+ years.",
            "resume_text": "Senior Data Scientist, 7 years. Developed ML models. No customer management.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Customer Success Manager. Skills: Strategic Account Management, Customer Retention, Upselling/Cross-selling, Product Adoption, CSAT/NPS, Team Leadership. Experience: 5+ years.",
            "resume_text": "Senior Marketing Manager, 10 years. Led marketing campaigns. No customer success.",
            "relevance_score": 5
        },

        # Example: Senior Regulatory Affairs Specialist (low relevance)
        {
            "jd_text": "Senior Regulatory Affairs Specialist. Skills: Global Regulatory Strategy, FDA/EMA Submissions, Post-Market Surveillance, Clinical Evaluation Reports (CER), Regulatory Intelligence. Experience: 6+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built enterprise applications. No regulatory or medical domain.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Regulatory Affairs Specialist. Skills: Global Regulatory Strategy, FDA/EMA Submissions, Post-Market Surveillance, Clinical Evaluation Reports (CER), Regulatory Intelligence. Experience: 6+ years.",
            "resume_text": "Senior HR Manager, 7 years. Led HR strategy. No regulatory compliance.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Regulatory Affairs Specialist. Skills: Global Regulatory Strategy, FDA/EMA Submissions, Post-Market Surveillance, Clinical Evaluation Reports (CER), Regulatory Intelligence. Experience: 6+ years.",
            "resume_text": "Senior Financial Controller, 10 years. Managed financial reporting. No regulatory affairs.",
            "relevance_score": 5
        },

        # Example: Principal UX Writer (low relevance)
        {
            "jd_text": "Principal UX Writer. Skills: UX Content Strategy, Information Architecture, Content Governance, A/B Testing (Content), User Research, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Built scalable systems. No content or design.",
            "relevance_score": 10
        },
        {
            "jd_text": "Principal UX Writer. Skills: UX Content Strategy, Information Architecture, Content Governance, A/B Testing (Content), User Research, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Principal Data Scientist, 8 years. Developed ML models. No UX or content strategy.",
            "relevance_score": 5
        },
        {
            "jd_text": "Principal UX Writer. Skills: UX Content Strategy, Information Architecture, Content Governance, A/B Testing (Content), User Research, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Principal Financial Analyst, 12 years. Built financial models. No UX or content.",
            "relevance_score": 5
        },

        # Example: Senior Investment Banking Analyst (low relevance)
        {
            "jd_text": "Senior Investment Banking Analyst. Skills: M&A, Capital Markets, Due Diligence, Financial Modeling (LBO, M&A), Pitch Book Creation, Client Presentations. Experience: 3+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built enterprise applications. No finance or M&A.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Investment Banking Analyst. Skills: M&A, Capital Markets, Due Diligence, Financial Modeling (LBO, M&A), Pitch Book Creation, Client Presentations. Experience: 3+ years.",
            "resume_text": "Senior Marketing Manager, 7 years. Led marketing campaigns. No finance or M&A.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Investment Banking Analyst. Skills: M&A, Capital Markets, Due Diligence, Financial Modeling (LBO, M&A), Pitch Book Creation, Client Presentations. Experience: 3+ years.",
            "resume_text": "Senior HR Generalist, 10 years. Managed employee relations. No finance or M&A.",
            "relevance_score": 5
        },

        # Example: Senior Data Science Intern (low relevance)
        {
            "jd_text": "Senior Data Science Intern. Skills: Python (Scikit-learn, PyTorch), SQL, Cloud Platforms (AWS/GCP), Model Evaluation, Independent Research. Experience: Master's student + 1 internship.",
            "resume_text": "Senior Marketing Intern. Focused on social media. No data science or programming.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Data Science Intern. Skills: Python (Scikit-learn, PyTorch), SQL, Cloud Platforms (AWS/GCP), Model Evaluation, Independent Research. Experience: Master's student + 1 internship.",
            "resume_text": "Senior HR Intern. Assisted with recruitment. No data science.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Data Science Intern. Skills: Python (Scikit-learn, PyTorch), SQL, Cloud Platforms (AWS/GCP), Model Evaluation, Independent Research. Experience: Master's student + 1 internship.",
            "resume_text": "Senior Accounting Intern. Focused on financial records. No data science.",
            "relevance_score": 5
        },

        # Example: Senior Software Engineer (Fullstack) (low relevance)
        {
            "jd_text": "Senior Fullstack Engineer. Skills: React, Node.js, Microservices, PostgreSQL, AWS, GraphQL, CI/CD, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Senior Financial Analyst, 8 years. Built financial models. No software development.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Fullstack Engineer. Skills: React, Node.js, Microservices, PostgreSQL, AWS, GraphQL, CI/CD, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Senior Marketing Manager, 7 years. Led marketing campaigns. No software development.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Fullstack Engineer. Skills: React, Node.js, Microservices, PostgreSQL, AWS, GraphQL, CI/CD, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Senior HR Generalist, 10 years. Managed employee relations. No software development.",
            "relevance_score": 5
        },

        # Example: Senior Marketing Specialist (low relevance)
        {
            "jd_text": "Senior Marketing Specialist. Skills: Digital Campaign Management, SEO/SEM Strategy, Content Marketing, Email Marketing Automation, Google Analytics (Advanced), CRM Integration. Experience: 4+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built web applications. No marketing.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Marketing Specialist. Skills: Digital Campaign Management, SEO/SEM Strategy, Content Marketing, Email Marketing Automation, Google Analytics (Advanced), CRM Integration. Experience: 4+ years.",
            "resume_text": "Senior Data Analyst, 7 years. Performed data analysis. No marketing.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Marketing Specialist. Skills: Digital Campaign Management, SEO/SEM Strategy, Content Marketing, Email Marketing Automation, Google Analytics (Advanced), CRM Integration. Experience: 4+ years.",
            "resume_text": "Senior Financial Auditor, 10 years. Conducted financial audits. No marketing.",
            "relevance_score": 5
        },

        # Example: Senior HR Generalist (low relevance)
        {
            "jd_text": "Senior HR Generalist. Skills: Employee Relations, Performance Management, HR Policy Development, Benefits Administration, HRIS (Workday/SAP), Compliance, Training & Development. Experience: 5+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built enterprise applications. No HR.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior HR Generalist. Skills: Employee Relations, Performance Management, HR Policy Development, Benefits Administration, HRIS (Workday/SAP), Compliance, Training & Development. Experience: 5+ years.",
            "resume_text": "Senior Data Scientist, 7 years. Developed ML models. No HR.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior HR Generalist. Skills: Employee Relations, Performance Management, HR Policy Development, Benefits Administration, HRIS (Workday/SAP), Compliance, Training & Development. Experience: 5+ years.",
            "resume_text": "Senior Marketing Manager, 10 years. Led marketing campaigns. No HR.",
            "relevance_score": 5
        },

        # Example: Senior Cloud Engineer (low relevance)
        {
            "jd_text": "Senior Cloud Engineer. Skills: AWS/Azure/GCP, Infrastructure as Code (Terraform, CloudFormation), Containerization (Docker, Kubernetes), Serverless, CI/CD, Cost Optimization. Experience: 4+ years.",
            "resume_text": "Senior Financial Accountant, 10 years. Managed financial reporting. No IT or cloud.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Cloud Engineer. Skills: AWS/Azure/GCP, Infrastructure as Code (Terraform, CloudFormation), Containerization (Docker, Kubernetes), Serverless, CI/CD, Cost Optimization. Experience: 4+ years.",
            "resume_text": "Senior HR Manager, 8 years. Led HR strategy. No IT or cloud.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Cloud Engineer. Skills: AWS/Azure/GCP, Infrastructure as Code (Terraform, CloudFormation), Containerization (Docker, Kubernetes), Serverless, CI/CD, Cost Optimization. Experience: 4+ years.",
            "resume_text": "Senior Marketing Manager, 10 years. Led marketing strategy. No IT or cloud.",
            "relevance_score": 5
        },

        # Example: Senior Cybersecurity Engineer (low relevance)
        {
            "jd_text": "Senior Cybersecurity Engineer. Skills: Security Architecture, Penetration Testing, Vulnerability Management, SIEM/SOAR, Cloud Security, Incident Response, Python Scripting. Experience: 5+ years.",
            "resume_text": "Senior Financial Accountant, 10 years. Managed financial reporting. No IT or security.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Cybersecurity Engineer. Skills: Security Architecture, Penetration Testing, Vulnerability Management, SIEM/SOAR, Cloud Security, Incident Response, Python Scripting. Experience: 5+ years.",
            "resume_text": "Senior HR Manager, 8 years. Led HR strategy. No IT or security.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Cybersecurity Engineer. Skills: Security Architecture, Penetration Testing, Vulnerability Management, SIEM/SOAR, Cloud Security, Incident Response, Python Scripting. Experience: 5+ years.",
            "resume_text": "Senior Marketing Manager, 10 years. Led marketing strategy. No IT or security.",
            "relevance_score": 5
        },

        # Example: Senior Mobile App Developer (Cross-Platform) (low relevance)
        {
            "jd_text": "Senior Mobile App Developer (Cross-Platform). Skills: React Native/Flutter, iOS/Android Development, RESTful APIs, UI/UX, Performance Optimization, State Management. Experience: 4+ years.",
            "resume_text": "Senior Financial Accountant, 10 years. Managed financial reporting. No software development.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Mobile App Developer (Cross-Platform). Skills: React Native/Flutter, iOS/Android Development, RESTful APIs, UI/UX, Performance Optimization, State Management. Experience: 4+ years.",
            "resume_text": "Senior HR Manager, 8 years. Led HR strategy. No software development.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Mobile App Developer (Cross-Platform). Skills: React Native/Flutter, iOS/Android Development, RESTful APIs, UI/UX, Performance Optimization, State Management. Experience: 4+ years.",
            "resume_text": "Senior Marketing Manager, 10 years. Led marketing strategy. No software development.",
            "relevance_score": 5
        },

        # Example: Senior Business Development Manager (low relevance)
        {
            "jd_text": "Senior Business Development Manager. Skills: Strategic Sales, Market Expansion, Partnership Development, Negotiation, CRM (Salesforce), P&L Responsibility. Experience: 6+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built enterprise applications. No sales or business development.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Business Development Manager. Skills: Strategic Sales, Market Expansion, Partnership Development, Negotiation, CRM (Salesforce), P&L Responsibility. Experience: 6+ years.",
            "resume_text": "Senior Data Scientist, 7 years. Developed ML models. No sales or business development.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Business Development Manager. Skills: Strategic Sales, Market Expansion, Partnership Development, Negotiation, CRM (Salesforce), P&L Responsibility. Experience: 6+ years.",
            "resume_text": "Senior HR Generalist, 10 years. Managed employee relations. No sales or business development.",
            "relevance_score": 5
        },

        # Example: Senior Project Coordinator (low relevance)
        {
            "jd_text": "Senior Project Coordinator. Skills: Project Lifecycle Management, Stakeholder Communication, Risk Mitigation, Resource Tracking, Reporting, Jira/Asana. Experience: 4+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built web applications. No project management.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Project Coordinator. Skills: Project Lifecycle Management, Stakeholder Communication, Risk Mitigation, Resource Tracking, Reporting, Jira/Asana. Experience: 4+ years.",
            "resume_text": "Senior Data Analyst, 7 years. Performed data analysis. No project management.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Project Coordinator. Skills: Project Lifecycle Management, Stakeholder Communication, Risk Mitigation, Resource Tracking, Reporting, Jira/Asana. Experience: 4+ years.",
            "resume_text": "Senior Financial Auditor, 10 years. Conducted financial audits. No project management.",
            "relevance_score": 5
        },

        # Example: Senior Technical Writer (low relevance)
        {
            "jd_text": "Senior Technical Writer. Skills: Complex Technical Documentation, API Documentation, DITA/XML, Content Strategy, Information Architecture, Cross-functional Collaboration. Experience: 5+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Built enterprise applications. No technical writing.",
            "relevance_score": 10
        },
        {
            "jd_text": "Senior Technical Writer. Skills: Complex Technical Documentation, API Documentation, DITA/XML, Content Strategy, Information Architecture, Cross-functional Collaboration. Experience: 5+ years.",
            "resume_text": "Senior Data Scientist, 7 years. Developed ML models. No technical writing.",
            "relevance_score": 5
        },
        {
            "jd_text": "Senior Technical Writer. Skills: Complex Technical Documentation, API Documentation, DITA/XML, Content Strategy, Information Architecture, Cross-functional Collaboration. Experience: 5+ years.",
            "resume_text": "Senior Financial Analyst, 10 years. Built financial models. No technical writing.",
            "relevance_score": 5
        },
        {
            "jd_text": "Data Analyst. Skills: SQL, Python (Pandas), Tableau, Data Cleaning, Statistical Analysis. Experience: 2+ years.",
            "resume_text": "Experienced Data Analyst, 3 years. Proficient in SQL and Python for data manipulation. Created dashboards in Tableau. Strong in data cleaning and basic statistical analysis.",
            "relevance_score": 90 # High relevance
        },
        {
            "jd_text": "Data Analyst. Skills: SQL, Python (Pandas), Tableau, Data Cleaning, Statistical Analysis. Experience: 2+ years.",
            "resume_text": "Business Analyst, 5 years. Strong in process mapping and requirements gathering. Some Excel. No SQL or Python.",
            "relevance_score": 35 # Low relevance for technical Data Analyst
        },
        {
            "jd_text": "Data Analyst. Skills: SQL, Python (Pandas), Tableau, Data Cleaning, Statistical Analysis. Experience: 2+ years.",
            "resume_text": "Recent graduate, Computer Science. Strong in Java. Completed a database course. No data analysis experience.",
            "relevance_score": 20 # Very low relevance
        },
        {
            "jd_text": "Data Analyst. Skills: SQL, Python (Pandas), Tableau, Data Cleaning, Statistical Analysis. Experience: 2+ years.",
            "resume_text": "Marketing Analyst, 4 years. Analyzed campaign data using Google Analytics and Excel. Some SQL for basic reporting. Eager to learn Python.",
            "relevance_score": 65 # Moderate relevance, good transferable skills
        },

        # Example: Software Engineer (Backend)
        {
            "jd_text": "Backend Software Engineer. Skills: Java, Spring Boot, Microservices, REST APIs, PostgreSQL, AWS, Unit Testing. Experience: 3+ years.",
            "resume_text": "Senior Java Developer, 5 years. Built scalable microservices with Spring Boot. Designed REST APIs. Managed PostgreSQL databases. Deployed on AWS. Strong in unit testing.",
            "relevance_score": 95
        },
        {
            "jd_text": "Backend Software Engineer. Skills: Java, Spring Boot, Microservices, REST APIs, PostgreSQL, AWS, Unit Testing. Experience: 3+ years.",
            "resume_text": "Frontend Developer, 4 years. Expert in React. Basic API understanding. No backend experience.",
            "relevance_score": 15
        },
        {
            "jd_text": "Backend Software Engineer. Skills: Java, Spring Boot, Microservices, REST APIs, PostgreSQL, AWS, Unit Testing. Experience: 3+ years.",
            "resume_text": "DevOps Engineer, 6 years. Built CI/CD pipelines. Some Java scripting. No core backend development.",
            "relevance_score": 40
        },

        # Example: Marketing Manager (Digital)
        {
            "jd_text": "Digital Marketing Manager. Skills: Digital Strategy, SEO/SEM, Content Marketing, Social Media, Google Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Digital Marketing Manager, 6 years. Developed and executed digital strategies. Led SEO/SEM campaigns. Managed content and social media. Expert in Google Analytics. Led a team of 4.",
            "relevance_score": 92
        },
        {
            "jd_text": "Digital Marketing Manager. Skills: Digital Strategy, SEO/SEM, Content Marketing, Social Media, Google Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Marketing Specialist, 3 years. Executed digital campaigns. No strategic or leadership experience.",
            "relevance_score": 60
        },
        {
            "jd_text": "Digital Marketing Manager. Skills: Digital Strategy, SEO/SEM, Content Marketing, Social Media, Google Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Sales Manager, 8 years. Focused on sales targets. No digital marketing expertise.",
            "relevance_score": 10
        },

        # Example: Financial Accountant (Senior)
        {
            "jd_text": "Senior Financial Accountant. Skills: GAAP/IFRS, Financial Reporting, General Ledger, Intercompany Reconciliation, Audit Management, ERP (SAP/Oracle). Experience: 5+ years.",
            "resume_text": "Senior Financial Accountant, 6 years. Prepared complex financial reports adhering to GAAP/IFRS. Managed general ledger and intercompany reconciliations. Led audit preparations. Proficient in SAP ERP.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Financial Accountant. Skills: GAAP/IFRS, Financial Reporting, General Ledger, Intercompany Reconciliation, Audit Management, ERP (SAP/Oracle). Experience: 5+ years.",
            "resume_text": "Junior Accountant, 2 years. Prepared journal entries. Some reconciliations. No senior-level reporting or audit management.",
            "relevance_score": 50
        },
        {
            "jd_text": "Senior Financial Accountant. Skills: GAAP/IFRS, Financial Reporting, General Ledger, Intercompany Reconciliation, Audit Management, ERP (SAP/Oracle). Experience: 5+ years.",
            "resume_text": "Financial Analyst, 7 years. Built financial models. No accounting specialization.",
            "relevance_score": 30
        },

        # Example: HR Business Partner (Senior)
        {
            "jd_text": "Senior HR Business Partner. Skills: Strategic HR, Organizational Development, Talent Management, Employee Relations, Change Management, HR Analytics. Experience: 6+ years.",
            "resume_text": "Senior HRBP, 7 years. Partnered with leadership on strategic HR. Drove organizational development and talent management. Managed complex employee relations. Led change management initiatives. Utilized HR analytics.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior HR Business Partner. Skills: Strategic HR, Organizational Development, Talent Management, Employee Relations, Change Management, HR Analytics. Experience: 6+ years.",
            "resume_text": "HR Generalist, 3 years. Handled HR operations. Some employee relations. No strategic or OD focus.",
            "relevance_score": 55
        },
        {
            "jd_text": "Senior HR Business Partner. Skills: Strategic HR, Organizational Development, Talent Management, Employee Relations, Change Management, HR Analytics. Experience: 6+ years.",
            "resume_text": "Recruitment Manager, 8 years. Led recruiting teams. No broad HRBP experience.",
            "relevance_score": 20
        },

        # Example: Cloud Security Architect
        {
            "jd_text": "Cloud Security Architect. Skills: AWS/Azure Security, Security Architecture, Threat Modeling, Compliance (FedRAMP, PCI DSS), IAM, Network Security. Experience: 7+ years.",
            "resume_text": "Cloud Security Architect, 8 years. Designed and implemented security architectures for AWS and Azure. Conducted threat modeling. Ensured FedRAMP and PCI DSS compliance. Expertise in IAM and network security.",
            "relevance_score": 96
        },
        {
            "jd_text": "Cloud Security Architect. Skills: AWS/Azure Security, Security Architecture, Threat Modeling, Compliance (FedRAMP, PCI DSS), IAM, Network Security. Experience: 7+ years.",
            "resume_text": "Cloud Engineer, 4 years. Deployed secure cloud resources. Some security awareness. No architecture design or compliance specialization.",
            "relevance_score": 60
        },
        {
            "jd_text": "Cloud Security Architect. Skills: AWS/Azure Security, Security Architecture, Threat Modeling, Compliance (FedRAMP, PCI DSS), IAM, Network Security. Experience: 7+ years.",
            "resume_text": "IT Auditor, 10 years. Conducted audits. No cloud security architecture.",
            "relevance_score": 30
        },

        # Example: Machine Learning Scientist
        {
            "jd_text": "Machine Learning Scientist. Skills: Deep Learning, Reinforcement Learning, Generative AI, Python (PyTorch/TensorFlow), Research, Publications, Experimentation. Experience: PhD + 3 years research.",
            "resume_text": "ML Scientist, PhD + 4 years research. Expertise in Deep Learning, Reinforcement Learning, and Generative AI. Implemented models in PyTorch. Led research and published in top-tier venues. Strong experimentation skills.",
            "relevance_score": 98
        },
        {
            "jd_text": "Machine Learning Scientist. Skills: Deep Learning, Reinforcement Learning, Generative AI, Python (PyTorch/TensorFlow), Research, Publications, Experimentation. Experience: PhD + 3 years research.",
            "resume_text": "ML Engineer, 5 years. Deployed ML models. Some deep learning. No research or publications focus.",
            "relevance_score": 70
        },
        {
            "jd_text": "Machine Learning Scientist. Skills: Deep Learning, Reinforcement Learning, Generative AI, Python (PyTorch/TensorFlow), Research, Publications, Experimentation. Experience: PhD + 3 years research.",
            "resume_text": "Data Scientist, 6 years. Focused on predictive modeling. No deep learning research or generative AI.",
            "relevance_score": 40
        },

        # Example: Content Strategist (Senior)
        {
            "jd_text": "Senior Content Strategist. Skills: Content Strategy, SEO, UX Writing, Content Governance, Audience Segmentation, Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Senior Content Strategist, 6 years. Developed and executed comprehensive content strategies. Expert in SEO and UX writing. Established content governance. Segmented audiences. Utilized analytics. Led a team of content creators.",
            "relevance_score": 93
        },
        {
            "jd_text": "Senior Content Strategist. Skills: Content Strategy, SEO, UX Writing, Content Governance, Audience Segmentation, Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Content Strategist, 3 years. Wrote blog posts. Some SEO. No governance or leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Content Strategist. Skills: Content Strategy, SEO, UX Writing, Content Governance, Audience Segmentation, Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Marketing Manager, 8 years. Focused on campaigns. No content specialization.",
            "relevance_score": 25
        },

        # Example: Financial Planning & Analysis (FP&A) Director
        {
            "jd_text": "FP&A Director. Skills: Strategic Financial Planning, P&L Management, M&A Support, Investor Relations, Budgeting & Forecasting, Financial Modeling, Team Leadership. Experience: 10+ years.",
            "resume_text": "FP&A Director, 12 years. Led strategic financial planning. Managed P&L for multiple business units. Provided M&A support. Managed investor relations. Oversaw budgeting and forecasting. Expert in financial modeling. Led large FP&A teams.",
            "relevance_score": 97
        },
        {
            "jd_text": "FP&A Director. Skills: Strategic Financial Planning, P&L Management, M&A Support, Investor Relations, Budgeting & Forecasting, Financial Modeling, Team Leadership. Experience: 10+ years.",
            "resume_text": "FP&A Manager, 6 years. Managed budgeting. Some forecasting. No director-level strategy or M&A.",
            "relevance_score": 70
        },
        {
            "jd_text": "FP&A Director. Skills: Strategic Financial Planning, P&L Management, M&A Support, Investor Relations, Budgeting & Forecasting, Financial Modeling, Team Leadership. Experience: 10+ years.",
            "resume_text": "Financial Controller, 15 years. Managed accounting. No FP&A leadership.",
            "relevance_score": 40
        },

        # Example: Talent Acquisition Manager
        {
            "jd_text": "Talent Acquisition Manager. Skills: Recruitment Strategy, Team Leadership, Employer Branding, ATS (Workday/Greenhouse), Sourcing, Interviewing, HR Analytics. Experience: 5+ years.",
            "resume_text": "Talent Acquisition Manager, 6 years. Developed and executed recruitment strategies. Led a team of recruiters. Built strong employer brand. Proficient in Workday ATS. Expert in sourcing and interviewing. Utilized HR analytics for insights.",
            "relevance_score": 94
        },
        {
            "jd_text": "Talent Acquisition Manager. Skills: Recruitment Strategy, Team Leadership, Employer Branding, ATS (Workday/Greenhouse), Sourcing, Interviewing, HR Analytics. Experience: 5+ years.",
            "resume_text": "Talent Acquisition Specialist, 3 years. Conducted full-cycle recruiting. Some sourcing. No leadership or strategy.",
            "relevance_score": 65
        },
        {
            "jd_text": "Talent Acquisition Manager. Skills: Recruitment Strategy, Team Leadership, Employer Branding, ATS (Workday/Greenhouse), Sourcing, Interviewing, HR Analytics. Experience: 5+ years.",
            "resume_text": "HR Generalist, 8 years. Managed employee relations. No talent acquisition specialization.",
            "relevance_score": 30
        },

        # Example: Principal Game Designer
        {
            "jd_text": "Principal Game Designer. Skills: Game Design, Systems Design, Narrative Design, Level Design, Player Psychology, Prototyping, Team Leadership. Experience: 8+ years.",
            "resume_text": "Principal Game Designer, 10 years. Led game design for multiple AAA titles. Expertise in systems design, narrative design, and level design. Deep understanding of player psychology. Rapid prototyping skills. Provided leadership to design team.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Game Designer. Skills: Game Design, Systems Design, Narrative Design, Level Design, Player Psychology, Prototyping, Team Leadership. Experience: 8+ years.",
            "resume_text": "Game Designer, 4 years. Designed levels. Some systems design. No principal-level leadership or player psychology depth.",
            "relevance_score": 70
        },
        {
            "jd_text": "Principal Game Designer. Skills: Game Design, Systems Design, Narrative Design, Level Design, Player Psychology, Prototyping, Team Leadership. Experience: 8+ years.",
            "resume_text": "Software Engineer, 12 years. Built game engines. No game design role.",
            "relevance_score": 20
        },

        # Example: E-commerce Director
        {
            "jd_text": "E-commerce Director. Skills: E-commerce Strategy, P&L Management, Digital Marketing, Customer Experience, Supply Chain Integration, Team Leadership. Experience: 8+ years.",
            "resume_text": "E-commerce Director, 10 years. Developed and executed global e-commerce strategy. Managed P&L for online channels. Oversaw digital marketing and customer experience. Integrated with supply chain. Led large e-commerce teams.",
            "relevance_score": 97
        },
        {
            "jd_text": "E-commerce Director. Skills: E-commerce Strategy, P&L Management, Digital Marketing, Customer Experience, Supply Chain Integration, Team Leadership. Experience: 8+ years.",
            "resume_text": "E-commerce Manager, 5 years. Managed a platform. Some digital marketing. No director-level strategy or P&L.",
            "relevance_score": 75
        },
        {
            "jd_text": "E-commerce Director. Skills: E-commerce Strategy, P&L Management, Digital Marketing, Customer Experience, Supply Chain Integration, Team Leadership. Experience: 8+ years.",
            "resume_text": "Retail Director, 15 years. Managed brick-and-mortar stores. No e-commerce expertise.",
            "relevance_score": 30
        },

        # Example: Data Governance Manager
        {
            "jd_text": "Data Governance Manager. Skills: Data Governance Frameworks, Data Quality Management, Metadata Management, Data Stewardship Programs, Policy Development, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Data Governance Manager, 7 years. Developed and implemented enterprise data governance frameworks. Led data quality management initiatives. Established metadata management and data stewardship programs. Developed data policies. Provided cross-functional leadership.",
            "relevance_score": 95
        },
        {
            "jd_text": "Data Governance Manager. Skills: Data Governance Frameworks, Data Quality Management, Metadata Management, Data Stewardship Programs, Policy Development, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Data Governance Analyst, 3 years. Supported data quality. Some metadata. No management or policy development.",
            "relevance_score": 60
        },
        {
            "jd_text": "Data Governance Manager. Skills: Data Governance Frameworks, Data Quality Management, Metadata Management, Data Stewardship Programs, Policy Development, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Compliance Manager, 10 years. Focused on regulatory compliance. No data specific governance.",
            "relevance_score": 40
        },

        # Example: Principal Robotics Engineer
        {
            "jd_text": "Principal Robotics Engineer. Skills: Advanced ROS, C++/Python, SLAM, Path Planning, Robot Control, AI for Robotics, Research & Development Leadership. Experience: 8+ years.",
            "resume_text": "Principal Robotics Engineer, 10 years. Led advanced robotics R&D. Expert in ROS, C++, and Python. Developed cutting-edge SLAM and path planning algorithms. Designed complex robot control systems. Applied AI for robotics. Provided technical leadership.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Robotics Engineer. Skills: Advanced ROS, C++/Python, SLAM, Path Planning, Robot Control, AI for Robotics, Research & Development Leadership. Experience: 8+ years.",
            "resume_text": "Senior Robotics Engineer, 5 years. Developed robot applications. Some SLAM. No principal-level research or leadership.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Robotics Engineer. Skills: Advanced ROS, C++/Python, SLAM, Path Planning, Robot Control, AI for Robotics, Research & Development Leadership. Experience: 8+ years.",
            "resume_text": "Software Engineer, 12 years. Strong in C++. No robotics specialization.",
            "relevance_score": 25
        },

        # Example: Director of Technical Account Management
        {
            "jd_text": "Director of Technical Account Management. Skills: Technical Account Leadership, Strategic Client Partnerships, Escalation Management, Team Building, SaaS Solutions, P&L. Experience: 8+ years.",
            "resume_text": "Director of TAM, 10 years. Led technical account management teams. Developed strategic client partnerships. Oversaw complex escalation management. Built and mentored high-performing teams. Drove success for SaaS solutions. Managed departmental P&L.",
            "relevance_score": 97
        },
        {
            "jd_text": "Director of Technical Account Management. Skills: Technical Account Leadership, Strategic Client Partnerships, Escalation Management, Team Building, SaaS Solutions, P&L. Experience: 8+ years.",
            "resume_text": "Senior Technical Account Manager, 5 years. Managed key accounts. Some leadership. No director-level strategy or P&L.",
            "relevance_score": 70
        },
        {
            "jd_text": "Director of Technical Account Management. Skills: Technical Account Leadership, Strategic Client Partnerships, Escalation Management, Team Building, SaaS Solutions, P&L. Experience: 8+ years.",
            "resume_text": "Sales Director, 15 years. Led sales. No technical account management.",
            "relevance_score": 30
        },

        # Example: Principal Biomedical Engineer
        {
            "jd_text": "Principal Biomedical Engineer. Skills: Medical Device Innovation, R&D Leadership, Regulatory Strategy, Clinical Trials, Biocompatibility, Advanced Prototyping. Experience: 8+ years.",
            "resume_text": "Principal Biomedical Engineer, 10 years. Led medical device innovation from concept to market. Directed R&D efforts. Developed regulatory strategies. Oversaw clinical trials. Expertise in biocompatibility and advanced prototyping.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Biomedical Engineer. Skills: Medical Device Innovation, R&D Leadership, Regulatory Strategy, Clinical Trials, Biocompatibility, Advanced Prototyping. Experience: 8+ years.",
            "resume_text": "Senior Biomedical Engineer, 5 years. Designed devices. Some regulatory awareness. No R&D leadership or clinical trials.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Biomedical Engineer. Skills: Medical Device Innovation, R&D Leadership, Regulatory Strategy, Clinical Trials, Biocompatibility, Advanced Prototyping. Experience: 8+ years.",
            "resume_text": "Research Scientist, 12 years. Focused on basic science. No medical device development.",
            "relevance_score": 25
        },

        # Example: Principal Investment Analyst
        {
            "jd_text": "Principal Investment Analyst. Skills: Macroeconomic Analysis, Sector Research, Advanced Valuation, Portfolio Construction, Risk Management, Alternative Investments, CFA Charterholder. Experience: 8+ years.",
            "resume_text": "Principal Investment Analyst, 10 years. Conducted macroeconomic and in-depth sector research. Performed advanced valuations. Led portfolio construction and risk management. Expertise in alternative investments. CFA Charterholder.",
            "relevance_score": 99
        },
        {
            "jd_text": "Principal Investment Analyst. Skills: Macroeconomic Analysis, Sector Research, Advanced Valuation, Portfolio Construction, Risk Management, Alternative Investments, CFA Charterholder. Experience: 8+ years.",
            "resume_text": "Senior Investment Analyst, 5 years. Performed equity research. Some valuation. No principal-level macro or alternative investments.",
            "relevance_score": 70
        },
        {
            "jd_text": "Principal Investment Analyst. Skills: Macroeconomic Analysis, Sector Research, Advanced Valuation, Portfolio Construction, Risk Management, Alternative Investments, CFA Charterholder. Experience: 8+ years.",
            "resume_text": "Financial Advisor, 15 years. Advised individual clients. No institutional investment analysis.",
            "relevance_score": 30
        },

        # Example: Principal Digital Marketing Analyst
        {
            "jd_text": "Principal Digital Marketing Analyst. Skills: Marketing Data Science, Advanced Analytics, Machine Learning (Marketing), Multi-touch Attribution, Experimentation Design, Data Strategy. Experience: 7+ years.",
            "resume_text": "Principal Digital Marketing Analyst, 8 years. Led marketing data science initiatives. Applied advanced analytics and machine learning to marketing problems. Designed and analyzed complex multi-touch attribution models. Led experimentation design. Developed marketing data strategy.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Digital Marketing Analyst. Skills: Marketing Data Science, Advanced Analytics, Machine Learning (Marketing), Multi-touch Attribution, Experimentation Design, Data Strategy. Experience: 7+ years.",
            "resume_text": "Senior Digital Marketing Analyst, 4 years. Performed A/B testing. Some SQL. No marketing data science or ML.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Digital Marketing Analyst. Skills: Marketing Data Science, Advanced Analytics, Machine Learning (Marketing), Multi-touch Attribution, Experimentation Design, Data Strategy. Experience: 7+ years.",
            "resume_text": "Data Scientist, 10 years. Strong in ML. No marketing domain expertise.",
            "relevance_score": 50
        },

        # Example: Principal Supply Chain Analyst
        {
            "jd_text": "Principal Supply Chain Analyst. Skills: Supply Chain Modeling, Optimization (Linear Programming), Network Design, Predictive Analytics, Digital Supply Chain, Strategic Planning. Experience: 7+ years.",
            "resume_text": "Principal Supply Chain Analyst, 8 years. Developed complex supply chain models and optimization solutions using linear programming. Led network design projects. Built advanced predictive analytics. Drove digital supply chain initiatives. Contributed to strategic planning.",
            "relevance_score": 97
        },
        {
            "jd_text": "Principal Supply Chain Analyst. Skills: Supply Chain Modeling, Optimization (Linear Programming), Network Design, Predictive Analytics, Digital Supply Chain, Strategic Planning. Experience: 7+ years.",
            "resume_text": "Senior Supply Chain Analyst, 4 years. Performed inventory forecasting. Some logistics. No advanced modeling or network design.",
            "relevance_score": 70
        },
        {
            "jd_text": "Principal Supply Chain Analyst. Skills: Supply Chain Modeling, Optimization (Linear Programming), Network Design, Predictive Analytics, Digital Supply Chain, Strategic Planning. Experience: 7+ years.",
            "resume_text": "Operations Director, 12 years. Managed operations. No supply chain analytics.",
            "relevance_score": 35
        },

        # Example: Principal Mechanical Engineer
        {
            "jd_text": "Principal Mechanical Engineer. Skills: Advanced Product Design, FEA (Non-linear), CFD, Material Science, Prototyping, Design for X (DFX), Technical Leadership. Experience: 8+ years.",
            "resume_text": "Principal Mechanical Engineer, 10 years. Led advanced product design. Performed non-linear FEA and CFD simulations. Deep expertise in material science. Oversaw complex prototyping. Applied DFX principles. Provided technical leadership to engineering teams.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Mechanical Engineer. Skills: Advanced Product Design, FEA (Non-linear), CFD, Material Science, Prototyping, Design for X (DFX), Technical Leadership. Experience: 8+ years.",
            "resume_text": "Senior Mechanical Engineer, 5 years. Designed components. Some FEA. No principal-level leadership or advanced simulations.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Mechanical Engineer. Skills: Advanced Product Design, FEA (Non-linear), CFD, Material Science, Prototyping, Design for X (DFX), Technical Leadership. Experience: 8+ years.",
            "resume_text": "Manufacturing Director, 15 years. Managed production. No mechanical engineering design.",
            "relevance_score": 25
        },

        # Example: Principal Electrical Engineer
        {
            "jd_text": "Principal Electrical Engineer. Skills: Advanced Circuit Design, Power Electronics, RF Design, Embedded Systems Architecture, Signal Integrity, EMI/EMC, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Principal Electrical Engineer, 10 years. Led advanced circuit design and power electronics. Expertise in RF design. Architected embedded systems. Ensured signal integrity and managed EMI/EMC. Provided technical leadership.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Electrical Engineer. Skills: Advanced Circuit Design, Power Electronics, RF Design, Embedded Systems Architecture, Signal Integrity, EMI/EMC, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Senior Electrical Engineer, 5 years. Designed circuits. Some embedded systems. No principal-level or RF design.",
            "relevance_score": 70
        },
        {
            "jd_text": "Principal Electrical Engineer. Skills: Advanced Circuit Design, Power Electronics, RF Design, Embedded Systems Architecture, Signal Integrity, EMI/EMC, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Software Engineer, 12 years. No electrical engineering.",
            "relevance_score": 20
        },

        # Example: Principal Structural Engineer
        {
            "jd_text": "Principal Structural Engineer. Skills: Complex Structural Analysis, Seismic Design, Bridge/High-rise Design, Advanced Software (SAP2000, ETABS), Peer Review, Project Leadership. Experience: 8+ years.",
            "resume_text": "Principal Structural Engineer, 10 years. Led complex structural analysis for high-rise buildings and bridges. Expertise in seismic design. Proficient in SAP2000 and ETABS. Conducted peer reviews. Provided project leadership.",
            "relevance_score": 97
        },
        {
            "jd_text": "Principal Structural Engineer. Skills: Complex Structural Analysis, Seismic Design, Bridge/High-rise Design, Advanced Software (SAP2000, ETABS), Peer Review, Project Leadership. Experience: 8+ years.",
            "resume_text": "Senior Structural Engineer, 5 years. Designed structures. Some analysis. No principal-level or seismic design.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Structural Engineer. Skills: Complex Structural Analysis, Seismic Design, Bridge/High-rise Design, Advanced Software (SAP2000, ETABS), Peer Review, Project Leadership. Experience: 8+ years.",
            "resume_text": "Civil Engineer, 12 years. Focused on transportation. No structural specialization.",
            "relevance_score": 30
        },

        # Example: Principal Analytical Chemist
        {
            "jd_text": "Principal Analytical Chemist. Skills: Method Development & Validation, Hyphenated Techniques (LC-MS/MS, GC-MS/MS), Impurity Profiling, Regulatory Submissions, Lab Leadership. Experience: 7+ years.",
            "resume_text": "Principal Analytical Chemist, 8 years. Led method development and validation. Expert in LC-MS/MS and GC-MS/MS. Performed impurity profiling. Prepared data for regulatory submissions. Provided leadership to lab team.",
            "relevance_score": 96
        },
        {
            "jd_text": "Principal Analytical Chemist. Skills: Method Development & Validation, Hyphenated Techniques (LC-MS/MS, GC-MS/MS), Impurity Profiling, Regulatory Submissions, Lab Leadership. Experience: 7+ years.",
            "resume_text": "Analytical Chemist, 4 years. Performed routine analysis. Some method development. No hyphenated techniques or leadership.",
            "relevance_score": 70
        },
        {
            "jd_text": "Principal Analytical Chemist. Skills: Method Development & Validation, Hyphenated Techniques (LC-MS/MS, GC-MS/MS), Impurity Profiling, Regulatory Submissions, Lab Leadership. Experience: 7+ years.",
            "resume_text": "Research Chemist, 10 years. Focused on synthesis. No analytical specialization.",
            "relevance_score": 35
        },

        # Example: Principal Clinical Research Associate (CRA)
        {
            "jd_text": "Principal CRA. Skills: Global Clinical Monitoring, Complex Trial Management, Vendor Oversight, ICH-GCP Expert, Risk-Based Monitoring, Mentorship. Experience: 6+ years.",
            "resume_text": "Principal CRA, 7 years. Led global clinical monitoring for complex trials. Provided expert vendor oversight. Deep expertise in ICH-GCP. Implemented risk-based monitoring. Mentored junior CRAs.",
            "relevance_score": 97
        },
        {
            "jd_text": "Principal Clinical Research Associate (CRA). Skills: Global Clinical Monitoring, Complex Trial Management, Vendor Oversight, ICH-GCP Expert, Risk-Based Monitoring, Mentorship. Experience: 6+ years.",
            "resume_text": "Senior CRA, 3 years. Conducted site visits. Some trial management. No global scope or mentorship.",
            "relevance_score": 70
        },
        {
            "jd_text": "Principal Clinical Research Associate (CRA). Skills: Global Clinical Monitoring, Complex Trial Management, Vendor Oversight, ICH-GCP Expert, Risk-Based Monitoring, Mentorship. Experience: 6+ years.",
            "resume_text": "Clinical Project Manager, 10 years. Managed trials. No CRA specific monitoring.",
            "relevance_score": 40
        },

        # Example: Principal University Lecturer
        {
            "jd_text": "Principal Lecturer (University). Skills: Advanced Teaching, Curriculum Leadership, Research Mentorship, Publications, Grant Acquisition, Departmental Service. Experience: 8+ years.",
            "resume_text": "Principal Lecturer, 10 years. Delivered advanced courses. Provided curriculum leadership. Mentored research students. Published extensively. Secured significant grants. Contributed to departmental service.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal University Lecturer. Skills: Advanced Teaching, Curriculum Leadership, Research Mentorship, Publications, Grant Acquisition, Departmental Service. Experience: 8+ years.",
            "resume_text": "University Lecturer, 5 years. Taught courses. Some research. No curriculum leadership or grant acquisition.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal University Lecturer. Skills: Advanced Teaching, Curriculum Leadership, Research Mentorship, Publications, Grant Acquisition, Departmental Service. Experience: 8+ years.",
            "resume_text": "High School Teacher, 15 years. Taught classes. No university-level or research.",
            "relevance_score": 30
        },

        # Example: Senior Financial Auditor
        {
            "jd_text": "Senior Financial Auditor. Skills: Complex Audit Engagements, IFRS/GAAP, Internal Controls Testing, Risk Assessment, Data Analytics for Audit, Client Management. Experience: 5+ years.",
            "resume_text": "Senior Financial Auditor, 6 years. Led complex audit engagements. Ensured IFRS/GAAP compliance. Performed rigorous internal controls testing. Conducted risk assessments. Utilized data analytics for audit efficiency. Managed client relationships.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Financial Auditor. Skills: Complex Audit Engagements, IFRS/GAAP, Internal Controls Testing, Risk Assessment, Data Analytics for Audit, Client Management. Experience: 5+ years.",
            "resume_text": "Financial Auditor, 2 years. Assisted with audits. Some controls testing. No complex engagements or client management.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Financial Auditor. Skills: Complex Audit Engagements, IFRS/GAAP, Internal Controls Testing, Risk Assessment, Data Analytics for Audit, Client Management. Experience: 5+ years.",
            "resume_text": "Accountant, 8 years. Prepared financial statements. No audit specialization.",
            "relevance_score": 35
        },

        # Example: Senior Manufacturing Engineer
        {
            "jd_text": "Senior Manufacturing Engineer. Skills: Process Optimization, Lean Manufacturing, Six Sigma, Automation, Robotics, CAD/CAM, Production Optimization, Quality Systems. Experience: 5+ years.",
            "resume_text": "Senior Manufacturing Engineer, 6 years. Led Lean Six Sigma initiatives for process improvement. Designed and implemented automation and robotics solutions. Proficient in CAD/CAM. Optimized production lines. Developed and maintained quality systems.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Manufacturing Engineer. Skills: Process Optimization, Lean Manufacturing, Six Sigma, Automation, Robotics, CAD/CAM, Production Optimization, Quality Systems. Experience: 5+ years.",
            "resume_text": "Manufacturing Engineer, 3 years. Optimized some processes. Some CAD. No Lean Six Sigma or automation leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Manufacturing Engineer. Skills: Process Optimization, Lean Manufacturing, Six Sigma, Automation, Robotics, CAD/CAM, Production Optimization, Quality Systems. Experience: 5+ years.",
            "resume_text": "Production Manager, 10 years. Managed production teams. No engineering background.",
            "relevance_score": 35
        },

        # Example: Senior Salesforce Administrator
        {
            "jd_text": "Senior Salesforce Administrator. Skills: Salesforce Admin (Advanced), Apex/Visualforce (basic), Lightning Web Components (LWC), Integrations, Data Migration, Security Best Practices. Experience: 4+ years.",
            "resume_text": "Senior Salesforce Administrator, 5 years. Managed complex Salesforce orgs. Wrote basic Apex and LWC components. Led integrations and data migrations. Implemented security best practices. Certified Advanced Admin.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Salesforce Administrator. Skills: Salesforce Admin (Advanced), Apex/Visualforce (basic), Lightning Web Components (LWC), Integrations, Data Migration, Security Best Practices. Experience: 4+ years.",
            "resume_text": "Salesforce Administrator, 2 years. Managed users and reports. No development or advanced features.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Salesforce Administrator. Skills: Salesforce Admin (Advanced), Apex/Visualforce (basic), Lightning Web Components (LWC), Integrations, Data Migration, Security Best Practices. Experience: 4+ years.",
            "resume_text": "CRM Support Specialist, 4 years. Supported CRM users. No Salesforce administration.",
            "relevance_score": 40
        },

        # Example: Senior Executive Assistant
        {
            "jd_text": "Senior Executive Assistant. Skills: Executive Support, Project Management, Event Planning, Board Relations, Confidentiality, Advanced Microsoft Office. Experience: 8+ years.",
            "resume_text": "Senior Executive Assistant, 10 years. Provided high-level support to C-suite executives. Managed special projects and planned corporate events. Coordinated board relations. Handled confidential information with discretion. Expert in Microsoft Office Suite.",
            "relevance_score": 92
        },
        {
            "jd_text": "Senior Executive Assistant. Skills: Executive Support, Project Management, Event Planning, Board Relations, Confidentiality, Advanced Microsoft Office. Experience: 8+ years.",
            "resume_text": "Executive Assistant, 4 years. Managed calendars. Some travel. No project management or board relations.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Executive Assistant. Skills: Executive Support, Project Management, Event Planning, Board Relations, Confidentiality, Advanced Microsoft Office. Experience: 8+ years.",
            "resume_text": "Office Manager, 12 years. Managed office operations. No executive support.",
            "relevance_score": 30
        },

        # Example: Senior Cloud Solutions Architect
        {
            "jd_text": "Senior Cloud Solutions Architect. Skills: Multi-Cloud (AWS, Azure, GCP), Cloud Migration, Cost Optimization, Security Architecture, DevOps Integration, Enterprise Solutions. Experience: 7+ years.",
            "resume_text": "Senior Cloud Solutions Architect, 8 years. Designed and led multi-cloud solutions across AWS, Azure, and GCP. Managed large-scale cloud migrations. Optimized cloud costs and security architecture. Integrated DevOps practices. Delivered enterprise-level solutions.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior Cloud Solutions Architect. Skills: Multi-Cloud (AWS, Azure, GCP), Cloud Migration, Cost Optimization, Security Architecture, DevOps Integration, Enterprise Solutions. Experience: 7+ years.",
            "resume_text": "Cloud Engineer, 4 years. Deployed resources in AWS. Some architecture. No multi-cloud or cost optimization focus.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Cloud Solutions Architect. Skills: Multi-Cloud (AWS, Azure, GCP), Cloud Migration, Cost Optimization, Security Architecture, DevOps Integration, Enterprise Solutions. Experience: 7+ years.",
            "resume_text": "IT Manager, 10 years. Managed IT infrastructure. No cloud architecture.",
            "relevance_score": 40
        },

        # Example: Principal Data Engineer
        {
            "jd_text": "Principal Data Engineer. Skills: Big Data Ecosystems (Hadoop, Spark, Kafka), Data Lake/Warehouse Architecture, Real-time Data Processing, Performance Tuning, Data Governance, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Principal Data Engineer, 10 years. Designed and built large-scale big data ecosystems using Hadoop, Spark, and Kafka. Architected data lakes and warehouses. Implemented real-time data processing. Optimized data pipeline performance. Led data governance initiatives and provided technical leadership.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Data Engineer. Skills: Big Data Ecosystems (Hadoop, Spark, Kafka), Data Lake/Warehouse Architecture, Real-time Data Processing, Performance Tuning, Data Governance, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Senior Data Engineer, 6 years. Built Spark pipelines. Some data warehousing. No principal-level architecture or governance.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Data Engineer. Skills: Big Data Ecosystems (Hadoop, Spark, Kafka), Data Lake/Warehouse Architecture, Real-time Data Processing, Performance Tuning, Data Governance, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Data Architect, 7 years. Designed data models. No hands-on big data engineering.",
            "relevance_score": 50
        },

        # Example: Senior Digital Content Creator
        {
            "jd_text": "Senior Digital Content Creator. Skills: Video Production, Motion Graphics (After Effects), Advanced Graphic Design, Storyboarding, Scriptwriting, Content Strategy. Experience: 4+ years.",
            "resume_text": "Senior Digital Content Creator, 5 years. Led end-to-end video production. Created motion graphics in After Effects. Designed advanced graphics. Developed storyboards and wrote compelling scripts. Contributed to content strategy.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Digital Content Creator. Skills: Video Production, Motion Graphics (After Effects), Advanced Graphic Design, Storyboarding, Scriptwriting, Content Strategy. Experience: 4+ years.",
            "resume_text": "Digital Content Creator, 2 years. Edited videos. Some graphic design. No motion graphics or content strategy leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Digital Content Creator. Skills: Video Production, Motion Graphics (After Effects), Advanced Graphic Design, Storyboarding, Scriptwriting, Content Strategy. Experience: 4+ years.",
            "resume_text": "Marketing Specialist, 7 years. Focused on campaigns. No content creation.",
            "relevance_score": 30
        },

        # Example: Senior Customer Success Manager
        {
            "jd_text": "Senior Customer Success Manager. Skills: Strategic Account Management, Customer Retention, Upselling/Cross-selling, Product Adoption, CSAT/NPS, Team Leadership. Experience: 5+ years.",
            "resume_text": "Senior Customer Success Manager, 6 years. Managed strategic accounts. Drove customer retention and identified upsell/cross-sell opportunities. Increased product adoption. Improved CSAT/NPS scores. Provided leadership to CSM team.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Customer Success Manager. Skills: Strategic Account Management, Customer Retention, Upselling/Cross-selling, Product Adoption, CSAT/NPS, Team Leadership. Experience: 5+ years.",
            "resume_text": "Customer Success Manager, 3 years. Managed smaller accounts. Some retention efforts. No strategic account management or leadership.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Customer Success Manager. Skills: Strategic Account Management, Customer Retention, Upselling/Cross-selling, Product Adoption, CSAT/NPS, Team Leadership. Experience: 5+ years.",
            "resume_text": "Sales Manager, 8 years. Focused on new sales. No customer retention.",
            "relevance_score": 35
        },

        # Example: Senior Regulatory Affairs Specialist
        {
            "jd_text": "Senior Regulatory Affairs Specialist. Skills: Global Regulatory Strategy, FDA/EMA Submissions, Post-Market Surveillance, Clinical Evaluation Reports (CER), Regulatory Intelligence. Experience: 6+ years.",
            "resume_text": "Senior Regulatory Affairs Specialist, 7 years. Developed global regulatory strategies. Prepared and managed FDA and EMA submissions. Oversaw post-market surveillance. Wrote Clinical Evaluation Reports. Conducted regulatory intelligence.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Regulatory Affairs Specialist. Skills: Global Regulatory Strategy, FDA/EMA Submissions, Post-Market Surveillance, Clinical Evaluation Reports (CER), Regulatory Intelligence. Experience: 6+ years.",
            "resume_text": "Regulatory Affairs Specialist, 3 years. Prepared some submissions. No global strategy or post-market surveillance.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Regulatory Affairs Specialist. Skills: Global Regulatory Strategy, FDA/EMA Submissions, Post-Market Surveillance, Clinical Evaluation Reports (CER), Regulatory Intelligence. Experience: 6+ years.",
            "resume_text": "Quality Assurance Manager, 10 years. Managed quality systems. No regulatory affairs.",
            "relevance_score": 40
        },

        # Example: Principal UX Writer
        {
            "jd_text": "Principal UX Writer. Skills: UX Content Strategy, Information Architecture, Content Governance, A/B Testing (Content), User Research, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Principal UX Writer, 7 years. Defined UX content strategy and information architecture. Established content governance. Led A/B testing for content. Conducted user research. Provided leadership to design and product teams.",
            "relevance_score": 97
        },
        {
            "jd_text": "Principal UX Writer. Skills: UX Content Strategy, Information Architecture, Content Governance, A/B Testing (Content), User Research, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Senior UX Writer, 3 years. Wrote microcopy. Some content strategy. No information architecture or leadership.",
            "relevance_score": 70
        },
        {
            "jd_text": "Principal UX Writer. Skills: UX Content Strategy, Information Architecture, Content Governance, A/B Testing (Content), User Research, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Content Strategist, 8 years. Focused on marketing content. No UX specific.",
            "relevance_score": 45
        },

        # Example: Senior Investment Banking Analyst
        {
            "jd_text": "Senior Investment Banking Analyst. Skills: M&A, Capital Markets, Due Diligence, Financial Modeling (LBO, M&A), Pitch Book Creation, Client Presentations. Experience: 3+ years.",
            "resume_text": "Senior Investment Banking Analyst, 4 years. Executed M&A and capital market transactions. Led due diligence processes. Built complex LBO and M&A financial models. Created compelling pitch books and delivered client presentations.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Investment Banking Analyst. Skills: M&A, Capital Markets, Due Diligence, Financial Modeling (LBO, M&A), Pitch Book Creation, Client Presentations. Experience: 3+ years.",
            "resume_text": "Investment Banking Analyst, 1 year. Assisted with modeling. Some pitch book work. No leadership or complex transactions.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Investment Banking Analyst. Skills: M&A, Capital Markets, Due Diligence, Financial Modeling (LBO, M&A), Pitch Book Creation, Client Presentations. Experience: 3+ years.",
            "resume_text": "Financial Analyst, 5 years. Built corporate finance models. No investment banking deal experience.",
            "relevance_score": 40
        },

        # Example: Senior Data Science Intern
        {
            "jd_text": "Senior Data Science Intern. Skills: Python (Scikit-learn, PyTorch), SQL, Cloud Platforms (AWS/GCP), Model Evaluation, Independent Research. Experience: Master's student + 1 internship.",
            "resume_text": "Master's student in Data Science with 1 previous internship. Proficient in Python (Scikit-learn, PyTorch). Strong SQL. Experience with AWS. Performed model evaluation. Conducted independent research projects.",
            "relevance_score": 88
        },
        {
            "jd_text": "Senior Data Science Intern. Skills: Python (Scikit-learn, PyTorch), SQL, Cloud Platforms (AWS/GCP), Model Evaluation, Independent Research. Experience: Master's student + 1 internship.",
            "resume_text": "Undergraduate student. Basic Python. No ML or cloud experience.",
            "relevance_score": 40
        },
        {
            "jd_text": "Senior Data Science Intern. Skills: Python (Scikit-learn, PyTorch), SQL, Cloud Platforms (AWS/GCP), Model Evaluation, Independent Research. Experience: Master's student + 1 internship.",
            "resume_text": "Software Engineering Intern. Strong in Java. No data science focus.",
            "relevance_score": 25
        },

        # Example: Senior Software Engineer (Fullstack)
        {
            "jd_text": "Senior Fullstack Engineer. Skills: React, Node.js, Microservices, PostgreSQL, AWS, GraphQL, CI/CD, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Senior Fullstack Engineer, 6 years. Built scalable web applications with React and Node.js. Designed and implemented microservices. Managed PostgreSQL databases. Deployed on AWS. Expertise in GraphQL and CI/CD. Optimized application performance.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Fullstack Engineer. Skills: React, Node.js, Microservices, PostgreSQL, AWS, GraphQL, CI/CD, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Fullstack Developer, 3 years. Built basic apps. Some React/Node. No microservices or performance optimization.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Fullstack Engineer. Skills: React, Node.js, Microservices, PostgreSQL, AWS, GraphQL, CI/CD, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Frontend Developer, 8 years. Expert in React. No backend or fullstack.",
            "relevance_score": 40
        },

        # Example: Senior Marketing Specialist
        {
            "jd_text": "Senior Marketing Specialist. Skills: Digital Campaign Management, SEO/SEM Strategy, Content Marketing, Email Marketing Automation, Google Analytics (Advanced), CRM Integration. Experience: 4+ years.",
            "resume_text": "Senior Marketing Specialist, 5 years. Led digital campaign management. Developed and executed SEO/SEM strategies. Managed content marketing initiatives. Implemented email marketing automation. Expert in advanced Google Analytics and CRM integration.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Marketing Specialist. Skills: Digital Campaign Management, SEO/SEM Strategy, Content Marketing, Email Marketing Automation, Google Analytics (Advanced), CRM Integration. Experience: 4+ years.",
            "resume_text": "Marketing Specialist, 2 years. Ran social media ads. Some content. No strategic campaign management.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Marketing Specialist. Skills: Digital Campaign Management, SEO/SEM Strategy, Content Marketing, Email Marketing Automation, Google Analytics (Advanced), CRM Integration. Experience: 4+ years.",
            "resume_text": "Sales Manager, 10 years. No marketing expertise.",
            "relevance_score": 20
        },

        # Example: Senior HR Generalist
        {
            "jd_text": "Senior HR Generalist. Skills: Employee Relations, Performance Management, HR Policy Development, Benefits Administration, HRIS (Workday/SAP), Compliance, Training & Development. Experience: 5+ years.",
            "resume_text": "Senior HR Generalist, 6 years. Managed complex employee relations cases. Oversaw performance management cycles. Developed HR policies. Administered benefits. Proficient in Workday HRIS. Ensured compliance. Led training and development programs.",
            "relevance_score": 93
        },
        {
            "jd_text": "Senior HR Generalist. Skills: Employee Relations, Performance Management, HR Policy Development, Benefits Administration, HRIS (Workday/SAP), Compliance, Training & Development. Experience: 5+ years.",
            "resume_text": "HR Generalist, 2 years. Handled onboarding. Some employee relations. No policy development or training leadership.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior HR Generalist. Skills: Employee Relations, Performance Management, HR Policy Development, Benefits Administration, HRIS (Workday/SAP), Compliance, Training & Development. Experience: 5+ years.",
            "resume_text": "Office Manager, 10 years. Managed administrative tasks. No HR specialization.",
            "relevance_score": 30
        },

        # Example: Senior Cloud Engineer
        {
            "jd_text": "Senior Cloud Engineer. Skills: AWS/Azure/GCP, Infrastructure as Code (Terraform, CloudFormation), Containerization (Docker, Kubernetes), Serverless, CI/CD, Cost Optimization. Experience: 4+ years.",
            "resume_text": "Senior Cloud Engineer, 5 years. Designed and implemented cloud infrastructure on AWS. Expert in Terraform and CloudFormation. Managed Docker and Kubernetes. Developed serverless applications. Built CI/CD pipelines. Optimized cloud costs.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Cloud Engineer. Skills: AWS/Azure/GCP, Infrastructure as Code (Terraform, CloudFormation), Containerization (Docker, Kubernetes), Serverless, CI/CD, Cost Optimization. Experience: 4+ years.",
            "resume_text": "Cloud Engineer, 2 years. Deployed basic resources. Some Docker. No advanced IaC or cost optimization.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Cloud Engineer. Skills: AWS/Azure/GCP, Infrastructure as Code (Terraform, CloudFormation), Containerization (Docker, Kubernetes), Serverless, CI/CD, Cost Optimization. Experience: 4+ years.",
            "resume_text": "System Administrator, 8 years. Managed on-premise servers. No cloud deployment.",
            "relevance_score": 30
        },

        # Example: Senior Cybersecurity Engineer
        {
            "jd_text": "Senior Cybersecurity Engineer. Skills: Security Architecture, Penetration Testing, Vulnerability Management, SIEM/SOAR, Cloud Security, Incident Response, Python Scripting. Experience: 5+ years.",
            "resume_text": "Senior Cybersecurity Engineer, 6 years. Designed and implemented security architectures. Conducted penetration tests and managed vulnerabilities. Proficient in SIEM/SOAR. Secured cloud environments. Led incident response. Wrote Python scripts for automation.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Cybersecurity Engineer. Skills: Security Architecture, Penetration Testing, Vulnerability Management, SIEM/SOAR, Cloud Security, Incident Response, Python Scripting. Experience: 5+ years.",
            "resume_text": "Cybersecurity Analyst, 3 years. Monitored alerts. Some vulnerability scanning. No architecture or penetration testing.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Cybersecurity Engineer. Skills: Security Architecture, Penetration Testing, Vulnerability Management, SIEM/SOAR, Cloud Security, Incident Response, Python Scripting. Experience: 5+ years.",
            "resume_text": "Network Engineer, 10 years. Managed firewalls. No deep cybersecurity engineering.",
            "relevance_score": 35
        },

        # Example: Senior Mobile App Developer (Cross-Platform)
        {
            "jd_text": "Senior Mobile App Developer (Cross-Platform). Skills: React Native/Flutter, iOS/Android Development, RESTful APIs, UI/UX, Performance Optimization, State Management. Experience: 4+ years.",
            "resume_text": "Senior Mobile App Developer, 5 years. Built high-performance cross-platform apps with React Native. Expertise in native iOS/Android development. Integrated complex RESTful APIs. Strong in UI/UX and performance optimization. Managed state effectively.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Mobile App Developer (Cross-Platform). Skills: React Native/Flutter, iOS/Android Development, RESTful APIs, UI/UX, Performance Optimization, State Management. Experience: 4+ years.",
            "resume_text": "Mobile App Developer (Native), 3 years. Focused on iOS. Some API integration. No cross-platform.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Mobile App Developer (Cross-Platform). Skills: React Native/Flutter, iOS/Android Development, RESTful APIs, UI/UX, Performance Optimization, State Management. Experience: 4+ years.",
            "resume_text": "Frontend Web Developer, 6 years. Proficient in React. No mobile development.",
            "relevance_score": 40
        },

        # Example: Senior Business Development Manager
        {
            "jd_text": "Senior Business Development Manager. Skills: Strategic Sales, Market Expansion, Partnership Development, Negotiation, CRM (Salesforce), P&L Responsibility. Experience: 6+ years.",
            "resume_text": "Senior Business Development Manager, 7 years. Developed and executed strategic sales plans. Drove market expansion. Forged key partnerships. Expert negotiator. Proficient in Salesforce CRM. Managed P&L for new ventures.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Business Development Manager. Skills: Strategic Sales, Market Expansion, Partnership Development, Negotiation, CRM (Salesforce), P&L Responsibility. Experience: 6+ years.",
            "resume_text": "Business Development Manager, 3 years. Generated leads. Some negotiation. No strategic or P&L.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Business Development Manager. Skills: Strategic Sales, Market Expansion, Partnership Development, Negotiation, CRM (Salesforce), P&L Responsibility. Experience: 6+ years.",
            "resume_text": "Marketing Manager, 8 years. Focused on brand. No sales or business development.",
            "relevance_score": 25
        },

        # Example: Senior Project Coordinator
        {
            "jd_text": "Senior Project Coordinator. Skills: Project Lifecycle Management, Stakeholder Communication, Risk Mitigation, Resource Tracking, Reporting, Jira/Asana. Experience: 4+ years.",
            "resume_text": "Senior Project Coordinator, 5 years. Managed full project lifecycle. Expert in stakeholder communication and risk mitigation. Tracked resources and provided detailed reports. Proficient in Jira and Asana.",
            "relevance_score": 92
        },
        {
            "jd_text": "Senior Project Coordinator. Skills: Project Lifecycle Management, Stakeholder Communication, Risk Mitigation, Resource Tracking, Reporting, Jira/Asana. Experience: 4+ years.",
            "resume_text": "Project Coordinator, 2 years. Assisted with schedules. Some reporting. No full lifecycle or risk mitigation.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Project Coordinator. Skills: Project Lifecycle Management, Stakeholder Communication, Risk Mitigation, Resource Tracking, Reporting, Jira/Asana. Experience: 4+ years.",
            "resume_text": "Administrative Manager, 10 years. Managed office operations. No project coordination.",
            "relevance_score": 30
        },

        # Example: Senior Technical Writer
        {
            "jd_text": "Senior Technical Writer. Skills: Complex Technical Documentation, API Documentation, DITA/XML, Content Strategy, Information Architecture, Cross-functional Collaboration. Experience: 5+ years.",
            "resume_text": "Senior Technical Writer, 6 years. Produced complex technical documentation and API docs. Expert in DITA XML. Developed content strategies and information architecture. Collaborated extensively with engineering teams.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Technical Writer. Skills: Complex Technical Documentation, API Documentation, DITA/XML, Content Strategy, Information Architecture, Cross-functional Collaboration. Experience: 5+ years.",
            "resume_text": "Technical Writer, 2 years. Wrote user guides. Some API docs. No complex DITA or content strategy.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Technical Writer. Skills: Complex Technical Documentation, API Documentation, DITA/XML, Content Strategy, Information Architecture, Cross-functional Collaboration. Experience: 5+ years.",
            "resume_text": "Copywriter, 8 years. Wrote marketing copy. No technical documentation.",
            "relevance_score": 25
        },

        # Example: Senior Sales Representative
        {
            "jd_text": "Senior Sales Representative. Skills: B2B Sales, Lead Generation, CRM (Salesforce), Negotiation, Quota Attainment, Client Relationship Management. Experience: 3+ years.",
            "resume_text": "Senior Sales Representative, 4 years. Consistently exceeded B2B sales quotas. Expert in lead generation. Proficient in Salesforce CRM. Master negotiator. Built strong client relationships.",
            "relevance_score": 93
        },
        {
            "jd_text": "Senior Sales Representative. Skills: B2B Sales, Lead Generation, CRM (Salesforce), Negotiation, Quota Attainment, Client Relationship Management. Experience: 3+ years.",
            "resume_text": "Sales Representative, 1 year. Achieved some sales. No senior-level quota or B2B focus.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Sales Representative. Skills: B2B Sales, Lead Generation, CRM (Salesforce), Negotiation, Quota Attainment, Client Relationship Management. Experience: 3+ years.",
            "resume_text": "Customer Service Manager, 7 years. Managed customer support. No direct sales.",
            "relevance_score": 30
        },

        # Example: Senior Customer Support Specialist
        {
            "jd_text": "Senior Customer Support Specialist. Skills: Advanced Troubleshooting, Escalation Management, Product Expertise, CRM (Zendesk/ServiceNow), Knowledge Base Management. Experience: 4+ years.",
            "resume_text": "Senior Customer Support Specialist, 5 years. Provided advanced troubleshooting for complex product issues. Managed escalated cases. Deep product expertise. Proficient in Zendesk. Maintained knowledge base articles.",
            "relevance_score": 92
        },
        {
            "jd_text": "Senior Customer Support Specialist. Skills: Advanced Troubleshooting, Escalation Management, Product Expertise, CRM (Zendesk/ServiceNow), Knowledge Base Management. Experience: 4+ years.",
            "resume_text": "Customer Support Specialist, 2 years. Handled routine inquiries. No advanced troubleshooting or escalation.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Customer Support Specialist. Skills: Advanced Troubleshooting, Escalation Management, Product Expertise, CRM (Zendesk/ServiceNow), Knowledge Base Management. Experience: 4+ years.",
            "resume_text": "IT Helpdesk, 6 years. Provided basic IT support. No customer support specific role.",
            "relevance_score": 40
        },

        # Example: Senior Operations Coordinator
        {
            "jd_text": "Senior Operations Coordinator. Skills: Process Optimization, Logistics Coordination, Inventory Management (Advanced), Vendor Management, ERP Systems, Reporting. Experience: 4+ years.",
            "resume_text": "Senior Operations Coordinator, 5 years. Led process optimization initiatives. Coordinated complex logistics. Managed advanced inventory systems. Oversaw vendor relationships. Proficient in ERP systems and detailed reporting.",
            "relevance_score": 93
        },
        {
            "jd_text": "Senior Operations Coordinator. Skills: Process Optimization, Logistics Coordination, Inventory Management (Advanced), Vendor Management, ERP Systems, Reporting. Experience: 4+ years.",
            "resume_text": "Operations Coordinator, 2 years. Assisted with daily tasks. Some inventory. No optimization or vendor management.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Operations Coordinator. Skills: Process Optimization, Logistics Coordination, Inventory Management (Advanced), Vendor Management, ERP Systems, Reporting. Experience: 4+ years.",
            "resume_text": "Administrative Assistant, 8 years. Managed office supplies. No operations focus.",
            "relevance_score": 25
        },

        # Example: Senior Supply Chain Manager
        {
            "jd_text": "Senior Supply Chain Manager. Skills: Strategic SCM, Global Logistics, Demand Planning, Supply Chain Analytics, Risk Management, ERP Implementation. Experience: 7+ years.",
            "resume_text": "Senior Supply Chain Manager, 8 years. Developed and executed strategic supply chain plans. Managed global logistics. Led demand planning. Utilized advanced supply chain analytics. Mitigated risks. Oversaw ERP implementation.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Supply Chain Manager. Skills: Strategic SCM, Global Logistics, Demand Planning, Supply Chain Analytics, Risk Management, ERP Implementation. Experience: 7+ years.",
            "resume_text": "Supply Chain Manager, 4 years. Managed inventory. Some logistics. No strategic or global scope.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Supply Chain Manager. Skills: Strategic SCM, Global Logistics, Demand Planning, Supply Chain Analytics, Risk Management, ERP Implementation. Experience: 7+ years.",
            "resume_text": "Operations Director, 12 years. Managed production. No supply chain specialization.",
            "relevance_score": 35
        },

        # Example: Senior Mechanical Design Engineer
        {
            "jd_text": "Senior Mechanical Design Engineer. Skills: CAD (SolidWorks/CATIA), FEA, DFM/DFA, Thermal Analysis, Product Development Lifecycle, Project Leadership. Experience: 5+ years.",
            "resume_text": "Senior Mechanical Design Engineer, 6 years. Designed complex products using SolidWorks and CATIA. Performed FEA and thermal analysis. Applied DFM/DFA principles. Managed product development lifecycle. Provided project leadership.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Mechanical Design Engineer. Skills: CAD (SolidWorks/CATIA), FEA, DFM/DFA, Thermal Analysis, Product Development Lifecycle, Project Leadership. Experience: 5+ years.",
            "resume_text": "Mechanical Design Engineer, 3 years. Designed components. Some CAD. No senior-level leadership or DFM/DFA.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Mechanical Design Engineer. Skills: CAD (SolidWorks/CATIA), FEA, DFM/DFA, Thermal Analysis, Product Development Lifecycle, Project Leadership. Experience: 5+ years.",
            "resume_text": "Manufacturing Engineer, 8 years. Focused on production. No design engineering.",
            "relevance_score": 30
        },

        # Example: Senior Electrical Design Engineer
        {
            "jd_text": "Senior Electrical Design Engineer. Skills: Analog/Digital Circuit Design, PCB Layout (Altium/Cadence), Power Electronics, Embedded Systems, Signal Integrity, EMI/EMC. Experience: 5+ years.",
            "resume_text": "Senior Electrical Design Engineer, 6 years. Designed advanced analog and digital circuits. Expert in PCB layout using Altium and Cadence. Developed power electronics. Architected embedded systems. Ensured signal integrity and managed EMI/EMC.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Electrical Design Engineer. Skills: Analog/Digital Circuit Design, PCB Layout (Altium/Cadence), Power Electronics, Embedded Systems, Signal Integrity, EMI/EMC. Experience: 5+ years.",
            "resume_text": "Electrical Design Engineer, 2 years. Designed basic circuits. Some PCB. No senior-level or advanced topics.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Electrical Design Engineer. Skills: Analog/Digital Circuit Design, PCB Layout (Altium/Cadence), Power Electronics, Embedded Systems, Signal Integrity, EMI/EMC. Experience: 5+ years.",
            "resume_text": "Software Engineer, 10 years. No electrical engineering.",
            "relevance_score": 20
        },

        # Example: Senior Structural Engineer
        {
            "jd_text": "Senior Structural Engineer. Skills: Structural Analysis (Advanced), Seismic Design, Concrete/Steel Design, ETABS/SAP2000, Building Codes, Project Coordination. Experience: 5+ years.",
            "resume_text": "Senior Structural Engineer, 6 years. Performed advanced structural analysis. Designed for seismic loads. Expertise in concrete and steel design. Proficient in ETABS and SAP2000. Ensured compliance with building codes. Coordinated complex projects.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Structural Engineer. Skills: Structural Analysis (Advanced), Seismic Design, Concrete/Steel Design, ETABS/SAP2000, Building Codes, Project Coordination. Experience: 5+ years.",
            "resume_text": "Structural Engineer, 3 years. Performed basic analysis. Some design. No advanced seismic or project coordination.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Structural Engineer. Skills: Structural Analysis (Advanced), Seismic Design, Concrete/Steel Design, ETABS/SAP2000, Building Codes, Project Coordination. Experience: 5+ years.",
            "resume_text": "Civil Engineer, 8 years. Focused on transportation. No structural specialization.",
            "relevance_score": 30
        },

        # Example: Senior Analytical Chemist
        {
            "jd_text": "Senior Analytical Chemist. Skills: Advanced Method Development, LC-MS/MS, GC-MS/MS, NMR, Impurity Analysis, Regulatory Compliance, Lab Management. Experience: 5+ years.",
            "resume_text": "Senior Analytical Chemist, 6 years. Led advanced method development. Expert in LC-MS/MS, GC-MS/MS, and NMR. Performed complex impurity analysis. Ensured regulatory compliance. Managed lab operations and mentored junior chemists.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Analytical Chemist. Skills: Advanced Method Development, LC-MS/MS, GC-MS/MS, NMR, Impurity Analysis, Regulatory Compliance, Lab Management. Experience: 5+ years.",
            "resume_text": "Analytical Chemist, 2 years. Performed routine tests. Some method development. No advanced techniques or management.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Analytical Chemist. Skills: Advanced Method Development, LC-MS/MS, GC-MS/MS, NMR, Impurity Analysis, Regulatory Compliance, Lab Management. Experience: 5+ years.",
            "resume_text": "Research Chemist, 10 years. Focused on synthesis. No analytical specialization.",
            "relevance_score": 35
        },

        # Example: Senior Clinical Biologist
        {
            "jd_text": "Senior Clinical Biologist. Skills: Molecular Diagnostics, Cell-based Assays, Flow Cytometry, qPCR, Data Analysis (R/Python), Clinical Validation. Experience: 5+ years.",
            "resume_text": "Senior Clinical Biologist, 6 years. Developed molecular diagnostics. Designed and executed cell-based assays. Expert in flow cytometry and qPCR. Performed complex data analysis in R/Python. Led clinical validation studies.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Clinical Biologist. Skills: Molecular Diagnostics, Cell-based Assays, Flow Cytometry, qPCR, Data Analysis (R/Python), Clinical Validation. Experience: 5+ years.",
            "resume_text": "Clinical Biologist, 2 years. Performed basic assays. Some data analysis. No molecular diagnostics or clinical validation.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Clinical Biologist. Skills: Molecular Diagnostics, Cell-based Assays, Flow Cytometry, qPCR, Data Analysis (R/Python), Clinical Validation. Experience: 5+ years.",
            "resume_text": "Medical Technologist, 8 years. Performed routine lab tests. No research or advanced biology.",
            "relevance_score": 30
        },

        # Example: Senior Registered Nurse (Specialty)
        {
            "jd_text": "Senior Registered Nurse (ICU). Skills: Critical Care, Advanced Life Support (ACLS, PALS), Ventilator Management, IV Therapy, Electronic Health Records (Epic/Cerner), Team Leadership. Experience: 5+ years.",
            "resume_text": "Senior ICU Nurse, 6 years. Provided critical care to complex patients. Certified in ACLS and PALS. Managed ventilators and advanced IV therapy. Expert in Epic EHR. Provided leadership and mentorship to nursing staff.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Registered Nurse (ICU). Skills: Critical Care, Advanced Life Support (ACLS, PALS), Ventilator Management, IV Therapy, Electronic Health Records (Epic/Cerner), Team Leadership. Experience: 5+ years.",
            "resume_text": "Registered Nurse (Med-Surg), 3 years. Provided general patient care. Some IV therapy. No critical care or ventilator management.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Registered Nurse (ICU). Skills: Critical Care, Advanced Life Support (ACLS, PALS), Ventilator Management, IV Therapy, Electronic Health Records (Epic/Cerner), Team Leadership. Experience: 5+ years.",
            "resume_text": "Paramedic, 10 years. Provided pre-hospital care. No hospital RN experience.",
            "relevance_score": 40
        },

        # Example: Senior High School Teacher (Science)
        {
            "jd_text": "Senior High School Science Teacher. Skills: Lesson Planning, Curriculum Development, Classroom Management, Physics/Chemistry/Biology, Student Engagement, Assessment. Experience: 5+ years.",
            "resume_text": "Senior High School Science Teacher, 6 years. Developed engaging lesson plans and contributed to curriculum development. Expert in classroom management. Taught Physics, Chemistry, and Biology. Fostered student engagement. Designed effective assessments.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior High School Science Teacher. Skills: Lesson Planning, Curriculum Development, Classroom Management, Physics/Chemistry/Biology, Student Engagement, Assessment. Experience: 5+ years.",
            "resume_text": "High School Math Teacher, 3 years. Strong in teaching. No science specialization.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior High School Science Teacher. Skills: Lesson Planning, Curriculum Development, Classroom Management, Physics/Chemistry/Biology, Student Engagement, Assessment. Experience: 5+ years.",
            "resume_text": "University Lecturer (Chemistry), 10 years. Taught college chemistry. No high school classroom management.",
            "relevance_score": 30
        },

        # Example: Junior Data Scientist
        {
            "jd_text": "Junior Data Scientist. Skills: Python (Pandas, Scikit-learn), SQL, Data Cleaning, Exploratory Data Analysis, Basic ML Models, Data Visualization. Experience: 0-2 years.",
            "resume_text": "Recent Master's graduate in Data Science. Proficient in Python (Pandas, Scikit-learn) and SQL. Strong in data cleaning and EDA. Built basic ML models. Created data visualizations. Completed multiple data science projects.",
            "relevance_score": 88
        },
        {
            "jd_text": "Junior Data Scientist. Skills: Python (Pandas, Scikit-learn), SQL, Data Cleaning, Exploratory Data Analysis, Basic ML Models, Data Visualization. Experience: 0-2 years.",
            "resume_text": "Data Analyst, 3 years. Strong in SQL and Excel. Some Python. No ML model building.",
            "relevance_score": 60
        },
        {
            "jd_text": "Junior Data Scientist. Skills: Python (Pandas, Scikit-learn), SQL, Data Cleaning, Exploratory Data Analysis, Basic ML Models, Data Visualization. Experience: 0-2 years.",
            "resume_text": "Software Engineer, 5 years. Strong in Java. No data science focus.",
            "relevance_score": 20
        },

        # Example: Junior Software Engineer (Backend)
        {
            "jd_text": "Junior Backend Software Engineer. Skills: Java/Python, REST APIs, SQL, Unit Testing, Git, Basic Cloud (AWS/Azure). Experience: 0-2 years.",
            "resume_text": "Recent Computer Science graduate. Proficient in Java. Built REST APIs for academic projects. Strong SQL skills. Wrote unit tests. Familiar with Git and basic AWS concepts.",
            "relevance_score": 85
        },
        {
            "jd_text": "Junior Backend Software Engineer. Skills: Java/Python, REST APIs, SQL, Unit Testing, Git, Basic Cloud (AWS/Azure). Experience: 0-2 years.",
            "resume_text": "Frontend Developer, 3 years. Expert in React. No backend experience.",
            "relevance_score": 15
        },
        {
            "jd_text": "Junior Backend Software Engineer. Skills: Java/Python, REST APIs, SQL, Unit Testing, Git, Basic Cloud (AWS/Azure). Experience: 0-2 years.",
            "resume_text": "QA Tester, 4 years. Automated tests. No development role.",
            "relevance_score": 30
        },

        # Example: Junior Digital Marketing Specialist
        {
            "jd_text": "Junior Digital Marketing Specialist. Skills: Social Media Management, Content Creation, SEO Basics, Google Analytics, Email Marketing. Experience: 0-2 years.",
            "resume_text": "Recent Marketing graduate. Managed social media accounts. Created engaging content. Basic understanding of SEO. Familiar with Google Analytics and email marketing platforms. Internship experience.",
            "relevance_score": 80
        },
        {
            "jd_text": "Junior Digital Marketing Specialist. Skills: Social Media Management, Content Creation, SEO Basics, Google Analytics, Email Marketing. Experience: 0-2 years.",
            "resume_text": "Sales Assistant, 1 year. Good communication. No marketing skills.",
            "relevance_score": 5
        },
        {
            "jd_text": "Junior Digital Marketing Specialist. Skills: Social Media Management, Content Creation, SEO Basics, Google Analytics, Email Marketing. Experience: 0-2 years.",
            "resume_text": "Graphic Designer, 2 years. Expert in design tools. No marketing strategy.",
            "relevance_score": 30
        },

        # Example: Junior Financial Accountant
        {
            "jd_text": "Junior Financial Accountant. Skills: Journal Entries, Reconciliations, Accounts Payable/Receivable, Excel, Basic GAAP. Experience: 0-2 years.",
            "resume_text": "Recent Accounting graduate. Proficient in journal entries and reconciliations. Handled accounts payable and receivable. Strong Excel skills. Basic understanding of GAAP principles. Internship experience.",
            "relevance_score": 85
        },
        {
            "jd_text": "Junior Financial Accountant. Skills: Journal Entries, Reconciliations, Accounts Payable/Receivable, Excel, Basic GAAP. Experience: 0-2 years.",
            "resume_text": "Bookkeeper, 5 years. Managed basic bookkeeping. No formal accounting degree.",
            "relevance_score": 60
        },
        {
            "jd_text": "Junior Financial Accountant. Skills: Journal Entries, Reconciliations, Accounts Payable/Receivable, Excel, Basic GAAP. Experience: 0-2 years.",
            "resume_text": "Financial Analyst, 3 years. Built financial models. No accounting specialization.",
            "relevance_score": 30
        },

        # Example: Junior HR Generalist
        {
            "jd_text": "Junior HR Generalist. Skills: Employee Onboarding, HRIS Data Entry, HR Policies, Recruitment Support, Communication, Employee Records. Experience: 0-2 years.",
            "resume_text": "Recent HR graduate. Assisted with employee onboarding. Proficient in HRIS data entry. Familiar with HR policies. Provided recruitment support. Strong communication skills. Managed employee records.",
            "relevance_score": 88
        },
        {
            "jd_text": "Junior HR Generalist. Skills: Employee Onboarding, HRIS Data Entry, HR Policies, Recruitment Support, Communication, Employee Records. Experience: 0-2 years.",
            "resume_text": "Administrative Assistant, 4 years. Managed office tasks. No HR specific experience.",
            "relevance_score": 20
        },
        {
            "jd_text": "Junior HR Generalist. Skills: Employee Onboarding, HRIS Data Entry, HR Policies, Recruitment Support, Communication, Employee Records. Experience: 0-2 years.",
            "resume_text": "Recruitment Coordinator, 1 year. Focused on sourcing. No generalist duties.",
            "relevance_score": 60
        },

        # Example: Junior Cloud Engineer
        {
            "jd_text": "Junior Cloud Engineer. Skills: AWS/Azure Basics, Linux, Scripting (Bash/Python), Networking Fundamentals, Cloud Monitoring, Basic IaC (Terraform). Experience: 0-2 years.",
            "resume_text": "Recent IT graduate. Strong understanding of AWS basics. Proficient in Linux and Bash scripting. Solid networking fundamentals. Familiar with cloud monitoring tools. Completed basic Terraform projects.",
            "relevance_score": 85
        },
        {
            "jd_text": "Junior Cloud Engineer. Skills: AWS/Azure Basics, Linux, Scripting (Bash/Python), Networking Fundamentals, Cloud Monitoring, Basic IaC (Terraform). Experience: 0-2 years.",
            "resume_text": "System Administrator, 5 years. Managed on-premise servers. No cloud experience.",
            "relevance_score": 20
        },
        {
            "jd_text": "Junior Cloud Engineer. Skills: AWS/Azure Basics, Linux, Scripting (Bash/Python), Networking Fundamentals, Cloud Monitoring, Basic IaC (Terraform). Experience: 0-2 years.",
            "resume_text": "DevOps Intern. Some CI/CD. No deep cloud infrastructure.",
            "relevance_score": 60
        },

        # Example: Junior Cybersecurity Analyst
        {
            "jd_text": "Junior Cybersecurity Analyst. Skills: Network Security, Incident Response Basics, Vulnerability Scanning, Security Tools, Linux, SIEM Basics. Experience: 0-2 years.",
            "resume_text": "Recent Cybersecurity graduate. Strong in network security fundamentals. Familiar with incident response basics and vulnerability scanning. Used various security tools. Proficient in Linux. Basic SIEM knowledge.",
            "relevance_score": 88
        },
        {
            "jd_text": "Junior Cybersecurity Analyst. Skills: Network Security, Incident Response Basics, Vulnerability Scanning, Security Tools, Linux, SIEM Basics. Experience: 0-2 years.",
            "resume_text": "IT Helpdesk, 3 years. Resolved user issues. Some security awareness. No specific cybersecurity analysis.",
            "relevance_score": 30
        },
        {
            "jd_text": "Junior Cybersecurity Analyst. Skills: Network Security, Incident Response Basics, Vulnerability Scanning, Security Tools, Linux, SIEM Basics. Experience: 0-2 years.",
            "resume_text": "Software Developer, 4 years. Built secure code. No security operations.",
            "relevance_score": 50
        },

        # Example: Junior Mobile App Developer
        {
            "jd_text": "Junior Mobile App Developer. Skills: iOS/Android SDK, Swift/Kotlin, RESTful APIs, UI/UX Principles, Debugging, Version Control (Git). Experience: 0-2 years.",
            "resume_text": "Recent Computer Science graduate. Developed mobile apps using iOS SDK and Swift for personal projects. Integrated RESTful APIs. Good understanding of UI/UX principles. Strong debugging skills. Familiar with Git.",
            "relevance_score": 85
        },
        {
            "jd_text": "Junior Mobile App Developer. Skills: iOS/Android SDK, Swift/Kotlin, RESTful APIs, UI/UX Principles, Debugging, Version Control (Git). Experience: 0-2 years.",
            "resume_text": "Web Developer, 3 years. Built responsive websites. No native mobile development.",
            "relevance_score": 20
        },
        {
            "jd_text": "Junior Mobile App Developer. Skills: iOS/Android SDK, Swift/Kotlin, RESTful APIs, UI/UX Principles, Debugging, Version Control (Git). Experience: 0-2 years.",
            "resume_text": "Game Developer, 2 years. Focused on PC games. No mobile platform.",
            "relevance_score": 40
        },

        # Example: Junior Project Manager
        {
            "jd_text": "Junior Project Manager. Skills: Project Coordination, Scheduling, Communication, Stakeholder Support, Risk Identification, Microsoft Office, Agile Basics. Experience: 0-2 years.",
            "resume_text": "Recent Business graduate. Assisted with project coordination and scheduling in internships. Strong communication skills. Supported stakeholders. Identified potential risks. Proficient in Microsoft Office. Basic understanding of Agile.",
            "relevance_score": 80
        },
        {
            "jd_text": "Junior Project Manager. Skills: Project Coordination, Scheduling, Communication, Stakeholder Support, Risk Identification, Microsoft Office, Agile Basics. Experience: 0-2 years.",
            "resume_text": "Administrative Assistant, 5 years. Organized office tasks. No project management.",
            "relevance_score": 30
        },
        {
            "jd_text": "Junior Project Manager. Skills: Project Coordination, Scheduling, Communication, Stakeholder Support, Risk Identification, Microsoft Office, Agile Basics. Experience: 0-2 years.",
            "resume_text": "Team Lead (non-project), 4 years. Led a small team. No formal project management.",
            "relevance_score": 50
        },

        # Example: Junior Technical Writer
        {
            "jd_text": "Junior Technical Writer. Skills: Technical Documentation, User Manuals, Editing, Research, Content Management Systems Basics, Communication. Experience: 0-2 years.",
            "resume_text": "Recent Communications graduate. Wrote technical documentation for academic projects. Created user manuals. Strong editing and research skills. Familiar with basic CMS. Excellent communication.",
            "relevance_score": 85
        },
        {
            "jd_text": "Junior Technical Writer. Skills: Technical Documentation, User Manuals, Editing, Research, Content Management Systems Basics, Communication. Experience: 0-2 years.",
            "resume_text": "Copywriter, 3 years. Wrote marketing copy. No technical focus.",
            "relevance_score": 40
        },
        {
            "jd_text": "Junior Technical Writer. Skills: Technical Documentation, User Manuals, Editing, Research, Content Management Systems Basics, Communication. Experience: 0-2 years.",
            "resume_text": "Software Developer, 5 years. Wrote code comments. No technical writing role.",
            "relevance_score": 20
        },

        # Example: Junior Sales Representative
        {
            "jd_text": "Junior Sales Representative. Skills: Sales, Lead Generation, Cold Calling, CRM Basics, Communication, Product Knowledge. Experience: 0-1 years.",
            "resume_text": "Recent Business graduate. Internship in sales. Generated leads. Practiced cold calling. Familiar with CRM basics. Good communication and quick learner of product knowledge.",
            "relevance_score": 80
        },
        {
            "jd_text": "Junior Sales Representative. Skills: Sales, Lead Generation, Cold Calling, CRM Basics, Communication, Product Knowledge. Experience: 0-1 years.",
            "resume_text": "Customer Service Rep, 2 years. Handled inquiries. No sales experience.",
            "relevance_score": 10
        },
        {
            "jd_text": "Junior Sales Representative. Skills: Sales, Lead Generation, Cold Calling, CRM Basics, Communication, Product Knowledge. Experience: 0-1 years.",
            "resume_text": "Marketing Intern. Some social media. No sales focus.",
            "relevance_score": 5
        },

        # Example: Junior Customer Support Specialist
        {
            "jd_text": "Junior Customer Support Specialist. Skills: Customer Service, Troubleshooting Basics, CRM (Zendesk), Communication, Problem Solving. Experience: 0-1 years.",
            "resume_text": "Recent graduate. Strong customer service skills. Basic troubleshooting ability. Familiar with Zendesk CRM. Excellent communication and problem-solving skills. Internship experience.",
            "relevance_score": 85
        },
        {
            "jd_text": "Junior Customer Support Specialist. Skills: Customer Service, Troubleshooting Basics, CRM (Zendesk), Communication, Problem Solving. Experience: 0-1 years.",
            "resume_text": "Retail Associate, 3 years. Assisted customers. No technical troubleshooting.",
            "relevance_score": 30
        },
        {
            "jd_text": "Junior Customer Support Specialist. Skills: Customer Service, Troubleshooting Basics, CRM (Zendesk), Communication, Problem Solving. Experience: 0-1 years.",
            "resume_text": "IT Helpdesk Intern. Some technical support. No direct customer service.",
            "relevance_score": 60
        },

        # Example: Junior Operations Coordinator
        {
            "jd_text": "Junior Operations Coordinator. Skills: Logistics Support, Inventory Tracking, Data Entry, Excel, Communication. Experience: 0-2 years.",
            "resume_text": "Recent Supply Chain graduate. Assisted with logistics support and inventory tracking in internships. Proficient in data entry and Excel. Strong communication skills. Eager to learn operations.",
            "relevance_score": 80
        },
        {
            "jd_text": "Junior Operations Coordinator. Skills: Logistics Support, Inventory Tracking, Data Entry, Excel, Communication. Experience: 0-2 years.",
            "resume_text": "Administrative Assistant, 4 years. Managed office tasks. No operations focus.",
            "relevance_score": 25
        },
        {
            "jd_text": "Junior Operations Coordinator. Skills: Logistics Support, Inventory Tracking, Data Entry, Excel, Communication. Experience: 0-2 years.",
            "resume_text": "Warehouse Assistant, 3 years. Handled shipping/receiving. No coordination role.",
            "relevance_score": 50
        },
        {
            "jd_text": "Senior Product Manager (Tech). Skills: Product Strategy, Technical Roadmaps, Agile/Scrum, SaaS, API Products, User Experience, Data-Driven Decisions. Experience: 6+ years.",
            "resume_text": "Senior Product Manager with 8 years. Defined and executed product strategies for SaaS and API products. Developed technical roadmaps. Led multiple Agile/Scrum teams. Strong in user experience and data-driven decision-making.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Product Manager (Tech). Skills: Product Strategy, Technical Roadmaps, Agile/Scrum, SaaS, API Products, User Experience, Data-Driven Decisions. Experience: 6+ years.",
            "resume_text": "Product Manager, 4 years. Focused on consumer products. Some Agile. No SaaS or API product experience.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Product Manager (Tech). Skills: Product Strategy, Technical Roadmaps, Agile/Scrum, SaaS, API Products, User Experience, Data-Driven Decisions. Experience: 6+ years.",
            "resume_text": "Software Engineer, 10 years. Built complex systems. No product management role.",
            "relevance_score": 30
        },

        # Example: AI/ML Research Engineer
        {
            "jd_text": "AI/ML Research Engineer. Skills: Deep Learning, Computer Vision, NLP, Python, PyTorch/TensorFlow, Research Publications, Algorithm Optimization. Experience: PhD + 2 years research.",
            "resume_text": "AI/ML Research Engineer, PhD + 3 years research. Conducted cutting-edge research in Computer Vision and NLP. Implemented deep learning models in PyTorch. Published in top conferences. Optimized algorithms for performance.",
            "relevance_score": 97
        },
        {
            "jd_text": "AI/ML Research Engineer. Skills: Deep Learning, Computer Vision, NLP, Python, PyTorch/TensorFlow, Research Publications, Algorithm Optimization. Experience: PhD + 2 years research.",
            "resume_text": "Machine Learning Engineer, 4 years. Deployed ML models. Some deep learning. No research or publications.",
            "relevance_score": 70
        },
        {
            "jd_text": "AI/ML Research Engineer. Skills: Deep Learning, Computer Vision, NLP, Python, PyTorch/TensorFlow, Research Publications, Algorithm Optimization. Experience: PhD + 2 years research.",
            "resume_text": "Data Scientist, 5 years. Focused on predictive modeling. No deep learning research or computer vision.",
            "relevance_score": 40
        },

        # Example: Cloud DevOps Architect
        {
            "jd_text": "Cloud DevOps Architect. Skills: AWS/Azure/GCP, Kubernetes, Microservices Architecture, Infrastructure as Code (Terraform, Ansible), CI/CD Strategy, SRE Principles. Experience: 8+ years.",
            "resume_text": "Cloud DevOps Architect, 9 years. Designed and implemented cloud-native architectures on AWS. Expertise in Kubernetes and microservices. Led IaC initiatives with Terraform/Ansible. Developed CI/CD strategies. Applied SRE principles for high availability.",
            "relevance_score": 98
        },
        {
            "jd_text": "Cloud DevOps Architect. Skills: AWS/Azure/GCP, Kubernetes, Microservices Architecture, Infrastructure as Code (Terraform, Ansible), CI/CD Strategy, SRE Principles. Experience: 8+ years.",
            "resume_text": "Senior DevOps Engineer, 5 years. Built CI/CD pipelines. Some Kubernetes. No architecture design or SRE leadership.",
            "relevance_score": 75
        },
        {
            "jd_text": "Cloud DevOps Architect. Skills: AWS/Azure/GCP, Kubernetes, Microservices Architecture, Infrastructure as Code (Terraform, Ansible), CI/CD Strategy, SRE Principles. Experience: 8+ years.",
            "resume_text": "Software Architect, 10 years. Designed software systems. No specific DevOps or cloud infrastructure.",
            "relevance_score": 45
        },

        # Example: Senior Data Architect
        {
            "jd_text": "Senior Data Architect. Skills: Data Lakehouse, Data Mesh, Snowflake/Databricks, Azure Data Factory, Data Governance, Master Data Management (MDM), Big Data Strategy. Experience: 8+ years.",
            "resume_text": "Senior Data Architect, 9 years. Designed and implemented data lakehouse and data mesh architectures. Expertise in Snowflake and Databricks. Led Azure Data Factory implementations. Developed data governance and MDM strategies. Defined big data strategy.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior Data Architect. Skills: Data Lakehouse, Data Mesh, Snowflake/Databricks, Azure Data Factory, Data Governance, Master Data Management (MDM), Big Data Strategy. Experience: 8+ years.",
            "resume_text": "Data Engineer, 6 years. Built data pipelines. Some data warehousing. No architecture or data mesh concepts.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Data Architect. Skills: Data Lakehouse, Data Mesh, Snowflake/Databricks, Azure Data Factory, Data Governance, Master Data Management (MDM), Big Data Strategy. Experience: 8+ years.",
            "resume_text": "Business Intelligence Manager, 10 years. Led BI teams. No data architecture or big data strategy.",
            "relevance_score": 40
        },

        # Example: Principal Software Engineer (Frontend)
        {
            "jd_text": "Principal Frontend Engineer. Skills: React, Next.js, Performance Optimization, Web Accessibility, Design Systems, Micro-frontends, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Principal Frontend Engineer, 10 years. Led development of complex web applications with React/Next.js. Optimized performance and ensured web accessibility. Designed and implemented design systems and micro-frontends. Provided technical leadership and mentorship.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Frontend Engineer. Skills: React, Next.js, Performance Optimization, Web Accessibility, Design Systems, Micro-frontends, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Senior Frontend Engineer, 6 years. Built React apps. Some performance optimization. No principal-level leadership or micro-frontends.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Frontend Engineer. Skills: React, Next.js, Performance Optimization, Web Accessibility, Design Systems, Micro-frontends, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Backend Engineer, 12 years. Strong in Java. No frontend experience.",
            "relevance_score": 25
        },

        # Example: Senior Cybersecurity Analyst (SOC)
        {
            "jd_text": "Senior Cybersecurity Analyst (SOC). Skills: SIEM (Splunk/Sentinel), Incident Response, Threat Hunting, Forensics, Malware Analysis, Scripting (Python/PowerShell). Experience: 5+ years.",
            "resume_text": "Senior SOC Analyst, 6 years. Monitored and analyzed security events in Splunk. Led incident response and threat hunting operations. Performed digital forensics and malware analysis. Wrote Python/PowerShell scripts for automation.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Cybersecurity Analyst (SOC). Skills: SIEM (Splunk/Sentinel), Incident Response, Threat Hunting, Forensics, Malware Analysis, Scripting (Python/PowerShell). Experience: 5+ years.",
            "resume_text": "Cybersecurity Analyst, 3 years. Monitored basic alerts. Some incident response. No threat hunting or forensics.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Cybersecurity Analyst (SOC). Skills: SIEM (Splunk/Sentinel), Incident Response, Threat Hunting, Forensics, Malware Analysis, Scripting (Python/PowerShell). Experience: 5+ years.",
            "resume_text": "Network Engineer, 8 years. Managed network security devices. No SOC operations.",
            "relevance_score": 35
        },

        # Example: Head of Product
        {
            "jd_text": "Head of Product. Skills: Product Vision, Portfolio Management, Market Leadership, P&L Responsibility, Cross-functional Leadership, Strategic Partnerships. Experience: 10+ years.",
            "resume_text": "Head of Product, 12 years. Defined and executed product vision across multiple product lines. Managed product portfolio. Established market leadership. Held P&L responsibility. Led large cross-functional teams. Forged strategic partnerships.",
            "relevance_score": 98
        },
        {
            "jd_text": "Head of Product. Skills: Product Vision, Portfolio Management, Market Leadership, P&L Responsibility, Cross-functional Leadership, Strategic Partnerships. Experience: 10+ years.",
            "resume_text": "Senior Product Manager, 7 years. Managed a single product. Some strategic input. No portfolio or P&L responsibility.",
            "relevance_score": 70
        },
        {
            "jd_text": "Head of Product. Skills: Product Vision, Portfolio Management, Market Leadership, P&L Responsibility, Cross-functional Leadership, Strategic Partnerships. Experience: 10+ years.",
            "resume_text": "Director of Engineering, 15 years. Led engineering teams. No product management focus.",
            "relevance_score": 40
        },

        # Example: Senior Marketing Manager (Performance)
        {
            "jd_text": "Senior Marketing Manager (Performance). Skills: Performance Marketing, Paid Search (Google Ads), Paid Social (Facebook Ads), Attribution Modeling, A/B Testing, Analytics. Experience: 5+ years.",
            "resume_text": "Senior Performance Marketing Manager, 6 years. Led performance marketing strategies. Managed large-scale Google Ads and Facebook Ads campaigns. Developed attribution models. Conducted rigorous A/B testing. Expert in marketing analytics.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Marketing Manager (Performance). Skills: Performance Marketing, Paid Search (Google Ads), Paid Social (Facebook Ads), Attribution Modeling, A/B Testing, Analytics. Experience: 5+ years.",
            "resume_text": "Digital Marketing Specialist, 3 years. Ran some paid campaigns. No strategic or attribution modeling.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Marketing Manager (Performance). Skills: Performance Marketing, Paid Search (Google Ads), Paid Social (Facebook Ads), Attribution Modeling, A/B Testing, Analytics. Experience: 5+ years.",
            "resume_text": "Brand Manager, 8 years. Focused on brand building. No direct performance marketing.",
            "relevance_score": 30
        },

        # Example: Senior Financial Controller
        {
            "jd_text": "Senior Financial Controller. Skills: Financial Reporting, GAAP/IFRS, Internal Controls, Budgeting, Forecasting, Treasury Management, ERP (SAP/Oracle). Experience: 10+ years.",
            "resume_text": "Senior Financial Controller, 12 years. Oversaw all financial reporting ensuring GAAP/IFRS compliance. Designed and implemented robust internal controls. Led budgeting and forecasting processes. Managed treasury operations. Expert in SAP ERP.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior Financial Controller. Skills: Financial Reporting, GAAP/IFRS, Internal Controls, Budgeting, Forecasting, Treasury Management, ERP (SAP/Oracle). Experience: 10+ years.",
            "resume_text": "Financial Analyst, 7 years. Assisted with budgeting. Some reporting. No controller-level responsibility.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Financial Controller. Skills: Financial Reporting, GAAP/IFRS, Internal Controls, Budgeting, Forecasting, Treasury Management, ERP (SAP/Oracle). Experience: 10+ years.",
            "resume_text": "Operations Manager, 15 years. Managed operational budgets. No financial reporting or GAAP.",
            "relevance_score": 35
        },

        # Example: Senior HR Manager (Compensation & Benefits)
        {
            "jd_text": "Senior HR Manager (C&B). Skills: Compensation & Benefits Strategy, Salary Benchmarking, Benefits Administration, HRIS (Workday), Compliance, Global C&B. Experience: 7+ years.",
            "resume_text": "Senior HR Manager, 8 years. Developed and implemented compensation and benefits strategies. Conducted salary benchmarking. Managed benefits administration. Proficient in Workday HRIS. Ensured global C&B compliance.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior HR Manager (C&B). Skills: Compensation & Benefits Strategy, Salary Benchmarking, Benefits Administration, HRIS (Workday), Compliance, Global C&B. Experience: 7+ years.",
            "resume_text": "HR Generalist, 4 years. Assisted with benefits enrollment. No strategic C&B focus.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior HR Manager (C&B). Skills: Compensation & Benefits Strategy, Salary Benchmarking, Benefits Administration, HRIS (Workday), Compliance, Global C&B. Experience: 7+ years.",
            "resume_text": "Recruitment Manager, 10 years. Focused on talent acquisition. No C&B expertise.",
            "relevance_score": 25
        },

        # Example: Senior Game Developer (Unreal Engine)
        {
            "jd_text": "Senior Game Developer (Unreal Engine). Skills: Unreal Engine 5, C++, Blueprint, Game Systems Design, Multiplayer Networking, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Senior Game Developer, 6 years. Developed complex game systems in Unreal Engine 5 using C++ and Blueprint. Designed multiplayer networking. Optimized game performance. Shipped 2 AAA titles.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Game Developer (Unreal Engine). Skills: Unreal Engine 5, C++, Blueprint, Game Systems Design, Multiplayer Networking, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Game Developer (Unity), 4 years. Proficient in Unity/C#. No Unreal experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "Senior Game Developer (Unreal Engine). Skills: Unreal Engine 5, C++, Blueprint, Game Systems Design, Multiplayer Networking, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Software Engineer (Web), 8 years. No game development.",
            "relevance_score": 20
        },

        # Example: Senior E-commerce Specialist
        {
            "jd_text": "Senior E-commerce Specialist. Skills: Shopify/Magento, E-commerce Strategy, SEO/SEM, Conversion Funnels, A/B Testing, Google Analytics, Customer Journey Mapping. Experience: 4+ years.",
            "resume_text": "Senior E-commerce Specialist, 5 years. Managed and optimized e-commerce strategy on Shopify. Led SEO/SEM initiatives. Optimized conversion funnels. Conducted extensive A/B testing. Expert in Google Analytics and customer journey mapping.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior E-commerce Specialist. Skills: Shopify/Magento, E-commerce Strategy, SEO/SEM, Conversion Funnels, A/B Testing, Google Analytics, Customer Journey Mapping. Experience: 4+ years.",
            "resume_text": "E-commerce Coordinator, 2 years. Assisted with product listings. Some SEO. No strategic or A/B testing.",
            "relevance_score": 55
        },
        {
            "jd_text": "Senior E-commerce Specialist. Skills: Shopify/Magento, E-commerce Strategy, SEO/SEM, Conversion Funnels, A/B Testing, Google Analytics, Customer Journey Mapping. Experience: 4+ years.",
            "resume_text": "Digital Marketing Manager, 7 years. Focused on general digital marketing. No specific e-commerce platform expertise.",
            "relevance_score": 65
        },

        # Example: Senior Robotics Engineer
        {
            "jd_text": "Senior Robotics Engineer. Skills: ROS, C++/Python, SLAM, Motion Planning, Robot Manipulation, Sensor Fusion, Machine Learning for Robotics. Experience: 5+ years.",
            "resume_text": "Senior Robotics Engineer, 6 years. Developed advanced robotics solutions using ROS, C++, and Python. Expertise in SLAM and motion planning. Implemented robot manipulation algorithms. Strong in sensor fusion and applied ML for robotics.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Robotics Engineer. Skills: ROS, C++/Python, SLAM, Motion Planning, Robot Manipulation, Sensor Fusion, Machine Learning for Robotics. Experience: 5+ years.",
            "resume_text": "Robotics Engineer, 2 years. Developed basic robot programs. Some ROS. No advanced SLAM or motion planning.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Robotics Engineer. Skills: ROS, C++/Python, SLAM, Motion Planning, Robot Manipulation, Sensor Fusion, Machine Learning for Robotics. Experience: 5+ years.",
            "resume_text": "Software Engineer (Embedded), 8 years. Developed firmware. No robotics specific. ",
            "relevance_score": 30
        },

        # Example: Senior Technical Account Manager
        {
            "jd_text": "Senior Technical Account Manager. Skills: Strategic Client Management, Technical Consulting, SaaS Solutions, Escalation Management, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Senior Technical Account Manager, 7 years. Managed strategic client relationships for complex SaaS solutions. Provided expert technical consulting. Led escalation management. Collaborated across engineering and sales. Provided leadership.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Technical Account Manager. Skills: Strategic Client Management, Technical Consulting, SaaS Solutions, Escalation Management, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Technical Account Manager, 3 years. Managed smaller accounts. Some technical support. No strategic client management.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Technical Account Manager. Skills: Strategic Client Management, Technical Consulting, SaaS Solutions, Escalation Management, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Sales Manager, 10 years. Focused on sales targets. No technical consulting or post-sales.",
            "relevance_score": 30
        },

        # Example: Senior Biomedical Engineer
        {
            "jd_text": "Senior Biomedical Engineer. Skills: Medical Device Development, Biocompatibility, Regulatory Affairs (FDA/CE), Quality Systems (ISO 13485), Prototyping, Validation. Experience: 5+ years.",
            "resume_text": "Senior Biomedical Engineer, 6 years. Led medical device development projects. Ensured biocompatibility. Managed regulatory affairs for FDA/CE approval. Implemented ISO 13485 quality systems. Oversaw prototyping and validation.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Biomedical Engineer. Skills: Medical Device Development, Biocompatibility, Regulatory Affairs (FDA/CE), Quality Systems (ISO 13485), Prototyping, Validation. Experience: 5+ years.",
            "resume_text": "Biomedical Engineer, 3 years. Designed components. Some prototyping. No regulatory or quality systems.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Biomedical Engineer. Skills: Medical Device Development, Biocompatibility, Regulatory Affairs (FDA/CE), Quality Systems (ISO 13485), Prototyping, Validation. Experience: 5+ years.",
            "resume_text": "Mechanical Engineer, 8 years. Designed industrial machinery. No medical device domain.",
            "relevance_score": 30
        },

        # Example: Senior Investment Analyst
        {
            "jd_text": "Senior Investment Analyst. Skills: Equity Research, Financial Modeling (DCF/Multiples), Valuation, Portfolio Management, Bloomberg/Refinitiv, CFA Charterholder. Experience: 5+ years.",
            "resume_text": "Senior Investment Analyst, 6 years. Led equity research for multiple sectors. Built complex DCF and multiples-based financial models. Performed in-depth valuations. Assisted with portfolio management. Expert in Bloomberg Terminal. CFA Charterholder.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior Investment Analyst. Skills: Equity Research, Financial Modeling (DCF/Multiples), Valuation, Portfolio Management, Bloomberg/Refinitiv, CFA Charterholder. Experience: 5+ years.",
            "resume_text": "Investment Analyst, 3 years. Built basic models. Some equity research. No senior-level leadership or full CFA.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Investment Analyst. Skills: Equity Research, Financial Modeling (DCF/Multiples), Valuation, Portfolio Management, Bloomberg/Refinitiv, CFA Charterholder. Experience: 5+ years.",
            "resume_text": "Financial Accountant, 10 years. Prepared financial statements. No investment analysis.",
            "relevance_score": 25
        },

        # Example: Senior Digital Marketing Analyst
        {
            "jd_text": "Senior Digital Marketing Analyst. Skills: Advanced Google Analytics, Data Studio/Looker, SQL, Python (Pandas), A/B Testing, Marketing Mix Modeling, Attribution. Experience: 4+ years.",
            "resume_text": "Senior Digital Marketing Analyst, 5 years. Expert in Google Analytics and Data Studio. Wrote complex SQL queries and Python scripts for marketing data analysis. Led A/B testing programs. Developed marketing mix models and attribution strategies.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Digital Marketing Analyst. Skills: Advanced Google Analytics, Data Studio/Looker, SQL, Python (Pandas), A/B Testing, Marketing Mix Modeling, Attribution. Experience: 4+ years.",
            "resume_text": "Digital Marketing Specialist, 2 years. Ran campaigns. Some Google Analytics. No advanced analytics or modeling.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Digital Marketing Analyst. Skills: Advanced Google Analytics, Data Studio/Looker, SQL, Python (Pandas), A/B Testing, Marketing Mix Modeling, Attribution. Experience: 4+ years.",
            "resume_text": "Data Analyst, 7 years. Strong in SQL and Python. No specific digital marketing domain.",
            "relevance_score": 75
        },

        # Example: Senior Supply Chain Analyst
        {
            "jd_text": "Senior Supply Chain Analyst. Skills: Supply Chain Optimization, Predictive Analytics, Inventory Forecasting, Network Design, SAP/Oracle SCM, Data Visualization. Experience: 5+ years.",
            "resume_text": "Senior Supply Chain Analyst, 6 years. Led supply chain optimization projects. Developed predictive analytics models for inventory forecasting. Designed supply chain networks. Proficient in SAP SCM and data visualization tools.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Supply Chain Analyst. Skills: Supply Chain Optimization, Predictive Analytics, Inventory Forecasting, Network Design, SAP/Oracle SCM, Data Visualization. Experience: 5+ years.",
            "resume_text": "Supply Chain Analyst, 3 years. Analyzed logistics data. Some inventory planning. No predictive analytics or network design.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Supply Chain Analyst. Skills: Supply Chain Optimization, Predictive Analytics, Inventory Forecasting, Network Design, SAP/Oracle SCM, Data Visualization. Experience: 5+ years.",
            "resume_text": "Operations Manager, 10 years. Managed warehouse operations. No analytical focus.",
            "relevance_score": 30
        },

        # Example: Principal Software Engineer (Backend)
        {
            "jd_text": "Principal Backend Engineer. Skills: Java/Go, Distributed Systems, Microservices, Cloud-Native (AWS/Azure), Database Scalability, Performance Tuning, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Principal Backend Engineer, 10 years. Designed and built highly scalable distributed systems and microservices in Java. Expertise in cloud-native development on AWS. Optimized database performance. Provided technical leadership and mentorship.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Backend Engineer. Skills: Java/Go, Distributed Systems, Microservices, Cloud-Native (AWS/Azure), Database Scalability, Performance Tuning, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Senior Backend Engineer, 6 years. Built APIs. Some microservices. No principal-level leadership or distributed systems expertise.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Backend Engineer. Skills: Java/Go, Distributed Systems, Microservices, Cloud-Native (AWS/Azure), Database Scalability, Performance Tuning, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Frontend Engineer, 12 years. No backend experience.",
            "relevance_score": 20
        },

        # Example: Senior Database Developer
        {
            "jd_text": "Senior Database Developer. Skills: SQL Server/PostgreSQL, Complex Stored Procedures, Database Optimization, Data Warehousing, ETL Frameworks, Performance Tuning. Experience: 5+ years.",
            "resume_text": "Senior Database Developer, 6 years. Developed and optimized complex stored procedures in SQL Server and PostgreSQL. Designed scalable databases for data warehousing. Built robust ETL frameworks. Expert in performance tuning.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Database Developer. Skills: SQL Server/PostgreSQL, Complex Stored Procedures, Database Optimization, Data Warehousing, ETL Frameworks, Performance Tuning. Experience: 5+ years.",
            "resume_text": "Database Developer, 3 years. Wrote SQL queries. Some stored procedures. No optimization or ETL framework design.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Database Developer. Skills: SQL Server/PostgreSQL, Complex Stored Procedures, Database Optimization, Data Warehousing, ETL Frameworks, Performance Tuning. Experience: 5+ years.",
            "resume_text": "Data Analyst, 7 years. Strong in SQL. No database development.",
            "relevance_score": 40
        },

        # Example: Director of Education
        {
            "jd_text": "Director of Education. Skills: Educational Leadership, Curriculum Development, Program Management, Budget Oversight, Staff Development, Stakeholder Relations. Experience: 10+ years.",
            "resume_text": "Director of Education, 12 years. Provided educational leadership. Oversaw curriculum development and program management. Managed departmental budgets. Led staff development initiatives. Cultivated strong stakeholder relations.",
            "relevance_score": 96
        },
        {
            "jd_text": "Director of Education. Skills: Educational Leadership, Curriculum Development, Program Management, Budget Oversight, Staff Development, Stakeholder Relations. Experience: 10+ years.",
            "resume_text": "Education Program Manager, 6 years. Managed programs. Some curriculum input. No director-level leadership.",
            "relevance_score": 70
        },
        {
            "jd_text": "Director of Education. Skills: Educational Leadership, Curriculum Development, Program Management, Budget Oversight, Staff Development, Stakeholder Relations. Experience: 10+ years.",
            "resume_text": "Principal (School), 15 years. Managed a school. No broader education program focus.",
            "relevance_score": 50
        },

        # Example: Senior Environmental Engineer
        {
            "jd_text": "Senior Environmental Engineer. Skills: Air Quality Modeling, Waste Management, Environmental Impact Assessment (EIA), Remediation Technologies, Regulatory Compliance. Experience: 5+ years.",
            "resume_text": "Senior Environmental Engineer, 6 years. Conducted air quality modeling. Managed complex waste streams. Led Environmental Impact Assessments. Implemented advanced remediation technologies. Ensured strict regulatory compliance.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Environmental Engineer. Skills: Air Quality Modeling, Waste Management, Environmental Impact Assessment (EIA), Remediation Technologies, Regulatory Compliance. Experience: 5+ years.",
            "resume_text": "Environmental Engineer, 3 years. Performed site assessments. Some regulatory knowledge. No advanced modeling or EIA leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Environmental Engineer. Skills: Air Quality Modeling, Waste Management, Environmental Impact Assessment (EIA), Remediation Technologies, Regulatory Compliance. Experience: 5+ years.",
            "resume_text": "Civil Engineer, 8 years. Focused on infrastructure. No environmental specialization.",
            "relevance_score": 30
        },

        # Example: Senior Business Intelligence Developer
        {
            "jd_text": "Senior Business Intelligence Developer. Skills: Data Warehousing, ETL Architecture, Power BI/Tableau, SQL Optimization, SSIS/SSRS, Cloud BI (Azure Synapse/AWS Redshift). Experience: 5+ years.",
            "resume_text": "Senior BI Developer, 6 years. Designed and built scalable data warehouses. Architected ETL processes. Developed advanced dashboards in Power BI. Optimized complex SQL queries. Expertise in SSIS/SSRS and Azure Synapse.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Business Intelligence Developer. Skills: Data Warehousing, ETL Architecture, Power BI/Tableau, SQL Optimization, SSIS/SSRS, Cloud BI (Azure Synapse/AWS Redshift). Experience: 5+ years.",
            "resume_text": "BI Developer, 3 years. Built dashboards. Some ETL. No architecture or cloud BI expertise.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Business Intelligence Developer. Skills: Data Warehousing, ETL Architecture, Power BI/Tableau, SQL Optimization, SSIS/SSRS, Cloud BI (Azure Synapse/AWS Redshift). Experience: 5+ years.",
            "resume_text": "Data Analyst, 7 years. Strong in SQL. No BI development or data warehousing.",
            "relevance_score": 40
        },

        # Example: Senior Pharmacist
        {
            "jd_text": "Senior Pharmacist. Skills: Clinical Pharmacy, Medication Therapy Management (MTM), Drug Information, Patient Safety, Pharmacy Leadership, Regulatory Compliance. Experience: 5+ years.",
            "resume_text": "Senior Pharmacist, 6 years. Provided advanced clinical pharmacy services. Conducted Medication Therapy Management. Expert in drug information. Championed patient safety initiatives. Provided leadership to pharmacy staff. Ensured regulatory compliance.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Pharmacist. Skills: Clinical Pharmacy, Medication Therapy Management (MTM), Drug Information, Patient Safety, Pharmacy Leadership, Regulatory Compliance. Experience: 5+ years.",
            "resume_text": "Pharmacist, 3 years. Dispensed medication. Some patient counseling. No MTM or leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Pharmacist. Skills: Clinical Pharmacy, Medication Therapy Management (MTM), Drug Information, Patient Safety, Pharmacy Leadership, Regulatory Compliance. Experience: 5+ years.",
            "resume_text": "Nurse Manager, 10 years. Managed nursing staff. No pharmacy background.",
            "relevance_score": 25
        },

        # Example: Senior Social Media Manager
        {
            "jd_text": "Senior Social Media Manager. Skills: Social Media Strategy, Influencer Marketing, Crisis Management, Advanced Analytics, Paid Social Advertising, Team Leadership. Experience: 5+ years.",
            "resume_text": "Senior Social Media Manager, 6 years. Developed and executed comprehensive social media strategies. Led influencer marketing campaigns. Managed social media crises. Utilized advanced analytics. Oversaw paid social advertising. Led a team of specialists.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Social Media Manager. Skills: Social Media Strategy, Influencer Marketing, Crisis Management, Advanced Analytics, Paid Social Advertising, Team Leadership. Experience: 5+ years.",
            "resume_text": "Social Media Specialist, 3 years. Created content. Ran basic ads. No strategic or crisis management.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Social Media Manager. Skills: Social Media Strategy, Influencer Marketing, Crisis Management, Advanced Analytics, Paid Social Advertising, Team Leadership. Experience: 5+ years.",
            "resume_text": "Marketing Manager, 8 years. Focused on traditional marketing. No social media expertise.",
            "relevance_score": 30
        },

        # Example: Senior Research Engineer
        {
            "jd_text": "Senior Research Engineer. Skills: Advanced R&D, Experimental Design, Data Modeling, Simulation (MATLAB/Python), Patent Writing, Scientific Publications. Experience: 6+ years.",
            "resume_text": "Senior Research Engineer, 7 years. Led advanced R&D projects. Designed complex experiments. Developed data models and simulations in MATLAB/Python. Wrote patent applications and published scientific papers.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior Research Engineer. Skills: Advanced R&D, Experimental Design, Data Modeling, Simulation (MATLAB/Python), Patent Writing, Scientific Publications. Experience: 6+ years.",
            "resume_text": "Research Engineer, 3 years. Conducted experiments. Some data analysis. No patent writing or senior leadership.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Research Engineer. Skills: Advanced R&D, Experimental Design, Data Modeling, Simulation (MATLAB/Python), Patent Writing, Scientific Publications. Experience: 6+ years.",
            "resume_text": "Software Engineer, 8 years. Built applications. No research or scientific focus.",
            "relevance_score": 25
        },

        # Example: Principal Solutions Architect
        {
            "jd_text": "Principal Solutions Architect. Skills: Enterprise Architecture, Cloud Strategy, Digital Transformation, Microservices, Security Architecture, Technical Governance, Executive Communication. Experience: 10+ years.",
            "resume_text": "Principal Solutions Architect, 12 years. Defined enterprise architecture and cloud strategy. Led digital transformation initiatives. Designed microservices and robust security architectures. Established technical governance. Presented to executive leadership.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Solutions Architect. Skills: Enterprise Architecture, Cloud Strategy, Digital Transformation, Microservices, Security Architecture, Technical Governance, Executive Communication. Experience: 10+ years.",
            "resume_text": "Solutions Architect, 6 years. Designed cloud solutions. Some microservices. No enterprise architecture or executive communication.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Solutions Architect. Skills: Enterprise Architecture, Cloud Strategy, Digital Transformation, Microservices, Security Architecture, Technical Governance, Executive Communication. Experience: 10+ years.",
            "resume_text": "Senior Project Manager, 15 years. Managed large projects. No technical architecture.",
            "relevance_score": 40
        },

        # Example: Senior Technical Trainer
        {
            "jd_text": "Senior Technical Trainer. Skills: Advanced Training Delivery, Instructional Design, Technical Curriculum Development, E-learning Platforms, Mentorship. Experience: 5+ years.",
            "resume_text": "Senior Technical Trainer, 6 years. Delivered advanced technical training. Led instructional design and curriculum development. Proficient in e-learning platforms. Mentored junior trainers.",
            "relevance_score": 93
        },
        {
            "jd_text": "Senior Technical Trainer. Skills: Advanced Training Delivery, Instructional Design, Technical Curriculum Development, E-learning Platforms, Mentorship. Experience: 5+ years.",
            "resume_text": "Technical Trainer, 2 years. Delivered training. Some curriculum input. No senior-level or mentorship.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Technical Trainer. Skills: Advanced Training Delivery, Instructional Design, Technical Curriculum Development, E-learning Platforms, Mentorship. Experience: 5+ years.",
            "resume_text": "Software Developer, 8 years. Strong technical skills. No training or instructional design.",
            "relevance_score": 30
        },

        # Example: Senior Data Privacy Officer
        {
            "jd_text": "Senior Data Privacy Officer. Skills: Global Data Privacy Laws (GDPR, CCPA, LGPD), Privacy by Design, Data Protection Impact Assessments (DPIA), Incident Response, Vendor Risk Management. Experience: 7+ years.",
            "resume_text": "Senior Data Privacy Officer, 8 years. Ensured compliance with global data privacy laws (GDPR, CCPA, LGPD). Implemented Privacy by Design principles. Led DPIAs. Managed privacy incident response. Oversaw vendor risk management.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Data Privacy Officer. Skills: Global Data Privacy Laws (GDPR, CCPA, LGPD), Privacy by Design, Data Protection Impact Assessments (DPIA), Incident Response, Vendor Risk Management. Experience: 7+ years.",
            "resume_text": "Data Privacy Officer, 4 years. Focused on GDPR. Some DPIA. No global scope or senior leadership.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Data Privacy Officer. Skills: Global Data Privacy Laws (GDPR, CCPA, LGPD), Privacy by Design, Data Protection Impact Assessments (DPIA), Incident Response, Vendor Risk Management. Experience: 7+ years.",
            "resume_text": "Compliance Manager, 10 years. Focused on general regulatory compliance. No data privacy specialization.",
            "relevance_score": 40
        },

        # Example: Senior Quantitative Analyst
        {
            "jd_text": "Senior Quantitative Analyst. Skills: Stochastic Calculus, Option Pricing, Risk Modeling, Python/R, C++, Financial Engineering, Derivatives. Experience: 5+ years.",
            "resume_text": "Senior Quantitative Analyst, 6 years. Applied stochastic calculus to option pricing models. Developed complex risk models. Proficient in Python, R, and C++. Expertise in financial engineering and derivatives.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior Quantitative Analyst. Skills: Stochastic Calculus, Option Pricing, Risk Modeling, Python/R, C++, Financial Engineering, Derivatives. Experience: 5+ years.",
            "resume_text": "Quantitative Analyst, 3 years. Built basic models. Some Python. No advanced financial engineering or derivatives.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Quantitative Analyst. Skills: Stochastic Calculus, Option Pricing, Risk Modeling, Python/R, C++, Financial Engineering, Derivatives. Experience: 5+ years.",
            "resume_text": "Financial Analyst, 8 years. Focused on corporate finance. No quantitative modeling or advanced math.",
            "relevance_score": 25
        },

        # Example: Senior Product Marketing Manager
        {
            "jd_text": "Senior Product Marketing Manager. Skills: Product Launch Strategy, GTM Planning, Competitive Intelligence, Messaging & Positioning, Sales Enablement, Analyst Relations. Experience: 5+ years.",
            "resume_text": "Senior Product Marketing Manager, 6 years. Led product launch strategies and GTM planning. Conducted competitive intelligence. Developed compelling messaging and positioning. Created sales enablement programs. Managed analyst relations.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Product Marketing Manager. Skills: Product Launch Strategy, GTM Planning, Competitive Intelligence, Messaging & Positioning, Sales Enablement, Analyst Relations. Experience: 5+ years.",
            "resume_text": "Product Marketing Specialist, 3 years. Assisted with launches. Some messaging. No strategic leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Product Marketing Manager. Skills: Product Launch Strategy, GTM Planning, Competitive Intelligence, Messaging & Positioning, Sales Enablement, Analyst Relations. Experience: 5+ years.",
            "resume_text": "Product Manager, 8 years. Defined product features. No marketing focus.",
            "relevance_score": 30
        },

        # Example: Principal Site Reliability Engineer (SRE)
        {
            "jd_text": "Principal SRE. Skills: Distributed Systems Reliability, Performance Engineering, Observability (OpenTelemetry), Chaos Engineering, Incident Management, Cloud-Native Architecture. Experience: 8+ years.",
            "resume_text": "Principal SRE, 10 years. Led reliability engineering for large-scale distributed systems. Optimized performance. Implemented OpenTelemetry for observability. Practiced chaos engineering. Oversaw critical incident management. Designed cloud-native architectures.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal SRE. Skills: Distributed Systems Reliability, Performance Engineering, Observability (OpenTelemetry), Chaos Engineering, Incident Management, Cloud-Native Architecture. Experience: 8+ years.",
            "resume_text": "Senior SRE, 5 years. Focused on incident response. Some cloud. No principal-level strategy or chaos engineering.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal SRE. Skills: Distributed Systems Reliability, Performance Engineering, Observability (OpenTelemetry), Chaos Engineering, Incident Management, Cloud-Native Architecture. Experience: 8+ years.",
            "resume_text": "DevOps Manager, 12 years. Managed DevOps teams. No deep SRE expertise.",
            "relevance_score": 40
        },

        # Example: Principal Data Architect
        {
            "jd_text": "Principal Data Architect. Skills: Enterprise Data Strategy, Data Governance, Cloud Data Platforms (Databricks, Snowflake), Data Mesh, Data Fabric, Big Data Ecosystems, Technical Leadership. Experience: 10+ years.",
            "resume_text": "Principal Data Architect, 12 years. Defined enterprise data strategy and governance. Designed data solutions on Databricks and Snowflake. Led data mesh and data fabric initiatives. Expertise in entire big data ecosystem. Provided technical leadership.",
            "relevance_score": 99
        },
        {
            "jd_text": "Principal Data Architect. Skills: Enterprise Data Strategy, Data Governance, Cloud Data Platforms (Databricks, Snowflake), Data Mesh, Data Fabric, Big Data Ecosystems, Technical Leadership. Experience: 10+ years.",
            "resume_text": "Senior Data Architect, 7 years. Designed data warehouses. Some cloud data platforms. No principal-level strategy or data fabric.",
            "relevance_score": 80
        },
        {
            "jd_text": "Principal Data Architect. Skills: Enterprise Data Strategy, Data Governance, Cloud Data Platforms (Databricks, Snowflake), Data Mesh, Data Fabric, Big Data Ecosystems, Technical Leadership. Experience: 10+ years.",
            "resume_text": "Data Scientist, 8 years. Built ML models. Some data engineering. No data architecture.",
            "relevance_score": 45
        },

        # Example: Senior Technical Sales Engineer
        {
            "jd_text": "Senior Technical Sales Engineer. Skills: Complex Solution Selling, Product Expertise (SaaS/Hardware), Technical Presentations, Proof-of-Concept (POC), Competitive Analysis, Relationship Building. Experience: 5+ years.",
            "resume_text": "Senior Technical Sales Engineer, 6 years. Drove complex solution selling cycles. Provided deep product expertise for SaaS and hardware. Delivered compelling technical presentations and POCs. Conducted competitive analysis. Built strong client relationships.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Technical Sales Engineer. Skills: Complex Solution Selling, Product Expertise (SaaS/Hardware), Technical Presentations, Proof-of-Concept (POC), Competitive Analysis, Relationship Building. Experience: 5+ years.",
            "resume_text": "Technical Sales Engineer, 3 years. Conducted demos. Some sales. No complex solution selling.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Technical Sales Engineer. Skills: Complex Solution Selling, Product Expertise (SaaS/Hardware), Technical Presentations, Proof-of-Concept (POC), Competitive Analysis, Relationship Building. Experience: 5+ years.",
            "resume_text": "Sales Manager, 10 years. Managed sales teams. No technical product expertise.",
            "relevance_score": 30
        },

        # Example: Research Scientist (Drug Discovery)
        {
            "jd_text": "Research Scientist (Drug Discovery). Skills: Medicinal Chemistry, Organic Synthesis, Assay Development, ADME/PK, Cell Biology, Publications. Experience: PhD + 3 years industry.",
            "resume_text": "Research Scientist, PhD + 4 years industry. Expertise in medicinal chemistry and organic synthesis for drug discovery. Developed and optimized assays. Strong understanding of ADME/PK. Proficient in cell biology techniques. Published in peer-reviewed journals.",
            "relevance_score": 97
        },
        {
            "jd_text": "Research Scientist (Drug Discovery). Skills: Medicinal Chemistry, Organic Synthesis, Assay Development, ADME/PK, Cell Biology, Publications. Experience: PhD + 3 years industry.",
            "resume_text": "Research Chemist, 2 years. Performed synthesis. Some assay work. No drug discovery focus.",
            "relevance_score": 60
        },
        {
            "jd_text": "Research Scientist (Drug Discovery). Skills: Medicinal Chemistry, Organic Synthesis, Assay Development, ADME/PK, Cell Biology, Publications. Experience: PhD + 3 years industry.",
            "resume_text": "Clinical Research Coordinator, 5 years. Managed trials. No lab research.",
            "relevance_score": 25
        },

        # Example: Chief Technology Officer (CTO)
        {
            "jd_text": "Chief Technology Officer (CTO). Skills: Technology Strategy, Software Architecture, R&D Leadership, Engineering Management, Scalability, Innovation, Budget Oversight. Experience: 15+ years.",
            "resume_text": "CTO with 20 years experience. Defined and executed technology strategy. Oversaw software architecture for global products. Led R&D and engineering teams. Ensured scalability and drove innovation. Managed large technology budgets.",
            "relevance_score": 99
        },
        {
            "jd_text": "Chief Technology Officer (CTO). Skills: Technology Strategy, Software Architecture, R&D Leadership, Engineering Management, Scalability, Innovation, Budget Oversight. Experience: 15+ years.",
            "resume_text": "Director of Engineering, 10 years. Led engineering teams. Some architecture. No CTO-level strategic leadership.",
            "relevance_score": 70
        },
        {
            "jd_text": "Chief Technology Officer (CTO). Skills: Technology Strategy, Software Architecture, R&D Leadership, Engineering Management, Scalability, Innovation, Budget Oversight. Experience: 15+ years.",
            "resume_text": "Principal Engineer, 12 years. Strong technical contributor. No management or strategic role.",
            "relevance_score": 40
        },

        # Example: Vice President of Sales
        {
            "jd_text": "VP of Sales. Skills: Sales Strategy, Revenue Growth, Team Leadership, Key Account Management, CRM (Salesforce), Global Sales, P&L Responsibility. Experience: 12+ years.",
            "resume_text": "VP of Sales, 15 years. Developed and executed global sales strategies. Consistently drove significant revenue growth. Led large sales organizations. Managed key accounts. Expert in Salesforce. Held P&L responsibility.",
            "relevance_score": 98
        },
        {
            "jd_text": "VP of Sales. Skills: Sales Strategy, Revenue Growth, Team Leadership, Key Account Management, CRM (Salesforce), Global Sales, P&L Responsibility. Experience: 12+ years.",
            "resume_text": "Sales Director, 8 years. Led regional sales teams. Some key account management. No VP-level strategy or global P&L.",
            "relevance_score": 75
        },
        {
            "jd_text": "VP of Sales. Skills: Sales Strategy, Revenue Growth, Team Leadership, Key Account Management, CRM (Salesforce), Global Sales, P&L Responsibility. Experience: 12+ years.",
            "resume_text": "Marketing Director, 10 years. Focused on marketing. No sales leadership.",
            "relevance_score": 30
        },

        # Example: Senior Manufacturing Engineer
        {
            "jd_text": "Senior Manufacturing Engineer. Skills: Lean Six Sigma, Process Improvement, Automation, Robotics, CAD/CAM, Production Optimization, Quality Systems. Experience: 5+ years.",
            "resume_text": "Senior Manufacturing Engineer, 6 years. Led Lean Six Sigma initiatives for process improvement. Designed and implemented automation and robotics solutions. Proficient in CAD/CAM. Optimized production lines. Developed and maintained quality systems.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Manufacturing Engineer. Skills: Lean Six Sigma, Process Improvement, Automation, Robotics, CAD/CAM, Production Optimization, Quality Systems. Experience: 5+ years.",
            "resume_text": "Manufacturing Engineer, 3 years. Optimized some processes. Some CAD. No Lean Six Sigma or automation leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Manufacturing Engineer. Skills: Lean Six Sigma, Process Improvement, Automation, Robotics, CAD/CAM, Production Optimization, Quality Systems. Experience: 5+ years.",
            "resume_text": "Production Manager, 10 years. Managed production teams. No engineering background.",
            "relevance_score": 35
        },

        # Example: Senior Salesforce Administrator
        {
            "jd_text": "Senior Salesforce Administrator. Skills: Salesforce Admin (Advanced), Apex/Visualforce (basic), Lightning Web Components (LWC), Integrations, Data Migration, Security Best Practices. Experience: 4+ years.",
            "resume_text": "Senior Salesforce Administrator, 5 years. Managed complex Salesforce orgs. Wrote basic Apex and LWC components. Led integrations and data migrations. Implemented security best practices. Certified Advanced Admin.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Salesforce Administrator. Skills: Salesforce Admin (Advanced), Apex/Visualforce (basic), Lightning Web Components (LWC), Integrations, Data Migration, Security Best Practices. Experience: 4+ years.",
            "resume_text": "Salesforce Administrator, 2 years. Managed users and reports. No development or advanced features.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Salesforce Administrator. Skills: Salesforce Admin (Advanced), Apex/Visualforce (basic), Lightning Web Components (LWC), Integrations, Data Migration, Security Best Practices. Experience: 4+ years.",
            "resume_text": "CRM Consultant, 7 years. Implemented various CRM systems. No specific Salesforce focus.",
            "relevance_score": 40
        },

        # Example: Senior Executive Assistant
        {
            "jd_text": "Senior Executive Assistant. Skills: Executive Support, Project Management, Event Planning, Board Relations, Confidentiality, Advanced Microsoft Office. Experience: 8+ years.",
            "resume_text": "Senior Executive Assistant, 10 years. Provided high-level support to C-suite executives. Managed special projects and planned corporate events. Coordinated board relations. Handled confidential information with discretion. Expert in Microsoft Office Suite.",
            "relevance_score": 92
        },
        {
            "jd_text": "Senior Executive Assistant. Skills: Executive Support, Project Management, Event Planning, Board Relations, Confidentiality, Advanced Microsoft Office. Experience: 8+ years.",
            "resume_text": "Executive Assistant, 4 years. Managed calendars. Some travel. No project management or board relations.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Executive Assistant. Skills: Executive Support, Project Management, Event Planning, Board Relations, Confidentiality, Advanced Microsoft Office. Experience: 8+ years.",
            "resume_text": "Office Manager, 12 years. Managed office operations. No executive support.",
            "relevance_score": 30
        },

        # Example: Senior Cloud Solutions Architect
        {
            "jd_text": "Senior Cloud Solutions Architect. Skills: Multi-Cloud (AWS, Azure, GCP), Cloud Migration, Cost Optimization, Security Architecture, DevOps Integration, Enterprise Solutions. Experience: 7+ years.",
            "resume_text": "Senior Cloud Solutions Architect, 8 years. Designed and led multi-cloud solutions across AWS, Azure, and GCP. Managed large-scale cloud migrations. Optimized cloud costs and security architecture. Integrated DevOps practices. Delivered enterprise-level solutions.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior Cloud Solutions Architect. Skills: Multi-Cloud (AWS, Azure, GCP), Cloud Migration, Cost Optimization, Security Architecture, DevOps Integration, Enterprise Solutions. Experience: 7+ years.",
            "resume_text": "Cloud Engineer, 4 years. Deployed resources in AWS. Some architecture. No multi-cloud or cost optimization focus.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Cloud Solutions Architect. Skills: Multi-Cloud (AWS, Azure, GCP), Cloud Migration, Cost Optimization, Security Architecture, DevOps Integration, Enterprise Solutions. Experience: 7+ years.",
            "resume_text": "IT Manager, 10 years. Managed IT infrastructure. No cloud architecture.",
            "relevance_score": 40
        },

        # Example: Principal Data Engineer
        {
            "jd_text": "Principal Data Engineer. Skills: Big Data Ecosystems (Hadoop, Spark, Kafka), Data Lake/Warehouse Architecture, Real-time Data Processing, Performance Tuning, Data Governance, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Principal Data Engineer, 10 years. Designed and built large-scale big data ecosystems using Hadoop, Spark, and Kafka. Architected data lakes and warehouses. Implemented real-time data processing. Optimized data pipeline performance. Led data governance initiatives and provided technical leadership.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Data Engineer. Skills: Big Data Ecosystems (Hadoop, Spark, Kafka), Data Lake/Warehouse Architecture, Real-time Data Processing, Performance Tuning, Data Governance, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Senior Data Engineer, 6 years. Built Spark pipelines. Some data warehousing. No principal-level architecture or governance.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Data Engineer. Skills: Big Data Ecosystems (Hadoop, Spark, Kafka), Data Lake/Warehouse Architecture, Real-time Data Processing, Performance Tuning, Data Governance, Technical Leadership. Experience: 8+ years.",
            "resume_text": "Data Architect, 7 years. Designed data models. No hands-on big data engineering.",
            "relevance_score": 50
        },

        # Example: Senior Digital Content Creator
        {
            "jd_text": "Senior Digital Content Creator. Skills: Video Production, Motion Graphics (After Effects), Advanced Graphic Design, Storyboarding, Scriptwriting, Content Strategy. Experience: 4+ years.",
            "resume_text": "Senior Digital Content Creator, 5 years. Led end-to-end video production. Created motion graphics in After Effects. Designed advanced graphics. Developed storyboards and wrote compelling scripts. Contributed to content strategy.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Digital Content Creator. Skills: Video Production, Motion Graphics (After Effects), Advanced Graphic Design, Storyboarding, Scriptwriting, Content Strategy. Experience: 4+ years.",
            "resume_text": "Digital Content Creator, 2 years. Edited videos. Some graphic design. No motion graphics or content strategy leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Digital Content Creator. Skills: Video Production, Motion Graphics (After Effects), Advanced Graphic Design, Storyboarding, Scriptwriting, Content Strategy. Experience: 4+ years.",
            "resume_text": "Marketing Specialist, 7 years. Focused on campaigns. No content creation.",
            "relevance_score": 30
        },

        # Example: Senior Customer Success Manager
        {
            "jd_text": "Senior Customer Success Manager. Skills: Strategic Account Management, Customer Retention, Upselling/Cross-selling, Product Adoption, CSAT/NPS, Team Leadership. Experience: 5+ years.",
            "resume_text": "Senior Customer Success Manager, 6 years. Managed strategic accounts. Drove customer retention and identified upsell/cross-sell opportunities. Increased product adoption. Improved CSAT/NPS scores. Provided leadership to CSM team.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Customer Success Manager. Skills: Strategic Account Management, Customer Retention, Upselling/Cross-selling, Product Adoption, CSAT/NPS, Team Leadership. Experience: 5+ years.",
            "resume_text": "Customer Success Manager, 3 years. Managed small accounts. Some retention efforts. No strategic account management or leadership.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Customer Success Manager. Skills: Strategic Account Management, Customer Retention, Upselling/Cross-selling, Product Adoption, CSAT/NPS, Team Leadership. Experience: 5+ years.",
            "resume_text": "Sales Manager, 8 years. Focused on new sales. No customer retention.",
            "relevance_score": 35
        },

        # Example: Senior Regulatory Affairs Specialist
        {
            "jd_text": "Senior Regulatory Affairs Specialist. Skills: Global Regulatory Strategy, FDA/EMA Submissions, Post-Market Surveillance, Clinical Evaluation Reports (CER), Regulatory Intelligence. Experience: 6+ years.",
            "resume_text": "Senior Regulatory Affairs Specialist, 7 years. Developed global regulatory strategies. Prepared and managed FDA and EMA submissions. Oversaw post-market surveillance. Wrote Clinical Evaluation Reports. Conducted regulatory intelligence.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Regulatory Affairs Specialist. Skills: Global Regulatory Strategy, FDA/EMA Submissions, Post-Market Surveillance, Clinical Evaluation Reports (CER), Regulatory Intelligence. Experience: 6+ years.",
            "resume_text": "Regulatory Affairs Specialist, 3 years. Prepared some submissions. No global strategy or post-market surveillance.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Regulatory Affairs Specialist. Skills: Global Regulatory Strategy, FDA/EMA Submissions, Post-Market Surveillance, Clinical Evaluation Reports (CER), Regulatory Intelligence. Experience: 6+ years.",
            "resume_text": "Quality Assurance Manager, 10 years. Managed quality systems. No regulatory affairs.",
            "relevance_score": 40
        },

        # Example: Principal UX Writer
        {
            "jd_text": "Principal UX Writer. Skills: UX Content Strategy, Information Architecture, Content Governance, A/B Testing (Content), User Research, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Principal UX Writer, 7 years. Defined UX content strategy and information architecture. Established content governance. Led A/B testing for content. Conducted user research. Provided leadership to design and product teams.",
            "relevance_score": 97
        },
        {
            "jd_text": "Principal UX Writer. Skills: UX Content Strategy, Information Architecture, Content Governance, A/B Testing (Content), User Research, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Senior UX Writer, 3 years. Wrote microcopy. Some content strategy. No information architecture or leadership.",
            "relevance_score": 70
        },
        {
            "jd_text": "Principal UX Writer. Skills: UX Content Strategy, Information Architecture, Content Governance, A/B Testing (Content), User Research, Cross-functional Leadership. Experience: 6+ years.",
            "resume_text": "Content Strategist, 8 years. Focused on marketing content. No UX specific.",
            "relevance_score": 45
        },

        # Example: Senior Investment Banking Analyst
        {
            "jd_text": "Senior Investment Banking Analyst. Skills: M&A, Capital Markets, Due Diligence, Financial Modeling (LBO, M&A), Pitch Book Creation, Client Presentations. Experience: 3+ years.",
            "resume_text": "Senior Investment Banking Analyst, 4 years. Executed M&A and capital market transactions. Led due diligence processes. Built complex LBO and M&A financial models. Created compelling pitch books and delivered client presentations.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Investment Banking Analyst. Skills: M&A, Capital Markets, Due Diligence, Financial Modeling (LBO, M&A), Pitch Book Creation, Client Presentations. Experience: 3+ years.",
            "resume_text": "Investment Banking Analyst, 1 year. Assisted with modeling. Some pitch book work. No leadership or complex transactions.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Investment Banking Analyst. Skills: M&A, Capital Markets, Due Diligence, Financial Modeling (LBO, M&A), Pitch Book Creation, Client Presentations. Experience: 3+ years.",
            "resume_text": "Financial Analyst, 5 years. Built corporate finance models. No investment banking deal experience.",
            "relevance_score": 40
        },

        # Example: Senior Data Science Intern
        {
            "jd_text": "Senior Data Science Intern. Skills: Python (Scikit-learn, PyTorch), SQL, Cloud Platforms (AWS/GCP), Model Evaluation, Independent Research. Experience: Master's student + 1 internship.",
            "resume_text": "Master's student in Data Science with 1 previous internship. Proficient in Python (Scikit-learn, PyTorch). Strong SQL. Experience with AWS. Performed model evaluation. Conducted independent research projects.",
            "relevance_score": 88
        },
        {
            "jd_text": "Senior Data Science Intern. Skills: Python (Scikit-learn, PyTorch), SQL, Cloud Platforms (AWS/GCP), Model Evaluation, Independent Research. Experience: Master's student + 1 internship.",
            "resume_text": "Undergraduate student. Basic Python. No ML or cloud experience.",
            "relevance_score": 40
        },
        {
            "jd_text": "Senior Data Science Intern. Skills: Python (Scikit-learn, PyTorch), SQL, Cloud Platforms (AWS/GCP), Model Evaluation, Independent Research. Experience: Master's student + 1 internship.",
            "resume_text": "Software Engineering Intern. Strong in Java. No data science focus.",
            "relevance_score": 25
        },

        # Example: Senior Software Engineer (Fullstack)
        {
            "jd_text": "Senior Fullstack Engineer. Skills: React, Node.js, Microservices, PostgreSQL, AWS, GraphQL, CI/CD, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Senior Fullstack Engineer, 6 years. Built scalable web applications with React and Node.js. Designed and implemented microservices. Managed PostgreSQL databases. Deployed on AWS. Expertise in GraphQL and CI/CD. Optimized application performance.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Fullstack Engineer. Skills: React, Node.js, Microservices, PostgreSQL, AWS, GraphQL, CI/CD, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Fullstack Developer, 3 years. Built basic apps. Some React/Node. No microservices or performance optimization.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Fullstack Engineer. Skills: React, Node.js, Microservices, PostgreSQL, AWS, GraphQL, CI/CD, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Frontend Developer, 8 years. Expert in React. No backend or fullstack.",
            "relevance_score": 40
        },

        # Example: Senior Marketing Specialist
        {
            "jd_text": "Senior Marketing Specialist. Skills: Digital Campaign Management, SEO/SEM Strategy, Content Marketing, Email Marketing Automation, Google Analytics (Advanced), CRM Integration. Experience: 4+ years.",
            "resume_text": "Senior Marketing Specialist, 5 years. Led digital campaign management. Developed and executed SEO/SEM strategies. Managed content marketing initiatives. Implemented email marketing automation. Expert in advanced Google Analytics and CRM integration.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Marketing Specialist. Skills: Digital Campaign Management, SEO/SEM Strategy, Content Marketing, Email Marketing Automation, Google Analytics (Advanced), CRM Integration. Experience: 4+ years.",
            "resume_text": "Marketing Specialist, 2 years. Ran social media ads. Some content. No strategic campaign management.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Marketing Specialist. Skills: Digital Campaign Management, SEO/SEM Strategy, Content Marketing, Email Marketing Automation, Google Analytics (Advanced), CRM Integration. Experience: 4+ years.",
            "resume_text": "Sales Manager, 10 years. No marketing expertise.",
            "relevance_score": 20
        },

        # Example: Senior HR Generalist
        {
            "jd_text": "Senior HR Generalist. Skills: Employee Relations, Performance Management, HR Policy Development, Benefits Administration, HRIS (Workday/SAP), Compliance, Training & Development. Experience: 5+ years.",
            "resume_text": "Senior HR Generalist, 6 years. Managed complex employee relations cases. Oversaw performance management cycles. Developed HR policies. Administered benefits. Proficient in Workday HRIS. Ensured compliance. Led training and development programs.",
            "relevance_score": 93
        },
        {
            "jd_text": "Senior HR Generalist. Skills: Employee Relations, Performance Management, HR Policy Development, Benefits Administration, HRIS (Workday/SAP), Compliance, Training & Development. Experience: 5+ years.",
            "resume_text": "HR Generalist, 2 years. Handled onboarding. Some employee relations. No policy development or training leadership.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior HR Generalist. Skills: Employee Relations, Performance Management, HR Policy Development, Benefits Administration, HRIS (Workday/SAP), Compliance, Training & Development. Experience: 5+ years.",
            "resume_text": "Office Manager, 10 years. Managed administrative tasks. No HR specialization.",
            "relevance_score": 30
        },

        # Example: Senior Cloud Engineer
        {
            "jd_text": "Senior Cloud Engineer. Skills: AWS/Azure/GCP, Infrastructure as Code (Terraform, CloudFormation), Containerization (Docker, Kubernetes), Serverless, CI/CD, Cost Optimization. Experience: 4+ years.",
            "resume_text": "Senior Cloud Engineer, 5 years. Designed and implemented cloud infrastructure on AWS. Expert in Terraform and CloudFormation. Managed Docker and Kubernetes. Developed serverless applications. Built CI/CD pipelines. Optimized cloud costs.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Cloud Engineer. Skills: AWS/Azure/GCP, Infrastructure as Code (Terraform, CloudFormation), Containerization (Docker, Kubernetes), Serverless, CI/CD, Cost Optimization. Experience: 4+ years.",
            "resume_text": "Cloud Engineer, 2 years. Deployed basic resources. Some Docker. No advanced IaC or cost optimization.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Cloud Engineer. Skills: AWS/Azure/GCP, Infrastructure as Code (Terraform, CloudFormation), Containerization (Docker, Kubernetes), Serverless, CI/CD, Cost Optimization. Experience: 4+ years.",
            "resume_text": "System Administrator, 8 years. Managed on-premise servers. No cloud deployment.",
            "relevance_score": 30
        },

        # Example: Senior Cybersecurity Engineer
        {
            "jd_text": "Senior Cybersecurity Engineer. Skills: Security Architecture, Penetration Testing, Vulnerability Management, SIEM/SOAR, Cloud Security, Incident Response, Python Scripting. Experience: 5+ years.",
            "resume_text": "Senior Cybersecurity Engineer, 6 years. Designed and implemented security architectures. Conducted penetration tests and managed vulnerabilities. Proficient in SIEM/SOAR. Secured cloud environments. Led incident response. Wrote Python scripts for automation.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Cybersecurity Engineer. Skills: Security Architecture, Penetration Testing, Vulnerability Management, SIEM/SOAR, Cloud Security, Incident Response, Python Scripting. Experience: 5+ years.",
            "resume_text": "Cybersecurity Analyst, 3 years. Monitored alerts. Some vulnerability scanning. No architecture or penetration testing.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Cybersecurity Engineer. Skills: Security Architecture, Penetration Testing, Vulnerability Management, SIEM/SOAR, Cloud Security, Incident Response, Python Scripting. Experience: 5+ years.",
            "resume_text": "Network Engineer, 10 years. Managed firewalls. No deep cybersecurity engineering.",
            "relevance_score": 35
        },

        # Example: Senior Mobile App Developer (Cross-Platform)
        {
            "jd_text": "Senior Mobile App Developer (Cross-Platform). Skills: React Native/Flutter, iOS/Android Development, RESTful APIs, UI/UX, Performance Optimization, State Management. Experience: 4+ years.",
            "resume_text": "Senior Mobile App Developer, 5 years. Built high-performance cross-platform apps with React Native. Expertise in native iOS/Android development. Integrated complex RESTful APIs. Strong in UI/UX and performance optimization. Managed state effectively.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Mobile App Developer (Cross-Platform). Skills: React Native/Flutter, iOS/Android Development, RESTful APIs, UI/UX, Performance Optimization, State Management. Experience: 4+ years.",
            "resume_text": "Mobile App Developer (Native), 3 years. Focused on iOS. Some API integration. No cross-platform.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Mobile App Developer (Cross-Platform). Skills: React Native/Flutter, iOS/Android Development, RESTful APIs, UI/UX, Performance Optimization, State Management. Experience: 4+ years.",
            "resume_text": "Frontend Web Developer, 6 years. Proficient in React. No mobile development.",
            "relevance_score": 40
        },

        # Example: Senior Business Development Manager
        {
            "jd_text": "Senior Business Development Manager. Skills: Strategic Sales, Market Expansion, Partnership Development, Negotiation, CRM (Salesforce), P&L Responsibility. Experience: 6+ years.",
            "resume_text": "Senior Business Development Manager, 7 years. Developed and executed strategic sales plans. Drove market expansion. Forged key partnerships. Expert negotiator. Proficient in Salesforce CRM. Managed P&L for new ventures.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Business Development Manager. Skills: Strategic Sales, Market Expansion, Partnership Development, Negotiation, CRM (Salesforce), P&L Responsibility. Experience: 6+ years.",
            "resume_text": "Business Development Manager, 3 years. Generated leads. Some negotiation. No strategic or P&L.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Business Development Manager. Skills: Strategic Sales, Market Expansion, Partnership Development, Negotiation, CRM (Salesforce), P&L Responsibility. Experience: 6+ years.",
            "resume_text": "Marketing Manager, 8 years. Focused on brand. No sales or business development.",
            "relevance_score": 25
        },

        # Example: Senior Project Coordinator
        {
            "jd_text": "Senior Project Coordinator. Skills: Project Lifecycle Management, Stakeholder Communication, Risk Mitigation, Resource Tracking, Reporting, Jira/Asana. Experience: 4+ years.",
            "resume_text": "Senior Project Coordinator, 5 years. Managed full project lifecycle. Expert in stakeholder communication and risk mitigation. Tracked resources and provided detailed reports. Proficient in Jira and Asana.",
            "relevance_score": 92
        },
        {
            "jd_text": "Senior Project Coordinator. Skills: Project Lifecycle Management, Stakeholder Communication, Risk Mitigation, Resource Tracking, Reporting, Jira/Asana. Experience: 4+ years.",
            "resume_text": "Project Coordinator, 2 years. Assisted with schedules. Some reporting. No full lifecycle or risk mitigation.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Project Coordinator. Skills: Project Lifecycle Management, Stakeholder Communication, Risk Mitigation, Resource Tracking, Reporting, Jira/Asana. Experience: 4+ years.",
            "resume_text": "Administrative Manager, 10 years. Managed office operations. No project coordination.",
            "relevance_score": 30
        },

        # Example: Senior Technical Writer
        {
            "jd_text": "Senior Technical Writer. Skills: Complex Technical Documentation, API Documentation, DITA/XML, Content Strategy, Information Architecture, Cross-functional Collaboration. Experience: 5+ years.",
            "resume_text": "Senior Technical Writer, 6 years. Produced complex technical documentation and API docs. Expert in DITA XML. Developed content strategies and information architecture. Collaborated extensively with engineering teams.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Technical Writer. Skills: Complex Technical Documentation, API Documentation, DITA/XML, Content Strategy, Information Architecture, Cross-functional Collaboration. Experience: 5+ years.",
            "resume_text": "Technical Writer, 2 years. Wrote user guides. Some API docs. No complex DITA or content strategy.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Technical Writer. Skills: Complex Technical Documentation, API Documentation, DITA/XML, Content Strategy, Information Architecture, Cross-functional Collaboration. Experience: 5+ years.",
            "resume_text": "Copywriter, 8 years. Wrote marketing copy. No technical documentation.",
            "relevance_score": 25
        },

        # Example: Senior Sales Representative
        {
            "jd_text": "Senior Sales Representative. Skills: B2B Sales, Lead Generation, CRM (Salesforce), Negotiation, Quota Attainment, Client Relationship Management. Experience: 3+ years.",
            "resume_text": "Senior Sales Representative, 4 years. Consistently exceeded B2B sales quotas. Expert in lead generation. Proficient in Salesforce CRM. Master negotiator. Built strong client relationships.",
            "relevance_score": 93
        },
        {
            "jd_text": "Senior Sales Representative. Skills: B2B Sales, Lead Generation, CRM (Salesforce), Negotiation, Quota Attainment, Client Relationship Management. Experience: 3+ years.",
            "resume_text": "Sales Representative, 1 year. Achieved some sales. No senior-level quota or B2B focus.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Sales Representative. Skills: B2B Sales, Lead Generation, CRM (Salesforce), Negotiation, Quota Attainment, Client Relationship Management. Experience: 3+ years.",
            "resume_text": "Customer Service Manager, 7 years. Managed customer support. No direct sales.",
            "relevance_score": 30
        },

        # Example: Senior Customer Support Specialist
        {
            "jd_text": "Senior Customer Support Specialist. Skills: Advanced Troubleshooting, Escalation Management, Product Expertise, CRM (Zendesk/ServiceNow), Knowledge Base Management. Experience: 4+ years.",
            "resume_text": "Senior Customer Support Specialist, 5 years. Provided advanced troubleshooting for complex product issues. Managed escalated cases. Deep product expertise. Proficient in Zendesk. Maintained knowledge base articles.",
            "relevance_score": 92
        },
        {
            "jd_text": "Senior Customer Support Specialist. Skills: Advanced Troubleshooting, Escalation Management, Product Expertise, CRM (Zendesk/ServiceNow), Knowledge Base Management. Experience: 4+ years.",
            "resume_text": "Customer Support Specialist, 2 years. Handled routine inquiries. No advanced troubleshooting or escalation.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Customer Support Specialist. Skills: Advanced Troubleshooting, Escalation Management, Product Expertise, CRM (Zendesk/ServiceNow), Knowledge Base Management. Experience: 4+ years.",
            "resume_text": "IT Helpdesk, 6 years. Provided basic IT support. No customer support specific role.",
            "relevance_score": 40
        },

        # Example: Senior Operations Coordinator
        {
            "jd_text": "Senior Operations Coordinator. Skills: Process Optimization, Logistics Coordination, Inventory Management (Advanced), Vendor Management, ERP Systems, Reporting. Experience: 4+ years.",
            "resume_text": "Senior Operations Coordinator, 5 years. Led process optimization initiatives. Coordinated complex logistics. Managed advanced inventory systems. Oversaw vendor relationships. Proficient in ERP systems and detailed reporting.",
            "relevance_score": 93
        },
        {
            "jd_text": "Senior Operations Coordinator. Skills: Process Optimization, Logistics Coordination, Inventory Management (Advanced), Vendor Management, ERP Systems, Reporting. Experience: 4+ years.",
            "resume_text": "Operations Coordinator, 2 years. Assisted with daily tasks. Some inventory. No optimization or vendor management.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Operations Coordinator. Skills: Process Optimization, Logistics Coordination, Inventory Management (Advanced), Vendor Management, ERP Systems, Reporting. Experience: 4+ years.",
            "resume_text": "Administrative Assistant, 8 years. Managed office supplies. No operations focus.",
            "relevance_score": 25
        },
        {
            "jd_text": "Senior Product Manager. Skills: Product Strategy, Technical Product Management, Agile, SaaS, API Products, User Experience, Market Analysis. Experience: 6+ years.",
            "resume_text": "Senior Product Manager with 8 years experience. Led product strategy for complex SaaS platforms. Deep expertise in technical product management and API products. Drove user experience initiatives. Strong market analysis skills. Agile advocate.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Product Manager. Skills: Product Strategy, Technical Product Management, Agile, SaaS, API Products, User Experience, Market Analysis. Experience: 6+ years.",
            "resume_text": "Product Manager, 4 years. Managed product backlog. Some experience with user stories. No technical product depth or senior strategy.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Product Manager. Skills: Product Strategy, Technical Product Management, Agile, SaaS, API Products, User Experience, Market Analysis. Experience: 6+ years.",
            "resume_text": "Software Engineer, 10 years. Built complex software. No product management experience.",
            "relevance_score": 30
        },

        # Example: Data Engineer (Mid-level)
        {
            "jd_text": "Data Engineer. Skills: ETL Pipelines, SQL, Python, Spark, Cloud (AWS/Azure), Data Warehousing, Data Lake. Experience: 3+ years.",
            "resume_text": "Data Engineer with 4 years experience. Designed and built ETL pipelines using Python and Spark. Proficient in SQL for data manipulation. Worked with data warehouses and data lakes on AWS. Strong data modeling.",
            "relevance_score": 90
        },
        {
            "jd_text": "Data Engineer. Skills: ETL Pipelines, SQL, Python, Spark, Cloud (AWS/Azure), Data Warehousing, Data Lake. Experience: 3+ years.",
            "resume_text": "Data Analyst, 5 years. Strong SQL and Tableau. Some Python. No experience with Spark or large-scale ETL architecture.",
            "relevance_score": 70
        },
        {
            "jd_text": "Data Engineer. Skills: ETL Pipelines, SQL, Python, Spark, Cloud (AWS/Azure), Data Warehousing, Data Lake. Experience: 3+ years.",
            "resume_text": "Web Developer, 6 years. Built web applications. No data engineering focus.",
            "relevance_score": 25
        },

        # Example: Senior Marketing Specialist (SEO/SEM)
        {
            "jd_text": "Senior Marketing Specialist (SEO/SEM). Skills: SEO Strategy, SEM Campaign Management, Google Ads, Google Analytics, Keyword Research, A/B Testing. Experience: 4+ years.",
            "resume_text": "Senior Marketing Specialist, 5 years. Developed and executed comprehensive SEO strategies. Managed large-scale SEM campaigns on Google Ads. Expert in Google Analytics. Led keyword research and A/B testing initiatives.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Marketing Specialist (SEO/SEM). Skills: SEO Strategy, SEM Campaign Management, Google Ads, Google Analytics, Keyword Research, A/B Testing. Experience: 4+ years.",
            "resume_text": "Marketing Coordinator, 2 years. Assisted with social media. Some Google Analytics. No strategic SEO/SEM.",
            "relevance_score": 50
        },
        {
            "jd_text": "Senior Marketing Specialist (SEO/SEM). Skills: SEO Strategy, SEM Campaign Management, Google Ads, Google Analytics, Keyword Research, A/B Testing. Experience: 4+ years.",
            "resume_text": "Content Writer, 7 years. Wrote blog posts. Some SEO awareness. No technical SEO/SEM.",
            "relevance_score": 30
        },

        # Example: Financial Controller (Corporate)
        {
            "jd_text": "Financial Controller. Skills: Financial Reporting, GAAP/IFRS, Budgeting, Forecasting, Internal Controls, Team Leadership, ERP Systems (SAP/Oracle). Experience: 8+ years.",
            "resume_text": "Financial Controller, 10 years. Oversaw all financial reporting adhering to GAAP. Led budgeting and forecasting processes. Strengthened internal controls. Managed a team of accountants. Proficient in SAP ERP.",
            "relevance_score": 95
        },
        {
            "jd_text": "Financial Controller. Skills: Financial Reporting, GAAP/IFRS, Budgeting, Forecasting, Internal Controls, Team Leadership, ERP Systems (SAP/Oracle). Experience: 8+ years.",
            "resume_text": "Senior Accountant, 6 years. Prepared financial statements. Some budgeting. No leadership or full controller duties.",
            "relevance_score": 65
        },
        {
            "jd_text": "Financial Controller. Skills: Financial Reporting, GAAP/IFRS, Budgeting, Forecasting, Internal Controls, Team Leadership, ERP Systems (SAP/Oracle). Experience: 8+ years.",
            "resume_text": "Financial Analyst, 5 years. Built financial models. No accounting or control experience.",
            "relevance_score": 35
        },

        # Example: HR Business Partner (Tech)
        {
            "jd_text": "HR Business Partner. Skills: Employee Relations, Talent Management, Performance Management, Compensation, HRIS (Workday), Coaching, Tech Industry Experience. Experience: 5+ years.",
            "resume_text": "HR Business Partner with 6 years. Partnered with tech leaders on employee relations and talent management. Implemented performance management programs. Advised on compensation. Proficient in Workday. Provided coaching.",
            "relevance_score": 93
        },
        {
            "jd_text": "HR Business Partner. Skills: Employee Relations, Talent Management, Performance Management, Compensation, HRIS (Workday), Coaching, Tech Industry Experience. Experience: 5+ years.",
            "resume_text": "HR Generalist, 3 years. Handled onboarding. Some employee relations. No strategic partnering.",
            "relevance_score": 60
        },
        {
            "jd_text": "HR Business Partner. Skills: Employee Relations, Talent Management, Performance Management, Compensation, HRIS (Workday), Coaching, Tech Industry Experience. Experience: 5+ years.",
            "resume_text": "Recruiter, 7 years. Focused on talent acquisition. No broader HRBP scope.",
            "relevance_score": 40
        },

        # Example: Senior Cloud Engineer (AWS Focus)
        {
            "jd_text": "Senior Cloud Engineer. Skills: AWS (EC2, S3, RDS, Lambda, VPC), Terraform, CloudFormation, CI/CD, Containerization (Docker/ECS), Python Scripting. Experience: 5+ years.",
            "resume_text": "Senior Cloud Engineer, 6 years. Designed and deployed scalable solutions on AWS (EC2, S3, RDS, Lambda). Wrote IaC with Terraform and CloudFormation. Automated CI/CD pipelines. Managed Docker containers on ECS. Proficient in Python scripting.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Cloud Engineer. Skills: AWS (EC2, S3, RDS, Lambda, VPC), Terraform, CloudFormation, CI/CD, Containerization (Docker/ECS), Python Scripting. Experience: 5+ years.",
            "resume_text": "DevOps Engineer, 3 years. Used Docker and Jenkins. Some AWS. No senior-level AWS services or IaC depth.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Cloud Engineer. Skills: AWS (EC2, S3, RDS, Lambda, VPC), Terraform, CloudFormation, CI/CD, Containerization (Docker/ECS), Python Scripting. Experience: 5+ years.",
            "resume_text": "System Administrator, 8 years. Managed on-premise Linux. No cloud experience.",
            "relevance_score": 30
        },

        # Example: Cybersecurity Architect
        {
            "jd_text": "Cybersecurity Architect. Skills: Security Architecture, Enterprise Security, Cloud Security, Zero Trust, Network Security, Data Security, Compliance (NIST, ISO). Experience: 8+ years.",
            "resume_text": "Cybersecurity Architect, 9 years. Designed and implemented enterprise security architectures. Expertise in cloud security and Zero Trust principles. Strong in network and data security. Ensured compliance with NIST and ISO standards.",
            "relevance_score": 97
        },
        {
            "jd_text": "Cybersecurity Architect. Skills: Security Architecture, Enterprise Security, Cloud Security, Zero Trust, Network Security, Data Security, Compliance (NIST, ISO). Experience: 8+ years.",
            "resume_text": "Cybersecurity Engineer, 5 years. Implemented security controls. Some cloud security. No architecture design or enterprise scope.",
            "relevance_score": 65
        },
        {
            "jd_text": "Cybersecurity Architect. Skills: Security Architecture, Enterprise Security, Cloud Security, Zero Trust, Network Security, Data Security, Compliance (NIST, ISO). Experience: 8+ years.",
            "resume_text": "IT Manager, 12 years. Managed IT operations. Some security oversight. No dedicated security architecture.",
            "relevance_score": 40
        },

        # Example: Senior Project Manager (IT)
        {
            "jd_text": "Senior Project Manager (IT). Skills: Project Management, Agile/Scrum, SDLC, Budget Management, Risk Management, Stakeholder Communication, Software Implementation. Experience: 7+ years.",
            "resume_text": "Senior IT Project Manager, 8 years. Led complex software implementation projects. Proficient in Agile/Scrum. Managed project budgets and risks. Excellent stakeholder communication. Deep understanding of SDLC.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Project Manager (IT). Skills: Project Management, Agile/Scrum, SDLC, Budget Management, Risk Management, Stakeholder Communication, Software Implementation. Experience: 7+ years.",
            "resume_text": "Project Coordinator, 4 years. Assisted with project tasks. Some Agile. No senior leadership or budget management.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Project Manager (IT). Skills: Project Management, Agile/Scrum, SDLC, Budget Management, Risk Management, Stakeholder Communication, Software Implementation. Experience: 7+ years.",
            "resume_text": "Business Analyst, 6 years. Gathered requirements. No project management leadership.",
            "relevance_score": 30
        },

        # Example: Marketing Operations Manager
        {
            "jd_text": "Marketing Operations Manager. Skills: Marketing Automation (Marketo/Pardot), CRM (Salesforce), Lead Management, Data Analytics, Process Optimization. Experience: 4+ years.",
            "resume_text": "Marketing Operations Manager, 5 years. Managed marketing automation platforms (Marketo). Integrated with Salesforce CRM. Optimized lead management processes. Utilized data analytics to improve campaign performance.",
            "relevance_score": 93
        },
        {
            "jd_text": "Marketing Operations Manager. Skills: Marketing Automation (Marketo/Pardot), CRM (Salesforce), Lead Management, Data Analytics, Process Optimization. Experience: 4+ years.",
            "resume_text": "Marketing Specialist, 3 years. Ran email campaigns. Some CRM exposure. No operations or automation platform management.",
            "relevance_score": 60
        },
        {
            "jd_text": "Marketing Operations Manager. Skills: Marketing Automation (Marketo/Pardot), CRM (Salesforce), Lead Management, Data Analytics, Process Optimization. Experience: 4+ years.",
            "resume_text": "Sales Operations Analyst, 6 years. Focused on sales data. No marketing operations.",
            "relevance_score": 40
        },

        # Example: Senior Accountant
        {
            "jd_text": "Senior Accountant. Skills: General Ledger, Month-End Close, Financial Reporting, Reconciliation, GAAP, ERP Systems (NetSuite/QuickBooks). Experience: 4+ years.",
            "resume_text": "Senior Accountant, 5 years. Managed general ledger, led month-end close processes. Prepared accurate financial reports. Performed complex reconciliations. Ensured GAAP compliance. Proficient in NetSuite.",
            "relevance_score": 90
        },
        {
            "jd_text": "Senior Accountant. Skills: General Ledger, Month-End Close, Financial Reporting, Reconciliation, GAAP, ERP Systems (NetSuite/QuickBooks). Experience: 4+ years.",
            "resume_text": "Staff Accountant, 2 years. Assisted with reconciliations. No leadership or full month-end close.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Accountant. Skills: General Ledger, Month-End Close, Financial Reporting, Reconciliation, GAAP, ERP Systems (NetSuite/QuickBooks). Experience: 4+ years.",
            "resume_text": "Financial Analyst, 3 years. Built financial models. No accounting operations.",
            "relevance_score": 30
        },

        # Example: Talent Management Specialist
        {
            "jd_text": "Talent Management Specialist. Skills: Performance Management, Learning & Development, Succession Planning, Employee Engagement, HRIS. Experience: 3+ years.",
            "resume_text": "Talent Management Specialist, 4 years. Designed and implemented performance management systems. Developed learning & development programs. Supported succession planning. Drove employee engagement initiatives. Proficient in HRIS.",
            "relevance_score": 92
        },
        {
            "jd_text": "Talent Management Specialist. Skills: Performance Management, Learning & Development, Succession Planning, Employee Engagement, HRIS. Experience: 3+ years.",
            "resume_text": "HR Coordinator, 2 years. Assisted with HR tasks. No talent management focus.",
            "relevance_score": 50
        },
        {
            "jd_text": "Talent Management Specialist. Skills: Performance Management, Learning & Development, Succession Planning, Employee Engagement, HRIS. Experience: 3+ years.",
            "resume_text": "Recruiter, 5 years. Focused on hiring. No post-hire talent development.",
            "relevance_score": 35
        },

        # Example: Senior Game Developer (Unreal Engine)
        {
            "jd_text": "Senior Game Developer (Unreal Engine). Skills: Unreal Engine 5, C++, Blueprints, Game Physics, AI Programming, Multiplayer Networking. Experience: 5+ years.",
            "resume_text": "Senior Game Developer, 6 years. Developed games in Unreal Engine 5 using C++ and Blueprints. Expertise in game physics and AI programming. Implemented multiplayer networking. Shipped 2 AAA titles.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior Game Developer (Unreal Engine). Skills: Unreal Engine 5, C++, Blueprints, Game Physics, AI Programming, Multiplayer Networking. Experience: 5+ years.",
            "resume_text": "Game Developer (Unity), 4 years. Proficient in Unity/C#. No Unreal Engine experience.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Game Developer (Unreal Engine). Skills: Unreal Engine 5, C++, Blueprints, Game Physics, AI Programming, Multiplayer Networking. Experience: 5+ years.",
            "resume_text": "Software Engineer, 8 years. Strong in C++. No game development.",
            "relevance_score": 30
        },

        # Example: E-commerce Specialist
        {
            "jd_text": "E-commerce Specialist. Skills: Product Listing Optimization, Shopify, SEO, Digital Marketing, Inventory Management, Customer Service. Experience: 2+ years.",
            "resume_text": "E-commerce Specialist, 3 years. Optimized product listings on Shopify. Implemented SEO strategies for online store. Assisted with digital marketing campaigns. Managed online inventory and customer service inquiries.",
            "relevance_score": 90
        },
        {
            "jd_text": "E-commerce Specialist. Skills: Product Listing Optimization, Shopify, SEO, Digital Marketing, Inventory Management, Customer Service. Experience: 2+ years.",
            "resume_text": "Retail Sales Associate, 4 years. Managed store sales. No e-commerce platform experience.",
            "relevance_score": 35
        },
        {
            "jd_text": "E-commerce Specialist. Skills: Product Listing Optimization, Shopify, SEO, Digital Marketing, Inventory Management, Customer Service. Experience: 2+ years.",
            "resume_text": "Marketing Coordinator, 1 year. Some digital marketing. No e-commerce specific skills.",
            "relevance_score": 55
        },

        # Example: Data Quality Analyst
        {
            "jd_text": "Data Quality Analyst. Skills: Data Profiling, Data Cleansing, SQL, Data Governance, Root Cause Analysis, Data Stewardship. Experience: 3+ years.",
            "resume_text": "Data Quality Analyst, 4 years. Performed data profiling and cleansing. Wrote complex SQL queries for data quality checks. Supported data governance initiatives. Conducted root cause analysis for data issues. Acted as data steward.",
            "relevance_score": 93
        },
        {
            "jd_text": "Data Quality Analyst. Skills: Data Profiling, Data Cleansing, SQL, Data Governance, Root Cause Analysis, Data Stewardship. Experience: 3+ years.",
            "resume_text": "Data Analyst, 2 years. Strong in SQL. Some data cleaning. No dedicated data quality role.",
            "relevance_score": 60
        },
        {
            "jd_text": "Data Quality Analyst. Skills: Data Profiling, Data Cleansing, SQL, Data Governance, Root Cause Analysis, Data Stewardship. Experience: 3+ years.",
            "resume_text": "Business Analyst, 5 years. Focused on requirements. No data quality focus.",
            "relevance_score": 30
        },

        # Example: AI/ML Research Engineer
        {
            "jd_text": "AI/ML Research Engineer. Skills: Machine Learning, Deep Learning, Python, TensorFlow/PyTorch, Algorithm Development, Research Publications. Experience: PhD + 2 years research.",
            "resume_text": "AI/ML Research Engineer, PhD + 3 years research. Developed novel ML/DL algorithms in Python with PyTorch. Contributed to research publications. Strong in experimental design and model evaluation.",
            "relevance_score": 96
        },
        {
            "jd_text": "AI/ML Research Engineer. Skills: Machine Learning, Deep Learning, Python, TensorFlow/PyTorch, Algorithm Development, Research Publications. Experience: PhD + 2 years research.",
            "resume_text": "ML Engineer, 4 years. Deployed ML models. Some deep learning. No research or publication focus.",
            "relevance_score": 70
        },
        {
            "jd_text": "AI/ML Research Engineer. Skills: Machine Learning, Deep Learning, Python, TensorFlow/PyTorch, Algorithm Development, Research Publications. Experience: PhD + 2 years research.",
            "resume_text": "Software Engineer, 7 years. Strong in Python. No ML/AI research.",
            "relevance_score": 25
        },

        # Example: Chief Operating Officer (COO)
        {
            "jd_text": "Chief Operating Officer (COO). Skills: Operations Strategy, P&L Management, Process Improvement, Supply Chain, Team Leadership, Business Growth. Experience: 15+ years.",
            "resume_text": "COO with 20 years experience. Developed and executed operations strategy. Managed multi-million dollar P&L. Drove significant process improvements. Oversaw global supply chain. Led large, diverse teams to achieve business growth.",
            "relevance_score": 98
        },
        {
            "jd_text": "Chief Operating Officer (COO). Skills: Operations Strategy, P&L Management, Process Improvement, Supply Chain, Team Leadership, Business Growth. Experience: 15+ years.",
            "resume_text": "Operations Manager, 10 years. Managed daily operations. Some process improvement. No executive leadership or P&L responsibility.",
            "relevance_score": 60
        },
        {
            "jd_text": "Chief Operating Officer (COO). Skills: Operations Strategy, P&L Management, Process Improvement, Supply Chain, Team Leadership, Business Growth. Experience: 15+ years.",
            "resume_text": "Project Manager, 18 years. Managed large projects. No operations strategy or P&L.",
            "relevance_score": 30
        },

        # Example: Director of Product
        {
            "jd_text": "Director of Product. Skills: Product Leadership, Product Strategy, Roadmap Development, Market Analysis, Team Management, Cross-functional Leadership. Experience: 10+ years.",
            "resume_text": "Director of Product, 12 years. Defined and executed product strategy across multiple product lines. Developed compelling roadmaps based on deep market analysis. Managed and mentored product teams. Drove cross-functional alignment.",
            "relevance_score": 97
        },
        {
            "jd_text": "Director of Product. Skills: Product Leadership, Product Strategy, Roadmap Development, Market Analysis, Team Management, Cross-functional Leadership. Experience: 10+ years.",
            "resume_text": "Senior Product Manager, 7 years. Managed specific products. Some team leadership. No director-level strategic oversight.",
            "relevance_score": 70
        },
        {
            "jd_text": "Director of Product. Skills: Product Leadership, Product Strategy, Roadmap Development, Market Analysis, Team Management, Cross-functional Leadership. Experience: 10+ years.",
            "resume_text": "Marketing Director, 15 years. Led marketing teams. No product management.",
            "relevance_score": 40
        },

        # Example: Head of Engineering
        {
            "jd_text": "Head of Engineering. Skills: Engineering Management, Software Architecture, Scalability, Technical Strategy, Budget Management, Talent Development. Experience: 12+ years.",
            "resume_text": "Head of Engineering, 15 years. Led engineering organization. Defined technical strategy and software architecture for highly scalable systems. Managed large engineering budgets. Focused on talent development and retention.",
            "relevance_score": 98
        },
        {
            "jd_text": "Head of Engineering. Skills: Engineering Management, Software Architecture, Scalability, Technical Strategy, Budget Management, Talent Development. Experience: 12+ years.",
            "resume_text": "Director of Engineering, 8 years. Managed engineering teams. Some architecture. No head-of-department strategic role.",
            "relevance_score": 75
        },
        {
            "jd_text": "Head of Engineering. Skills: Engineering Management, Software Architecture, Scalability, Technical Strategy, Budget Management, Talent Development. Experience: 12+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Strong technical contributor. No management experience.",
            "relevance_score": 45
        },

        # Example: Senior Frontend Engineer (Vue.js)
        {
            "jd_text": "Senior Frontend Engineer. Skills: Vue.js, JavaScript, TypeScript, Nuxt.js, State Management (Vuex), Component Libraries, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Senior Frontend Engineer, 6 years. Developed complex SPAs with Vue.js and Nuxt.js. Proficient in JavaScript/TypeScript and Vuex for state management. Built reusable component libraries. Focused on performance optimization.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Frontend Engineer. Skills: Vue.js, JavaScript, TypeScript, Nuxt.js, State Management (Vuex), Component Libraries, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Frontend Developer (React), 4 years. Strong in React. No Vue.js experience.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Frontend Engineer. Skills: Vue.js, JavaScript, TypeScript, Nuxt.js, State Management (Vuex), Component Libraries, Performance Optimization. Experience: 5+ years.",
            "resume_text": "Backend Engineer, 8 years. No frontend experience.",
            "relevance_score": 20
        },

        # Example: Senior HR Manager
        {
            "jd_text": "Senior HR Manager. Skills: Strategic HR, Employee Relations, Compensation & Benefits, HRIS Implementation, Compliance, Team Leadership. Experience: 7+ years.",
            "resume_text": "Senior HR Manager, 8 years. Led strategic HR initiatives. Managed complex employee relations cases. Oversaw compensation and benefits programs. Led HRIS implementation projects. Ensured compliance. Managed HR team.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior HR Manager. Skills: Strategic HR, Employee Relations, Compensation & Benefits, HRIS Implementation, Compliance, Team Leadership. Experience: 7+ years.",
            "resume_text": "HR Generalist, 4 years. Handled daily HR tasks. Some employee relations. No strategic or leadership role.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior HR Manager. Skills: Strategic HR, Employee Relations, Compensation & Benefits, HRIS Implementation, Compliance, Team Leadership. Experience: 7+ years.",
            "resume_text": "Office Manager, 10 years. Managed administrative staff. No HR expertise.",
            "relevance_score": 30
        },

        # Example: Senior UX Researcher
        {
            "jd_text": "Senior UX Researcher. Skills: Quantitative & Qualitative Research, Usability Testing, A/B Testing, Survey Design, Statistical Analysis (R/Python), Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Senior UX Researcher, 6 years. Led mixed-methods research (quantitative & qualitative). Designed and executed usability and A/B tests. Proficient in survey design and statistical analysis using R. Managed research projects and presented to stakeholders.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior UX Researcher. Skills: Quantitative & Qualitative Research, Usability Testing, A/B Testing, Survey Design, Statistical Analysis (R/Python), Stakeholder Management. Experience: 5+ years.",
            "resume_text": "UX Researcher, 3 years. Conducted user interviews. Some usability testing. No senior-level research or statistical depth.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior UX Researcher. Skills: Quantitative & Qualitative Research, Usability Testing, A/B Testing, Survey Design, Statistical Analysis (R/Python), Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Market Research Analyst, 8 years. Designed surveys. No UX research focus.",
            "relevance_score": 40
        },

        # Example: Machine Learning Engineer (Senior)
        {
            "jd_text": "Senior Machine Learning Engineer. Skills: MLOps, Production ML Systems, Distributed Computing, Python, TensorFlow/PyTorch, AWS/GCP ML Services, Model Deployment. Experience: 6+ years.",
            "resume_text": "Senior ML Engineer, 7 years. Designed and built production-grade ML systems. Expertise in MLOps and distributed computing. Proficient in Python with TensorFlow. Deployed models using AWS SageMaker. Strong in model deployment and monitoring.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior Machine Learning Engineer. Skills: MLOps, Production ML Systems, Distributed Computing, Python, TensorFlow/PyTorch, AWS/GCP ML Services, Model Deployment. Experience: 6+ years.",
            "resume_text": "Machine Learning Engineer, 3 years. Built ML models. Some Python. No MLOps or production experience.",
            "relevance_score": 75
        },
        {
            "jd_text": "Senior Machine Learning Engineer. Skills: MLOps, Production ML Systems, Distributed Computing, Python, TensorFlow/PyTorch, AWS/GCP ML Services, Model Deployment. Experience: 6+ years.",
            "resume_text": "Data Scientist, 8 years. Focused on model development. No engineering or MLOps.",
            "relevance_score": 50
        },

        # Example: Cloud Solutions Architect (Senior)
        {
            "jd_text": "Senior Cloud Solutions Architect. Skills: Enterprise Cloud Strategy, AWS/Azure/GCP, Hybrid Cloud, Microservices Architecture, Cost Optimization, Security Best Practices. Experience: 8+ years.",
            "resume_text": "Senior Cloud Solutions Architect, 9 years. Developed enterprise cloud strategies. Designed complex hybrid cloud solutions on AWS and Azure. Architected microservices. Focused on cost optimization and security best practices.",
            "relevance_score": 98
        },
        {
            "jd_text": "Senior Cloud Solutions Architect. Skills: Enterprise Cloud Strategy, AWS/Azure/GCP, Hybrid Cloud, Microservices Architecture, Cost Optimization, Security Best Practices. Experience: 8+ years.",
            "resume_text": "Cloud Engineer, 5 years. Deployed cloud resources. Some architecture. No enterprise strategy.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Cloud Solutions Architect. Skills: Enterprise Cloud Strategy, AWS/Azure/GCP, Hybrid Cloud, Microservices Architecture, Cost Optimization, Security Best Practices. Experience: 8+ years.",
            "resume_text": "IT Manager, 12 years. Managed IT infrastructure. No cloud architecture.",
            "relevance_score": 40
        },

        # Example: Principal Cybersecurity Engineer
        {
            "jd_text": "Principal Cybersecurity Engineer. Skills: Advanced Threat Detection, Incident Response, Forensics, Security Automation, Cloud Security, Red Teaming, Leadership. Experience: 8+ years.",
            "resume_text": "Principal Cybersecurity Engineer, 10 years. Led advanced threat detection and incident response. Conducted forensics investigations. Implemented security automation. Expertise in cloud security and red teaming. Provided technical leadership.",
            "relevance_score": 97
        },
        {
            "jd_text": "Principal Cybersecurity Engineer. Skills: Advanced Threat Detection, Incident Response, Forensics, Security Automation, Cloud Security, Red Teaming, Leadership. Experience: 8+ years.",
            "resume_text": "Cybersecurity Engineer, 5 years. Handled incident response. Some automation. No principal-level or red teaming.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Cybersecurity Engineer. Skills: Advanced Threat Detection, Incident Response, Forensics, Security Automation, Cloud Security, Red Teaming, Leadership. Experience: 8+ years.",
            "resume_text": "Network Security Analyst, 7 years. Managed firewalls. No advanced threat detection.",
            "relevance_score": 45
        },

        # Example: Senior Business Analyst (Financial Services)
        {
            "jd_text": "Senior Business Analyst (Financial Services). Skills: Requirements Elicitation, Process Modeling, Agile, SQL, Financial Products, Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Senior Business Analyst, 6 years. Led requirements elicitation for financial products. Modeled complex business processes. Proficient in SQL. Worked in Agile teams. Strong stakeholder management. Deep financial services domain knowledge.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Business Analyst (Financial Services). Skills: Requirements Elicitation, Process Modeling, Agile, SQL, Financial Products, Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Business Analyst, 3 years. Gathered requirements. Some Agile. No financial services specialization or senior role.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Business Analyst (Financial Services). Skills: Requirements Elicitation, Process Modeling, Agile, SQL, Financial Products, Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Financial Analyst, 7 years. Focused on financial reporting. No business analysis.",
            "relevance_score": 30
        },

        # Example: Senior Marketing Manager (Product Focus)
        {
            "jd_text": "Senior Marketing Manager (Product). Skills: Product Marketing, Go-to-Market Strategy, Product Launch, Messaging, Competitive Analysis, Demand Generation. Experience: 6+ years.",
            "resume_text": "Senior Product Marketing Manager, 7 years. Developed and executed go-to-market strategies for new products. Led successful product launches. Crafted compelling messaging. Conducted competitive analysis. Drove demand generation campaigns.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Marketing Manager (Product). Skills: Product Marketing, Go-to-Market Strategy, Product Launch, Messaging, Competitive Analysis, Demand Generation. Experience: 6+ years.",
            "resume_text": "Marketing Manager, 4 years. Managed digital campaigns. Some product exposure. No strategic product marketing.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Marketing Manager (Product). Skills: Product Marketing, Go-to-Market Strategy, Product Launch, Messaging, Competitive Analysis, Demand Generation. Experience: 6+ years.",
            "resume_text": "Product Manager, 8 years. Defined product features. No marketing focus.",
            "relevance_score": 40
        },

        # Example: Senior Financial Analyst (Investment)
        {
            "jd_text": "Senior Financial Analyst (Investment). Skills: Investment Analysis, Valuation, Financial Modeling, Equity Research, Portfolio Management, Bloomberg Terminal. Experience: 4+ years.",
            "resume_text": "Senior Investment Analyst, 5 years. Performed in-depth investment analysis and valuation. Built complex financial models. Conducted equity research. Assisted with portfolio management. Proficient with Bloomberg Terminal.",
            "relevance_score": 93
        },
        {
            "jd_text": "Senior Financial Analyst (Investment). Skills: Investment Analysis, Valuation, Financial Modeling, Equity Research, Portfolio Management, Bloomberg Terminal. Experience: 4+ years.",
            "resume_text": "Financial Analyst, 2 years. Built basic models. No investment focus.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Financial Analyst (Investment). Skills: Investment Analysis, Valuation, Financial Modeling, Equity Research, Portfolio Management, Bloomberg Terminal. Experience: 4+ years.",
            "resume_text": "Accountant, 8 years. No investment analysis.",
            "relevance_score": 25
        },

        # Example: Senior Talent Acquisition Specialist
        {
            "jd_text": "Senior Talent Acquisition Specialist. Skills: Strategic Sourcing, Full-Cycle Recruiting, Employer Branding, ATS Optimization, Interviewing, Diversity & Inclusion. Experience: 5+ years.",
            "resume_text": "Senior Talent Acquisition Specialist, 6 years. Developed and executed strategic sourcing plans. Managed full-cycle recruiting for executive roles. Enhanced employer branding. Optimized ATS workflows. Expert in D&I recruiting practices.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Talent Acquisition Specialist. Skills: Strategic Sourcing, Full-Cycle Recruiting, Employer Branding, ATS Optimization, Interviewing, Diversity & Inclusion. Experience: 5+ years.",
            "resume_text": "Recruiter, 3 years. Sourced candidates. Some full-cycle. No strategic or D&I focus.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Talent Acquisition Specialist. Skills: Strategic Sourcing, Full-Cycle Recruiting, Employer Branding, ATS Optimization, Interviewing, Diversity & Inclusion. Experience: 5+ years.",
            "resume_text": "HR Generalist, 8 years. Managed employee relations. No direct recruiting.",
            "relevance_score": 30
        },

        # Example: Technical Writer (Senior)
        {
            "jd_text": "Senior Technical Writer. Skills: Complex Technical Documentation, API Documentation, DITA/XML, Content Strategy, Information Architecture, Cross-functional Collaboration. Experience: 5+ years.",
            "resume_text": "Senior Technical Writer, 6 years. Produced complex technical documentation and API docs. Proficient in DITA/XML. Developed content strategies and information architecture. Collaborated extensively with engineering teams.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Technical Writer. Skills: Complex Technical Documentation, API Documentation, DITA/XML, Content Strategy, Information Architecture, Cross-functional Collaboration. Experience: 5+ years.",
            "resume_text": "Technical Writer, 3 years. Wrote user guides. Some API docs. No complex content strategy or DITA.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Technical Writer. Skills: Complex Technical Documentation, API Documentation, DITA/XML, Content Strategy, Information Architecture, Cross-functional Collaboration. Experience: 5+ years.",
            "resume_text": "Content Writer, 8 years. Wrote marketing content. No technical domain.",
            "relevance_score": 30
        },

        # Example: Sales Director
        {
            "jd_text": "Sales Director. Skills: Sales Strategy, Team Leadership, P&L Management, Key Account Management, CRM (Salesforce), Forecasting. Experience: 10+ years.",
            "resume_text": "Sales Director, 12 years. Developed and executed sales strategies. Led and mentored large sales teams. Managed multi-million dollar P&L. Grew key accounts. Expert in Salesforce CRM and sales forecasting.",
            "relevance_score": 97
        },
        {
            "jd_text": "Sales Director. Skills: Sales Strategy, Team Leadership, P&L Management, Key Account Management, CRM (Salesforce), Forecasting. Experience: 10+ years.",
            "resume_text": "Sales Manager, 7 years. Managed a sales team. Some forecasting. No director-level strategy or P&L.",
            "relevance_score": 70
        },
        {
            "jd_text": "Sales Director. Skills: Sales Strategy, Team Leadership, P&L Management, Key Account Management, CRM (Salesforce), Forecasting. Experience: 10+ years.",
            "resume_text": "Marketing Director, 15 years. Led marketing. No sales leadership.",
            "relevance_score": 35
        },

        # Example: Customer Service Manager
        {
            "jd_text": "Customer Service Manager. Skills: Team Leadership, Customer Satisfaction, Call Center Operations, Training & Development, Conflict Resolution, CRM. Experience: 5+ years.",
            "resume_text": "Customer Service Manager, 6 years. Led a team of 15 customer service reps. Improved customer satisfaction scores by 20%. Managed call center operations. Developed training programs. Expert in conflict resolution. Proficient in Zendesk CRM.",
            "relevance_score": 94
        },
        {
            "jd_text": "Customer Service Manager. Skills: Team Leadership, Customer Satisfaction, Call Center Operations, Training & Development, Conflict Resolution, CRM. Experience: 5+ years.",
            "resume_text": "Customer Service Rep, 3 years. Handled inquiries. No leadership or operations.",
            "relevance_score": 50
        },
        {
            "jd_text": "Customer Service Manager. Skills: Team Leadership, Customer Satisfaction, Call Center Operations, Training & Development, Conflict Resolution, CRM. Experience: 5+ years.",
            "resume_text": "Operations Coordinator, 8 years. Managed logistics. No customer service focus.",
            "relevance_score": 25
        },

        # Example: Operations Director
        {
            "jd_text": "Operations Director. Skills: Strategic Operations, Process Optimization, P&L Management, Supply Chain, Logistics, Lean/Six Sigma, Team Leadership. Experience: 10+ years.",
            "resume_text": "Operations Director, 12 years. Developed and executed strategic operations plans. Drove significant process optimization using Lean Six Sigma. Managed large P&L. Oversaw supply chain and logistics. Led large operations teams.",
            "relevance_score": 97
        },
        {
            "jd_text": "Operations Director. Skills: Strategic Operations, Process Optimization, P&L Management, Supply Chain, Logistics, Lean/Six Sigma, Team Leadership. Experience: 10+ years.",
            "resume_text": "Operations Manager, 7 years. Managed daily operations. Some process improvement. No director-level strategy or P&L.",
            "relevance_score": 70
        },
        {
            "jd_text": "Operations Director. Skills: Strategic Operations, Process Optimization, P&L Management, Supply Chain, Logistics, Lean/Six Sigma, Team Leadership. Experience: 10+ years.",
            "resume_text": "Project Manager, 15 years. Managed projects. No operations leadership.",
            "relevance_score": 35
        },

        # Example: Senior Supply Chain Analyst
        {
            "jd_text": "Senior Supply Chain Analyst. Skills: Supply Chain Analytics, Demand Planning, Inventory Optimization, Logistics Modeling, SQL, Python (Pandas), SAP/Oracle. Experience: 4+ years.",
            "resume_text": "Senior Supply Chain Analyst, 5 years. Led supply chain analytics projects. Developed demand plans and optimized inventory. Built logistics models. Proficient in SQL and Python. Used SAP for analysis.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Supply Chain Analyst. Skills: Supply Chain Analytics, Demand Planning, Inventory Optimization, Logistics Modeling, SQL, Python (Pandas), SAP/Oracle. Experience: 4+ years.",
            "resume_text": "Supply Chain Analyst, 2 years. Some data analysis. No demand planning or optimization leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Supply Chain Analyst. Skills: Supply Chain Analytics, Demand Planning, Inventory Optimization, Logistics Modeling, SQL, Python (Pandas), SAP/Oracle. Experience: 4+ years.",
            "resume_text": "Data Analyst, 6 years. Strong in SQL and Python. No supply chain domain.",
            "relevance_score": 70
        },

        # Example: Senior Electrical Design Engineer
        {
            "jd_text": "Senior Electrical Design Engineer. Skills: Analog/Digital Design, PCB Layout (Altium/Cadence), Power Electronics, FPGA/ASIC Design, EMI/EMC, Test & Validation. Experience: 6+ years.",
            "resume_text": "Senior Electrical Design Engineer, 7 years. Led complex analog and digital circuit designs. Expert in PCB layout using Altium. Designed power electronics. Experience with FPGA/ASIC. Ensured EMI/EMC compliance. Led test and validation.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Electrical Design Engineer. Skills: Analog/Digital Design, PCB Layout (Altium/Cadence), Power Electronics, FPGA/ASIC Design, EMI/EMC, Test & Validation. Experience: 6+ years.",
            "resume_text": "Electrical Design Engineer, 3 years. Designed basic circuits. Some PCB. No senior-level or FPGA/ASIC.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Electrical Design Engineer. Skills: Analog/Digital Design, PCB Layout (Altium/Cadence), Power Electronics, FPGA/ASIC Design, EMI/EMC, Test & Validation. Experience: 6+ years.",
            "resume_text": "Software Engineer, 9 years. No electrical engineering.",
            "relevance_score": 20
        },

        # Example: Senior Civil Structural Engineer
        {
            "jd_text": "Senior Civil Structural Engineer. Skills: Advanced Structural Analysis, Seismic Design, Steel/Concrete Design, Bridge/Building Structures, AutoCAD/Revit, Project Leadership. Experience: 6+ years.",
            "resume_text": "Senior Civil Structural Engineer, 7 years. Led advanced structural analysis for complex bridge and building structures. Expertise in seismic design and steel/concrete design. Proficient in AutoCAD and Revit. Provided project leadership.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Civil Structural Engineer. Skills: Advanced Structural Analysis, Seismic Design, Steel/Concrete Design, Bridge/Building Structures, AutoCAD/Revit, Project Leadership. Experience: 6+ years.",
            "resume_text": "Structural Engineer, 3 years. Performed basic structural analysis. Some AutoCAD. No senior-level design or project leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Civil Structural Engineer. Skills: Advanced Structural Analysis, Seismic Design, Steel/Concrete Design, Bridge/Building Structures, AutoCAD/Revit, Project Leadership. Experience: 6+ years.",
            "resume_text": "Construction Project Manager, 10 years. Managed construction sites. No structural design.",
            "relevance_score": 30
        },

        # Example: Senior Research Chemist
        {
            "jd_text": "Senior Research Chemist. Skills: Organic Synthesis, Reaction Kinetics, Spectroscopy (NMR, MS, IR), Chromatography (HPLC, GC), Drug Discovery, Method Development. Experience: PhD + 4 years research.",
            "resume_text": "Senior Research Chemist, PhD + 5 years research. Led organic synthesis projects with focus on reaction kinetics. Expert in NMR, MS, IR spectroscopy. Proficient in HPLC/GC. Contributed significantly to drug discovery and method development.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior Research Chemist. Skills: Organic Synthesis, Reaction Kinetics, Spectroscopy (NMR, MS, IR), Chromatography (HPLC, GC), Drug Discovery, Method Development. Experience: PhD + 4 years research.",
            "resume_text": "Research Chemist, 2 years. Performed synthesis. Some analytical techniques. No senior-level leadership or drug discovery focus.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Data Analyst. Skills: Advanced SQL, Python (Pandas, SciPy), Data Storytelling, Power BI, BigQuery, Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Senior Data Analyst with 6 years experience. Expert in SQL and Python for complex data analysis. Developed interactive dashboards in Power BI. Presented insights to executive stakeholders. Worked with BigQuery.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Data Analyst. Skills: Advanced SQL, Python (Pandas, SciPy), Data Storytelling, Power BI, BigQuery, Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Data Analyst, 3 years. Proficient in SQL and Excel. Basic Python. No experience with BigQuery or senior stakeholder management.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Data Analyst. Skills: Advanced SQL, Python (Pandas, SciPy), Data Storytelling, Power BI, BigQuery, Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Marketing Analyst, 4 years. Focused on campaign performance. Used Google Analytics. Some Excel. No SQL or Python.",
            "relevance_score": 30
        },

        # Example: Fullstack Developer
        {
            "jd_text": "Fullstack Developer. Skills: React, Node.js, Express, MongoDB, AWS Lambda, REST APIs, Git. Experience: 3+ years.",
            "resume_text": "Fullstack Developer with 4 years experience. Built responsive UIs with React. Developed backend APIs with Node.js/Express and MongoDB. Deployed serverless functions on AWS Lambda. Strong Git.",
            "relevance_score": 94
        },
        {
            "jd_text": "Fullstack Developer. Skills: React, Node.js, Express, MongoDB, AWS Lambda, REST APIs, Git. Experience: 3+ years.",
            "resume_text": "Frontend Developer, 5 years. Expert in React. No backend or cloud experience.",
            "relevance_score": 65
        },
        {
            "jd_text": "Fullstack Developer. Skills: React, Node.js, Express, MongoDB, AWS Lambda, REST APIs, Git. Experience: 3+ years.",
            "resume_text": "Backend Java Developer, 6 years. Built Spring Boot microservices. No frontend experience.",
            "relevance_score": 40
        },

        # Example: UX Researcher (Senior)
        {
            "jd_text": "Senior UX Researcher. Skills: Mixed Methods Research, Ethnography, A/B Testing, Statistical Analysis, User Journey Mapping, Workshop Facilitation. Experience: 5+ years.",
            "resume_text": "Lead UX Researcher, 7 years. Conducted mixed-methods research including ethnography and A/B testing. Strong in statistical analysis. Led user journey mapping workshops. Mentored junior researchers.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior UX Researcher. Skills: Mixed Methods Research, Ethnography, A/B Testing, Statistical Analysis, User Journey Mapping, Workshop Facilitation. Experience: 5+ years.",
            "resume_text": "Junior UX Researcher, 2 years. Conducted usability tests. Assisted with surveys. No senior-level experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "Senior UX Researcher. Skills: Mixed Methods Research, Ethnography, A/B Testing, Statistical Analysis, User Journey Mapping, Workshop Facilitation. Experience: 5+ years.",
            "resume_text": "Market Research Analyst, 8 years. Designed and analyzed large-scale surveys. No specific UX research methodologies.",
            "relevance_score": 45
        },

        # Example: Cloud Security Engineer
        {
            "jd_text": "Cloud Security Engineer. Skills: AWS Security, Azure Security, GCP Security, IAM, Network Security, Compliance (NIST, ISO 27001), Scripting (Python). Experience: 4+ years.",
            "resume_text": "Cloud Security Engineer with 5 years. Implemented security controls on AWS and Azure. Configured IAM policies. Strong in network security. Ensured compliance with NIST and ISO 27001. Wrote Python scripts for automation.",
            "relevance_score": 95
        },
        {
            "jd_text": "Cloud Security Engineer. Skills: AWS Security, Azure Security, GCP Security, IAM, Network Security, Compliance (NIST, ISO 27001), Scripting (Python). Experience: 4+ years.",
            "resume_text": "Traditional Security Analyst, 6 years. Managed on-premise firewalls and IDS. No cloud-specific security experience.",
            "relevance_score": 40
        },
        {
            "jd_text": "Cloud Security Engineer. Skills: AWS Security, Azure Security, GCP Security, IAM, Network Security, Compliance (NIST, ISO 27001), Scripting (Python). Experience: 4+ years.",
            "resume_text": "DevOps Engineer, 3 years. Deployed applications to AWS. Some understanding of security groups. No dedicated security role.",
            "relevance_score": 60
        },

        # Example: Machine Learning Engineer
        {
            "jd_text": "Machine Learning Engineer. Skills: Python, TensorFlow/PyTorch, MLOps, Distributed ML, AWS SageMaker, Data Pipelines. Experience: 4+ years.",
            "resume_text": "ML Engineer with 5 years. Developed and deployed ML models in Python using TensorFlow. Built MLOps pipelines. Experience with distributed ML and AWS SageMaker. Managed data pipelines.",
            "relevance_score": 97
        },
        {
            "jd_text": "Machine Learning Engineer. Skills: Python, TensorFlow/PyTorch, MLOps, Distributed ML, AWS SageMaker, Data Pipelines. Experience: 4+ years.",
            "resume_text": "Data Scientist, 3 years. Built models in Python. No MLOps or deployment experience.",
            "relevance_score": 70
        },
        {
            "jd_text": "Machine Learning Engineer. Skills: Python, TensorFlow/PyTorch, MLOps, Distributed ML, AWS SageMaker, Data Pipelines. Experience: 4+ years.",
            "resume_text": "Software Engineer, 6 years. Proficient in Java. No ML experience.",
            "relevance_score": 20
        },

        # Example: Content Strategist
        {
            "jd_text": "Content Strategist. Skills: Content Strategy, SEO, Audience Research, Content Calendar, Copywriting, Analytics. Experience: 4+ years.",
            "resume_text": "Content Strategist, 5 years. Developed and executed comprehensive content strategies. Strong in SEO and audience research. Managed content calendars. Produced high-quality copywriting. Used analytics to measure performance.",
            "relevance_score": 93
        },
        {
            "jd_text": "Content Strategist. Skills: Content Strategy, SEO, Audience Research, Content Calendar, Copywriting, Analytics. Experience: 4+ years.",
            "resume_text": "Social Media Manager, 3 years. Created social media content. Some understanding of audience. No strategic content planning.",
            "relevance_score": 55
        },
        {
            "jd_text": "Content Strategist. Skills: Content Strategy, SEO, Audience Research, Content Calendar, Copywriting, Analytics. Experience: 4+ years.",
            "resume_text": "Public Relations Specialist, 7 years. Managed media relations. Wrote press releases. No digital content strategy.",
            "relevance_score": 30
        },

        # Example: Senior Financial Analyst
        {
            "jd_text": "Senior Financial Analyst. Skills: Financial Planning & Analysis (FP&A), Budgeting, Forecasting, Variance Analysis, SAP/Oracle ERP, Advanced Excel. Experience: 5+ years.",
            "resume_text": "Senior Financial Analyst, 6 years. Led FP&A cycles, including budgeting and forecasting. Performed detailed variance analysis. Expert in SAP ERP and advanced Excel for financial modeling.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Financial Analyst. Skills: Financial Planning & Analysis (FP&A), Budgeting, Forecasting, Variance Analysis, SAP/Oracle ERP, Advanced Excel. Experience: 5+ years.",
            "resume_text": "Financial Analyst, 3 years. Assisted with budgeting. Proficient in basic Excel. No FP&A leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Financial Analyst. Skills: Financial Planning & Analysis (FP&A), Budgeting, Forecasting, Variance Analysis, SAP/Oracle ERP, Advanced Excel. Experience: 5+ years.",
            "resume_text": "Auditor, 7 years. Conducted financial audits. Strong in compliance. No FP&A experience.",
            "relevance_score": 40
        },

        # Example: Talent Acquisition Specialist
        {
            "jd_text": "Talent Acquisition Specialist. Skills: Full-Cycle Recruiting, Sourcing (LinkedIn Recruiter), Applicant Tracking Systems (ATS), Interviewing, Employer Branding. Experience: 3+ years.",
            "resume_text": "Talent Acquisition Specialist, 4 years. Managed full-cycle recruiting for tech roles. Expert in sourcing via LinkedIn Recruiter. Proficient in Greenhouse ATS. Conducted behavioral interviews. Contributed to employer branding.",
            "relevance_score": 92
        },
        {
            "jd_text": "Talent Acquisition Specialist. Skills: Full-Cycle Recruiting, Sourcing (LinkedIn Recruiter), Applicant Tracking Systems (ATS), Interviewing, Employer Branding. Experience: 3+ years.",
            "resume_text": "HR Coordinator, 2 years. Assisted with onboarding and scheduling interviews. No direct recruiting experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "Talent Acquisition Specialist. Skills: Full-Cycle Recruiting, Sourcing (LinkedIn Recruiter), Applicant Tracking Systems (ATS), Interviewing, Employer Branding. Experience: 3+ years.",
            "resume_text": "Sales Manager, 8 years. Led a sales team. Strong negotiation skills. No HR or recruiting experience.",
            "relevance_score": 20
        },

        # Example: Game Developer (Unity)
        {
            "jd_text": "Game Developer (Unity). Skills: Unity 3D, C#, Game Design, Shaders, Physics, Mobile Game Development. Experience: 2+ years.",
            "resume_text": "Game Developer with 3 years experience. Developed mobile games in Unity 3D using C#. Strong in game design principles, shaders, and physics. Published 2 games on app stores.",
            "relevance_score": 95
        },
        {
            "jd_text": "Game Developer (Unity). Skills: Unity 3D, C#, Game Design, Shaders, Physics, Mobile Game Development. Experience: 2+ years.",
            "resume_text": "Web Developer, 4 years. Proficient in JavaScript. Some graphics experience. No game development.",
            "relevance_score": 30
        },
        {
            "jd_text": "Game Developer (Unity). Skills: Unity 3D, C#, Game Design, Shaders, Physics, Mobile Game Development. Experience: 2+ years.",
            "resume_text": "3D Artist, 5 years. Created 3D models and textures. Familiar with game assets. No programming.",
            "relevance_score": 40
        },

        # Example: E-commerce Manager
        {
            "jd_text": "E-commerce Manager. Skills: E-commerce Platform Management (Shopify/Magento), Digital Marketing, SEO, Conversion Rate Optimization (CRO), Analytics. Experience: 4+ years.",
            "resume_text": "E-commerce Manager, 5 years. Managed Shopify store operations. Drove digital marketing campaigns and SEO. Implemented CRO strategies resulting in 15% increase in sales. Expert in Google Analytics.",
            "relevance_score": 93
        },
        {
            "jd_text": "E-commerce Manager. Skills: E-commerce Platform Management (Shopify/Magento), Digital Marketing, SEO, Conversion Rate Optimization (CRO), Analytics. Experience: 4+ years.",
            "resume_text": "Retail Store Manager, 8 years. Managed daily store operations. Some sales experience. No e-commerce platform or digital marketing.",
            "relevance_score": 35
        },
        {
            "jd_text": "E-commerce Manager. Skills: E-commerce Platform Management (Shopify/Magento), Digital Marketing, SEO, Conversion Rate Optimization (CRO), Analytics. Experience: 4+ years.",
            "resume_text": "Digital Marketing Specialist, 3 years. Focused on social media and content. Some SEO. No e-commerce platform management.",
            "relevance_score": 60
        },

        # Example: Data Governance Analyst
        {
            "jd_text": "Data Governance Analyst. Skills: Data Governance Frameworks, Data Quality, Metadata Management, Data Stewardship, SQL, Compliance (GDPR/HIPAA). Experience: 3+ years.",
            "resume_text": "Data Governance Analyst, 4 years. Developed and implemented data governance frameworks. Ensured data quality. Managed metadata. Supported data stewardship. Proficient in SQL. Ensured GDPR compliance.",
            "relevance_score": 92
        },
        {
            "jd_text": "Data Governance Analyst. Skills: Data Governance Frameworks, Data Quality, Metadata Management, Data Stewardship, SQL, Compliance (GDPR/HIPAA). Experience: 3+ years.",
            "resume_text": "Data Analyst, 2 years. Focused on data analysis. Some SQL. No specific data governance experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "Data Governance Analyst. Skills: Data Governance Frameworks, Data Quality, Metadata Management, Data Stewardship, SQL, Compliance (GDPR/HIPAA). Experience: 3+ years.",
            "resume_text": "Compliance Officer, 7 years. Ensured regulatory compliance. No data specific governance.",
            "relevance_score": 40
        },

        # Example: Robotics Engineer
        {
            "jd_text": "Robotics Engineer. Skills: ROS, C++, Python, Robot Kinematics, Control Systems, Sensor Integration. Experience: 3+ years.",
            "resume_text": "Robotics Engineer, 4 years. Developed robot control software in C++ and Python using ROS. Expertise in robot kinematics and control systems. Integrated various sensors (Lidar, Camera).",
            "relevance_score": 96
        },
        {
            "jd_text": "Robotics Engineer. Skills: ROS, C++, Python, Robot Kinematics, Control Systems, Sensor Integration. Experience: 3+ years.",
            "resume_text": "Software Engineer, 5 years. Proficient in Python. No robotics or hardware experience.",
            "relevance_score": 30
        },
        {
            "jd_text": "Robotics Engineer. Skills: ROS, C++, Python, Robot Kinematics, Control Systems, Sensor Integration. Experience: 3+ years.",
            "resume_text": "Mechanical Engineer, 6 years. Designed mechanical systems. Some programming. No robotics specific skills.",
            "relevance_score": 45
        },

        # Example: Technical Account Manager
        {
            "jd_text": "Technical Account Manager. Skills: Client Relationship Management, Technical Support, Product Knowledge (SaaS), Troubleshooting, Communication. Experience: 4+ years.",
            "resume_text": "Technical Account Manager, 5 years. Managed key client relationships for SaaS products. Provided advanced technical support and troubleshooting. Deep product knowledge. Excellent communication.",
            "relevance_score": 92
        },
        {
            "jd_text": "Technical Account Manager. Skills: Client Relationship Management, Technical Support, Product Knowledge (SaaS), Troubleshooting, Communication. Experience: 4+ years.",
            "resume_text": "Customer Service Rep, 3 years. Handled customer inquiries. No technical product knowledge.",
            "relevance_score": 30
        },
        {
            "jd_text": "Technical Account Manager. Skills: Client Relationship Management, Technical Support, Product Knowledge (SaaS), Troubleshooting, Communication. Experience: 4+ years.",
            "resume_text": "Sales Engineer, 6 years. Presented technical solutions to clients. Some sales focus. No post-sales account management.",
            "relevance_score": 70
        },

        # Example: Biomedical Engineer
        {
            "jd_text": "Biomedical Engineer. Skills: Medical Device Design, Biomechanics, Signal Processing, MATLAB, FDA Regulations, Prototyping. Experience: 3+ years.",
            "resume_text": "Biomedical Engineer, 4 years. Designed medical devices. Applied biomechanics principles. Processed biological signals in MATLAB. Familiar with FDA regulations. Built prototypes.",
            "relevance_score": 94
        },
        {
            "jd_text": "Biomedical Engineer. Skills: Medical Device Design, Biomechanics, Signal Processing, MATLAB, FDA Regulations, Prototyping. Experience: 3+ years.",
            "resume_text": "Electrical Engineer, 5 years. Designed circuits. Some signal processing. No medical device or biology focus.",
            "relevance_score": 40
        },
        {
            "jd_text": "Biomedical Engineer. Skills: Medical Device Design, Biomechanics, Signal Processing, MATLAB, FDA Regulations, Prototyping. Experience: 3+ years.",
            "resume_text": "Clinical Research Coordinator, 6 years. Managed clinical trials. No engineering background.",
            "relevance_score": 20
        },

        # Example: Investment Analyst
        {
            "jd_text": "Investment Analyst. Skills: Financial Modeling, Valuation, Equity Research, Portfolio Analysis, Bloomberg Terminal, CFA. Experience: 2+ years.",
            "resume_text": "Investment Analyst, 3 years. Built detailed financial models and valuations. Conducted equity research. Performed portfolio analysis. Proficient with Bloomberg Terminal. CFA Level II candidate.",
            "relevance_score": 93
        },
        {
            "jd_text": "Investment Analyst. Skills: Financial Modeling, Valuation, Equity Research, Portfolio Analysis, Bloomberg Terminal, CFA. Experience: 2+ years.",
            "resume_text": "Financial Planner, 5 years. Advised individual clients on financial goals. No institutional investment analysis.",
            "relevance_score": 50
        },
        {
            "jd_text": "Investment Analyst. Skills: Financial Modeling, Valuation, Equity Research, Portfolio Analysis, Bloomberg Terminal, CFA. Experience: 2+ years.",
            "resume_text": "Accountant, 7 years. Prepared financial statements. No investment analysis.",
            "relevance_score": 25
        },

        # Example: Digital Marketing Analyst
        {
            "jd_text": "Digital Marketing Analyst. Skills: Google Analytics, SEO, SEM, Data Visualization, A/B Testing, Excel, SQL. Experience: 2+ years.",
            "resume_text": "Digital Marketing Analyst, 3 years. Analyzed website performance using Google Analytics. Optimized SEO/SEM campaigns. Created data visualizations. Conducted A/B tests. Proficient in Excel and basic SQL.",
            "relevance_score": 90
        },
        {
            "jd_text": "Digital Marketing Analyst. Skills: Google Analytics, SEO, SEM, Data Visualization, A/B Testing, Excel, SQL. Experience: 2+ years.",
            "resume_text": "Marketing Coordinator, 1 year. Managed social media. Some analytics exposure. No in-depth analysis.",
            "relevance_score": 55
        },
        {
            "jd_text": "Digital Marketing Analyst. Skills: Google Analytics, SEO, SEM, Data Visualization, A/B Testing, Excel, SQL. Experience: 2+ years.",
            "resume_text": "Data Analyst, 4 years. Strong in SQL and Python. No specific digital marketing domain knowledge.",
            "relevance_score": 65
        },

        # Example: Supply Chain Analyst
        {
            "jd_text": "Supply Chain Analyst. Skills: Supply Chain Analytics, Inventory Optimization, Logistics, Data Modeling, Excel, SQL, ERP Systems. Experience: 3+ years.",
            "resume_text": "Supply Chain Analyst, 4 years. Performed in-depth supply chain analytics. Optimized inventory levels. Analyzed logistics data. Built data models in Excel. Used SQL for data extraction from ERP systems.",
            "relevance_score": 93
        },
        {
            "jd_text": "Supply Chain Analyst. Skills: Supply Chain Analytics, Inventory Optimization, Logistics, Data Modeling, Excel, SQL, ERP Systems. Experience: 3+ years.",
            "resume_text": "Operations Coordinator, 2 years. Managed daily operations. Some inventory tracking. No analytical role.",
            "relevance_score": 40
        },
        {
            "jd_text": "Supply Chain Analyst. Skills: Supply Chain Analytics, Inventory Optimization, Logistics, Data Modeling, Excel, SQL, ERP Systems. Experience: 3+ years.",
            "resume_text": "Business Analyst, 5 years. Focused on process improvement. Some data analysis. No supply chain domain expertise.",
            "relevance_score": 50
        },

        # Example: Frontend Engineer (Senior)
        {
            "jd_text": "Senior Frontend Engineer. Skills: React, TypeScript, Redux, Webpack, Performance Optimization, Unit Testing, UI/UX Principles. Experience: 5+ years.",
            "resume_text": "Senior Frontend Engineer, 6 years. Developed complex SPAs with React and TypeScript. Managed state with Redux. Optimized build processes with Webpack. Focused on performance and unit testing. Strong UI/UX principles.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Frontend Engineer. Skills: React, TypeScript, Redux, Webpack, Performance Optimization, Unit Testing, UI/UX Principles. Experience: 5+ years.",
            "resume_text": "Mid-level Frontend Developer, 3 years. Proficient in React. No senior-level experience or performance optimization focus.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Frontend Engineer. Skills: React, TypeScript, Redux, Webpack, Performance Optimization, Unit Testing, UI/UX Principles. Experience: 5+ years.",
            "resume_text": "Backend Developer, 7 years. Strong in Python. No frontend experience.",
            "relevance_score": 20
        },

        # Example: Database Developer
        {
            "jd_text": "Database Developer. Skills: SQL, Stored Procedures, Database Design, Performance Tuning, ETL, Data Warehousing. Experience: 3+ years.",
            "resume_text": "Database Developer, 4 years. Wrote complex SQL queries and stored procedures. Designed relational databases. Performed performance tuning. Built ETL processes for data warehousing.",
            "relevance_score": 94
        },
        {
            "jd_text": "Database Developer. Skills: SQL, Stored Procedures, Database Design, Performance Tuning, ETL, Data Warehousing. Experience: 3+ years.",
            "resume_text": "Data Analyst, 2 years. Strong in SQL for querying. No database design or tuning experience.",
            "relevance_score": 60
        },
        {
            "jd_text": "Database Developer. Skills: SQL, Stored Procedures, Database Design, Performance Tuning, ETL, Data Warehousing. Experience: 3+ years.",
            "resume_text": "Application Developer, 5 years. Used ORMs to interact with databases. Limited direct SQL knowledge.",
            "relevance_score": 35
        },

        # Example: Education Program Manager
        {
            "jd_text": "Education Program Manager. Skills: Program Development, Curriculum Design, Stakeholder Engagement, Budget Management, K-12/Higher Ed. Experience: 5+ years.",
            "resume_text": "Education Program Manager, 6 years. Developed and managed educational programs. Designed curriculum for K-12. Engaged with diverse stakeholders. Managed program budgets.",
            "relevance_score": 92
        },
        {
            "jd_text": "Education Program Manager. Skills: Program Development, Curriculum Design, Stakeholder Engagement, Budget Management, K-12/Higher Ed. Experience: 5+ years.",
            "resume_text": "Teacher, 8 years. Taught in a classroom. No program management or curriculum design at a broader level.",
            "relevance_score": 50
        },
        {
            "jd_text": "Education Program Manager. Skills: Program Development, Curriculum Design, Stakeholder Engagement, Budget Management, K-12/Higher Ed. Experience: 5+ years.",
            "resume_text": "Project Coordinator, 3 years. Assisted with project planning. No education specific experience.",
            "relevance_score": 25
        },

        # Example: Environmental Engineer
        {
            "jd_text": "Environmental Engineer. Skills: Environmental Regulations (EPA), Site Assessment, Remediation, Water/Air Quality, CAD. Experience: 3+ years.",
            "resume_text": "Environmental Engineer, 4 years. Conducted site assessments and designed remediation plans. Expert in EPA regulations. Monitored water and air quality. Used CAD for designs.",
            "relevance_score": 94
        },
        {
            "jd_text": "Environmental Engineer. Skills: Environmental Regulations (EPA), Site Assessment, Remediation, Water/Air Quality, CAD. Experience: 3+ years.",
            "resume_text": "Civil Engineer, 5 years. Focused on infrastructure. Some CAD. No environmental specific expertise.",
            "relevance_score": 40
        },
        {
            "jd_text": "Environmental Engineer. Skills: Environmental Regulations (EPA), Site Assessment, Remediation, Water/Air Quality, CAD. Experience: 3+ years.",
            "resume_text": "Lab Technician, 2 years. Performed lab tests. No engineering or regulatory experience.",
            "relevance_score": 15
        },

        # Example: Business Intelligence Developer
        {
            "jd_text": "Business Intelligence Developer. Skills: Power BI, Tableau, SQL, Data Warehousing, ETL, SSIS/SSRS. Experience: 4+ years.",
            "resume_text": "BI Developer, 5 years. Built interactive dashboards in Power BI and Tableau. Designed and managed data warehouses. Developed ETL processes using SSIS. Strong SQL skills.",
            "relevance_score": 95
        },
        {
            "jd_text": "Business Intelligence Developer. Skills: Power BI, Tableau, SQL, Data Warehousing, ETL, SSIS/SSRS. Experience: 4+ years.",
            "resume_text": "Data Analyst, 3 years. Used Tableau for visualization. Some SQL. No data warehousing or ETL development.",
            "relevance_score": 60
        },
        {
            "jd_text": "Business Intelligence Developer. Skills: Power BI, Tableau, SQL, Data Warehousing, ETL, SSIS/SSRS. Experience: 4+ years.",
            "resume_text": "Software Developer, 6 years. Built web applications. No BI or data warehousing experience.",
            "relevance_score": 25
        },

        # Example: Pharmacist
        {
            "jd_text": "Pharmacist. Skills: Medication Dispensing, Patient Counseling, Drug Interactions, Pharmacy Operations, EHR. Experience: 2+ years.",
            "resume_text": "Pharmacist, 3 years. Dispensed medications accurately. Provided patient counseling on drug usage and interactions. Managed pharmacy operations. Proficient in EHR systems.",
            "relevance_score": 93
        },
        {
            "jd_text": "Pharmacist. Skills: Medication Dispensing, Patient Counseling, Drug Interactions, Pharmacy Operations, EHR. Experience: 2+ years.",
            "resume_text": "Pharmacy Technician, 5 years. Assisted pharmacists with dispensing. No counseling or full pharmacist duties.",
            "relevance_score": 45
        },
        {
            "jd_text": "Pharmacist. Skills: Medication Dispensing, Patient Counseling, Drug Interactions, Pharmacy Operations, EHR. Experience: 2+ years.",
            "resume_text": "Nurse, 7 years. Administered medications. No pharmacy specific knowledge.",
            "relevance_score": 30
        },

        # Example: Social Media Manager
        {
            "jd_text": "Social Media Manager. Skills: Social Media Strategy, Content Creation, Community Management, Analytics, Paid Social Campaigns. Experience: 3+ years.",
            "resume_text": "Social Media Manager, 4 years. Developed and executed social media strategies. Created engaging content. Managed online communities. Analyzed performance using analytics tools. Ran successful paid social campaigns.",
            "relevance_score": 94
        },
        {
            "jd_text": "Social Media Manager. Skills: Social Media Strategy, Content Creation, Community Management, Analytics, Paid Social Campaigns. Experience: 3+ years.",
            "resume_text": "Marketing Assistant, 2 years. Assisted with social media posts. No strategic or paid campaign experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "Social Media Manager. Skills: Social Media Strategy, Content Creation, Community Management, Analytics, Paid Social Campaigns. Experience: 3+ years.",
            "resume_text": "Public Relations Specialist, 6 years. Managed media relations. No social media focus.",
            "relevance_score": 25
        },

        # Example: Research Engineer
        {
            "jd_text": "Research Engineer. Skills: R&D, Prototyping, Data Analysis (Python/MATLAB), Experimental Design, Technical Writing. Experience: 4+ years.",
            "resume_text": "Research Engineer, 5 years. Led R&D projects. Developed prototypes. Analyzed experimental data using Python and MATLAB. Designed complex experiments. Wrote technical reports and papers.",
            "relevance_score": 96
        },
        {
            "jd_text": "Research Engineer. Skills: R&D, Prototyping, Data Analysis (Python/MATLAB), Experimental Design, Technical Writing. Experience: 4+ years.",
            "resume_text": "Software Engineer, 3 years. Proficient in Python. No research or experimental design.",
            "relevance_score": 40
        },
        {
            "jd_text": "Research Engineer. Skills: R&D, Prototyping, Data Analysis (Python/MATLAB), Experimental Design, Technical Writing. Experience: 4+ years.",
            "resume_text": "Lab Technician, 7 years. Performed routine lab procedures. No research or engineering role.",
            "relevance_score": 20
        },

        # Example: Solutions Architect
        {
            "jd_text": "Solutions Architect. Skills: Enterprise Architecture, Cloud Solutions (AWS/Azure), Microservices, API Design, Stakeholder Management, Technical Leadership. Experience: 7+ years.",
            "resume_text": "Solutions Architect, 8 years. Designed enterprise-level cloud solutions on AWS and Azure. Architected microservices and robust APIs. Managed complex stakeholder relationships. Provided technical leadership to development teams.",
            "relevance_score": 97
        },
        {
            "jd_text": "Solutions Architect. Skills: Enterprise Architecture, Cloud Solutions (AWS/Azure), Microservices, API Design, Stakeholder Management, Technical Leadership. Experience: 7+ years.",
            "resume_text": "Senior Software Engineer, 5 years. Built microservices. Some AWS experience. No architecture design or leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Solutions Architect. Skills: Enterprise Architecture, Cloud Solutions (AWS/Azure), Microservices, API Design, Stakeholder Management, Technical Leadership. Experience: 7+ years.",
            "resume_text": "Project Manager, 10 years. Managed IT projects. No technical architecture expertise.",
            "relevance_score": 30
        },

        # Example: Technical Trainer
        {
            "jd_text": "Technical Trainer. Skills: Training Delivery, Curriculum Development, Technical Concepts (Software/IT), Adult Learning Principles, Presentation Skills. Experience: 3+ years.",
            "resume_text": "Technical Trainer, 4 years. Delivered engaging technical training sessions on software products. Developed comprehensive curriculum based on adult learning principles. Excellent presentation skills.",
            "relevance_score": 92
        },
        {
            "jd_text": "Technical Trainer. Skills: Training Delivery, Curriculum Development, Technical Concepts (Software/IT), Adult Learning Principles, Presentation Skills. Experience: 3+ years.",
            "resume_text": "Software Developer, 5 years. Strong technical skills. No training or teaching experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "Technical Trainer. Skills: Training Delivery, Curriculum Development, Technical Concepts (Software/IT), Adult Learning Principles, Presentation Skills. Experience: 3+ years.",
            "resume_text": "Customer Service Manager, 7 years. Trained customer service reps. No technical domain knowledge.",
            "relevance_score": 35
        },

        # Example: Data Privacy Officer
        {
            "jd_text": "Data Privacy Officer. Skills: GDPR, CCPA, Data Protection, Privacy Impact Assessments, Compliance, Legal Frameworks. Experience: 5+ years.",
            "resume_text": "Data Privacy Officer, 6 years. Ensured compliance with GDPR and CCPA. Conducted privacy impact assessments. Developed data protection policies. Strong knowledge of legal frameworks.",
            "relevance_score": 95
        },
        {
            "jd_text": "Data Privacy Officer. Skills: GDPR, CCPA, Data Protection, Privacy Impact Assessments, Compliance, Legal Frameworks. Experience: 5+ years.",
            "resume_text": "Compliance Analyst, 3 years. Reviewed company policies. Some regulatory knowledge. No specific data privacy focus.",
            "relevance_score": 60
        },
        {
            "jd_text": "Data Privacy Officer. Skills: GDPR, CCPA, Data Protection, Privacy Impact Assessments, Compliance, Legal Frameworks. Experience: 5+ years.",
            "resume_text": "IT Security Analyst, 4 years. Focused on network security. No privacy law expertise.",
            "relevance_score": 30
        },

        # Example: Quantitative Analyst (Quant)
        {
            "jd_text": "Quantitative Analyst. Skills: Python/R, C++, Statistical Modeling, Time Series Analysis, Financial Markets, Risk Management. Experience: 3+ years.",
            "resume_text": "Quantitative Analyst, 4 years. Developed statistical models in Python and C++. Performed time series analysis for financial markets. Built risk management models.",
            "relevance_score": 96
        },
        {
            "jd_text": "Quantitative Analyst. Skills: Python/R, C++, Statistical Modeling, Time Series Analysis, Financial Markets, Risk Management. Experience: 3+ years.",
            "resume_text": "Data Scientist, 3 years. Strong in Python for ML. No specific financial markets or C++ experience.",
            "relevance_score": 70
        },
        {
            "jd_text": "Quantitative Analyst. Skills: Python/R, C++, Statistical Modeling, Time Series Analysis, Financial Markets, Risk Management. Experience: 3+ years.",
            "resume_text": "Financial Analyst, 5 years. Focused on corporate finance. No quantitative modeling.",
            "relevance_score": 25
        },

        # Example: Product Marketing Manager
        {
            "jd_text": "Product Marketing Manager. Skills: Go-to-Market Strategy, Product Launch, Market Positioning, Messaging, Sales Enablement, Competitive Analysis. Experience: 4+ years.",
            "resume_text": "Product Marketing Manager, 5 years. Developed and executed go-to-market strategies. Led successful product launches. Defined market positioning and messaging. Created sales enablement materials. Conducted competitive analysis.",
            "relevance_score": 94
        },
        {
            "jd_text": "Product Marketing Manager. Skills: Go-to-Market Strategy, Product Launch, Market Positioning, Messaging, Sales Enablement, Competitive Analysis. Experience: 4+ years.",
            "resume_text": "Marketing Specialist, 3 years. Created content and managed social media. No product launch or strategic positioning.",
            "relevance_score": 55
        },
        {
            "jd_text": "Product Marketing Manager. Skills: Go-to-Market Strategy, Product Launch, Market Positioning, Messaging, Sales Enablement, Competitive Analysis. Experience: 4+ years.",
            "resume_text": "Product Manager, 6 years. Defined product features. No marketing focus.",
            "relevance_score": 40
        },

        # Example: Site Reliability Engineer (SRE)
        {
            "jd_text": "Site Reliability Engineer (SRE). Skills: Reliability Engineering, Distributed Systems, Cloud (AWS/GCP), Kubernetes, Prometheus/Grafana, Incident Management. Experience: 5+ years.",
            "resume_text": "SRE with 6 years experience. Focused on reliability engineering for distributed systems on AWS. Expert in Kubernetes. Implemented monitoring with Prometheus/Grafana. Led incident management.",
            "relevance_score": 97
        },
        {
            "jd_text": "Site Reliability Engineer (SRE). Skills: Reliability Engineering, Distributed Systems, Cloud (AWS/GCP), Kubernetes, Prometheus/Grafana, Incident Management. Experience: 5+ years.",
            "resume_text": "DevOps Engineer, 4 years. Built CI/CD pipelines. Used Docker and some AWS. No deep reliability focus.",
            "relevance_score": 70
        },
        {
            "jd_text": "Site Reliability Engineer (SRE). Skills: Reliability Engineering, Distributed Systems, Cloud (AWS/GCP), Kubernetes, Prometheus/Grafana, Incident Management. Experience: 5+ years.",
            "resume_text": "System Administrator, 8 years. Managed Linux servers. No cloud or distributed systems experience.",
            "relevance_score": 30
        },

        # Example: Data Architect
        {
            "jd_text": "Data Architect. Skills: Data Modeling, Data Warehousing, Cloud Data Platforms (Snowflake/Databricks), ETL Architecture, Big Data Technologies, Data Governance. Experience: 7+ years.",
            "resume_text": "Data Architect, 8 years. Designed complex data models and data warehouses. Architected solutions on Snowflake and Databricks. Led ETL architecture. Expertise in big data technologies and data governance.",
            "relevance_score": 98
        },
        {
            "jd_text": "Data Architect. Skills: Data Modeling, Data Warehousing, Cloud Data Platforms (Snowflake/Databricks), ETL Architecture, Big Data Technologies, Data Governance. Experience: 7+ years.",
            "resume_text": "Data Engineer, 5 years. Built ETL pipelines. Some data modeling. No architecture leadership.",
            "relevance_score": 75
        },
        {
            "jd_text": "Data Architect. Skills: Data Modeling, Data Warehousing, Cloud Data Platforms (Snowflake/Databricks), ETL Architecture, Big Data Technologies, Data Governance. Experience: 7+ years.",
            "resume_text": "Business Intelligence Developer, 6 years. Built dashboards. Strong SQL. No data architecture.",
            "relevance_score": 40
        },

        # Example: Technical Sales Engineer
        {
            "jd_text": "Technical Sales Engineer. Skills: Technical Sales, Product Demonstrations, Solution Selling, Client Presentations, CRM (Salesforce), Networking. Experience: 3+ years.",
            "resume_text": "Technical Sales Engineer, 4 years. Drove technical sales cycles. Conducted compelling product demonstrations. Applied solution selling methodologies. Delivered client presentations. Proficient in Salesforce.",
            "relevance_score": 93
        },
        {
            "jd_text": "Technical Sales Engineer. Skills: Technical Sales, Product Demonstrations, Solution Selling, Client Presentations, CRM (Salesforce), Networking. Experience: 3+ years.",
            "resume_text": "Sales Representative, 5 years. Strong in sales. No technical product knowledge or demonstrations.",
            "relevance_score": 50
        },
        {
            "jd_text": "Technical Sales Engineer. Skills: Technical Sales, Product Demonstrations, Solution Selling, Client Presentations, CRM (Salesforce), Networking. Experience: 3+ years.",
            "resume_text": "Software Engineer, 7 years. Strong technical skills. No sales experience.",
            "relevance_score": 20
        },

        # Example: Research Scientist (AI/ML)
        {
            "jd_text": "Research Scientist (AI/ML). Skills: Deep Learning, Reinforcement Learning, Computer Vision, NLP, Python, PyTorch/TensorFlow, Publications. Experience: PhD + 3 years research.",
            "resume_text": "AI Research Scientist, PhD + 4 years research. Expertise in Deep Learning and Reinforcement Learning. Published extensively in Computer Vision and NLP. Implemented models in PyTorch. Strong Python.",
            "relevance_score": 98
        },
        {
            "jd_text": "Research Scientist (AI/ML). Skills: Deep Learning, Reinforcement Learning, Computer Vision, NLP, Python, PyTorch/TensorFlow, Publications. Experience: PhD + 3 years research.",
            "resume_text": "Machine Learning Engineer, 5 years. Deployed ML models. Some deep learning. No research focus or publications.",
            "relevance_score": 70
        },
        {
            "jd_text": "Research Scientist (AI/ML). Skills: Deep Learning, Reinforcement Learning, Computer Vision, NLP, Python, PyTorch/TensorFlow, Publications. Experience: PhD + 3 years research.",
            "resume_text": "Data Scientist, 6 years. Focused on predictive modeling. No deep learning research.",
            "relevance_score": 40
        },

        # Example: Chief Financial Officer (CFO)
        {
            "jd_text": "Chief Financial Officer (CFO). Skills: Financial Strategy, Corporate Finance, M&A, Investor Relations, GAAP, Team Leadership, Board Reporting. Experience: 15+ years.",
            "resume_text": "CFO with 20 years experience. Developed and executed financial strategies. Led M&A activities. Managed investor relations. Ensured GAAP compliance. Led large finance teams and reported to board.",
            "relevance_score": 98
        },
        {
            "jd_text": "Chief Financial Officer (CFO). Skills: Financial Strategy, Corporate Finance, M&A, Investor Relations, GAAP, Team Leadership, Board Reporting. Experience: 15+ years.",
            "resume_text": "Financial Controller, 10 years. Managed financial reporting. Some budgeting. No strategic finance or M&A leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Chief Financial Officer (CFO). Skills: Financial Strategy, Corporate Finance, M&A, Investor Relations, GAAP, Team Leadership, Board Reporting. Experience: 15+ years.",
            "resume_text": "Senior Financial Analyst, 7 years. Built financial models. No executive leadership.",
            "relevance_score": 30
        },

        # Example: Director of Engineering
        {
            "jd_text": "Director of Engineering. Skills: Engineering Leadership, Software Architecture, Scalability, Team Management, Budgeting, Agile Methodologies. Experience: 10+ years.",
            "resume_text": "Director of Engineering, 12 years. Led multiple engineering teams. Defined software architecture for scalable systems. Managed departmental budgets. Championed Agile methodologies.",
            "relevance_score": 97
        },
        {
            "jd_text": "Director of Engineering. Skills: Engineering Leadership, Software Architecture, Scalability, Team Management, Budgeting, Agile Methodologies. Experience: 10+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Strong in architecture. Some team lead experience. No director-level management.",
            "relevance_score": 70
        },
        {
            "jd_text": "Director of Engineering. Skills: Engineering Leadership, Software Architecture, Scalability, Team Management, Budgeting, Agile Methodologies. Experience: 10+ years.",
            "resume_text": "Project Manager, 15 years. Managed IT projects. No direct engineering leadership.",
            "relevance_score": 40
        },

        # Example: Head of Marketing
        {
            "jd_text": "Head of Marketing. Skills: Marketing Strategy, Brand Management, Digital Marketing, Team Leadership, P&L Management, Market Analysis. Experience: 10+ years.",
            "resume_text": "Head of Marketing, 12 years. Developed and executed global marketing strategies. Built strong brands. Led digital marketing initiatives. Managed large teams and P&L. Strong market analysis.",
            "relevance_score": 96
        },
        {
            "jd_text": "Head of Marketing. Skills: Marketing Strategy, Brand Management, Digital Marketing, Team Leadership, P&L Management, Market Analysis. Experience: 10+ years.",
            "resume_text": "Marketing Manager, 7 years. Managed campaigns. Some team leadership. No head-of-department experience or P&L.",
            "relevance_score": 65
        },
        {
            "jd_text": "Head of Marketing. Skills: Marketing Strategy, Brand Management, Digital Marketing, Team Leadership, P&L Management, Market Analysis. Experience: 10+ years.",
            "resume_text": "Sales Director, 15 years. Led sales teams. No marketing strategy or brand management.",
            "relevance_score": 35
        },

        # Example: Principal Software Engineer
        {
            "jd_text": "Principal Software Engineer. Skills: System Design, Distributed Systems, Scalability, Mentorship, Code Quality, Performance Optimization. Experience: 8+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Led system design for highly scalable distributed systems. Mentored multiple engineers. Championed code quality and performance optimization initiatives.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Software Engineer. Skills: System Design, Distributed Systems, Scalability, Mentorship, Code Quality, Performance Optimization. Experience: 8+ years.",
            "resume_text": "Senior Software Engineer, 6 years. Built complex features. Some design experience. No principal-level leadership or system-wide impact.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Software Engineer. Skills: System Design, Distributed Systems, Scalability, Mentorship, Code Quality, Performance Optimization. Experience: 8+ years.",
            "resume_text": "DevOps Engineer, 7 years. Focused on CI/CD. Some scripting. No core software engineering design.",
            "relevance_score": 45
        },

        # Example: Senior HR Business Partner
        {
            "jd_text": "Senior HR Business Partner. Skills: Strategic HR, Organizational Development, Change Management, Leadership Coaching, Workforce Planning, HR Analytics. Experience: 8+ years.",
            "resume_text": "Senior HRBP, 9 years. Partnered with executive leadership on strategic HR initiatives. Drove organizational development and change management. Provided leadership coaching. Led workforce planning and utilized HR analytics.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior HR Business Partner. Skills: Strategic HR, Organizational Development, Change Management, Leadership Coaching, Workforce Planning, HR Analytics. Experience: 8+ years.",
            "resume_text": "HR Generalist, 5 years. Managed employee relations. Some HRIS experience. No strategic or OD focus.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior HR Business Partner. Skills: Strategic HR, Organizational Development, Change Management, Leadership Coaching, Workforce Planning, HR Analytics. Experience: 8+ years.",
            "resume_text": "Recruitment Manager, 10 years. Led recruiting teams. No broad HRBP experience.",
            "relevance_score": 30
        },

        # Example: Senior UX Designer
        {
            "jd_text": "Senior UX Designer. Skills: User-Centered Design, Wireframing, Prototyping (Figma/Sketch), Usability Testing, Design Systems, Interaction Design, User Flows. Experience: 5+ years.",
            "resume_text": "Senior UX Designer, 6 years. Led end-to-end user-centered design processes. Created complex wireframes and prototypes in Figma. Conducted extensive usability testing. Contributed to design systems and defined user flows. Expertise in interaction design.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior UX Designer. Skills: User-Centered Design, Wireframing, Prototyping (Figma/Sketch), Usability Testing, Design Systems, Interaction Design, User Flows. Experience: 5+ years.",
            "resume_text": "Mid-level UI Designer, 3 years. Focused on visual design. Some prototyping. No senior UX focus.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior UX Designer. Skills: User-Centered Design, Wireframing, Prototyping (Figma/Sketch), Usability Testing, Design Systems, Interaction Design, User Flows. Experience: 5+ years.",
            "resume_text": "Graphic Designer, 8 years. Created marketing collateral. No product design or UX.",
            "relevance_score": 25
        },

        # Example: Data Scientist (Lead)
        {
            "jd_text": "Lead Data Scientist. Skills: Machine Learning, Deep Learning, Statistical Modeling, Python (Scikit-learn, PyTorch), Spark, Mentorship, Project Leadership. Experience: 7+ years.",
            "resume_text": "Lead Data Scientist, 8 years. Developed and deployed advanced ML/DL models in Python and PyTorch. Led statistical modeling efforts. Utilized Spark for big data. Mentored junior data scientists and led multiple data science projects.",
            "relevance_score": 98
        },
        {
            "jd_text": "Lead Data Scientist. Skills: Machine Learning, Deep Learning, Statistical Modeling, Python (Scikit-learn, PyTorch), Spark, Mentorship, Project Leadership. Experience: 7+ years.",
            "resume_text": "Data Scientist, 4 years. Built ML models. Some Python. No leadership or big data scale.",
            "relevance_score": 70
        },
        {
            "jd_text": "Lead Data Scientist. Skills: Machine Learning, Deep Learning, Statistical Modeling, Python (Scikit-learn, PyTorch), Spark, Mentorship, Project Leadership. Experience: 7+ years.",
            "resume_text": "Business Intelligence Manager, 10 years. Led BI teams. No hands-on ML/DL.",
            "relevance_score": 40
        },

        # Example: Senior DevOps Engineer
        {
            "jd_text": "Senior DevOps Engineer. Skills: AWS/Azure/GCP, Kubernetes, Helm, Terraform, CI/CD Automation, Observability (Prometheus, Grafana), Scripting (Go/Python). Experience: 6+ years.",
            "resume_text": "Senior DevOps Engineer, 7 years. Designed and implemented scalable infrastructure on AWS using Terraform. Managed Kubernetes clusters with Helm. Built advanced CI/CD pipelines. Implemented observability with Prometheus/Grafana. Proficient in Go and Python scripting.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior DevOps Engineer. Skills: AWS/Azure/GCP, Kubernetes, Helm, Terraform, CI/CD Automation, Observability (Prometheus, Grafana), Scripting (Go/Python). Experience: 6+ years.",
            "resume_text": "DevOps Engineer, 3 years. Used Docker and Jenkins. Some AWS. No senior-level experience or advanced IaC.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior DevOps Engineer. Skills: AWS/Azure/GCP, Kubernetes, Helm, Terraform, CI/CD Automation, Observability (Prometheus, Grafana), Scripting (Go/Python). Experience: 6+ years.",
            "resume_text": "System Administrator, 10 years. Managed on-premise Linux servers. No cloud automation.",
            "relevance_score": 30
        },

        # Example: Product Owner
        {
            "jd_text": "Product Owner. Skills: Agile, Scrum, Backlog Management, User Stories, Stakeholder Communication, Product Vision. Experience: 3+ years.",
            "resume_text": "Product Owner, 4 years. Managed product backlogs and wrote detailed user stories in Scrum teams. Communicated product vision to stakeholders. Ensured alignment with business goals.",
            "relevance_score": 90
        },
        {
            "jd_text": "Product Owner. Skills: Agile, Scrum, Backlog Management, User Stories, Stakeholder Communication, Product Vision. Experience: 3+ years.",
            "resume_text": "Business Analyst, 5 years. Gathered requirements. Some Agile exposure. No product ownership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Product Owner. Skills: Agile, Scrum, Backlog Management, User Stories, Stakeholder Communication, Product Vision. Experience: 3+ years.",
            "resume_text": "Project Manager, 7 years. Managed project timelines. No product-specific role.",
            "relevance_score": 35
        },

        # Example: Research & Development Manager
        {
            "jd_text": "R&D Manager. Skills: Research Leadership, Innovation, Project Management, Budgeting, Team Building, Scientific Method. Experience: 8+ years.",
            "resume_text": "R&D Manager, 10 years. Led multiple research teams. Drove innovation from concept to product. Managed complex R&D projects and budgets. Built high-performing scientific teams. Strong in scientific method.",
            "relevance_score": 95
        },
        {
            "jd_text": "R&D Manager. Skills: Research Leadership, Innovation, Project Management, Budgeting, Team Building, Scientific Method. Experience: 8+ years.",
            "resume_text": "Research Scientist, 5 years. Conducted experiments. No managerial experience.",
            "relevance_score": 60
        },
        {
            "jd_text": "R&D Manager. Skills: Research Leadership, Innovation, Project Management, Budgeting, Team Building, Scientific Method. Experience: 8+ years.",
            "resume_text": "Project Manager, 12 years. Managed construction projects. No R&D domain.",
            "relevance_score": 20
        },

        # Example: Cybersecurity Consultant
        {
            "jd_text": "Cybersecurity Consultant. Skills: Security Assessments, Penetration Testing, Compliance Audits, Risk Management, Client Advisory. Experience: 5+ years.",
            "resume_text": "Cybersecurity Consultant, 6 years. Performed comprehensive security assessments and penetration tests. Conducted compliance audits (ISO 27001, NIST). Advised clients on risk management strategies.",
            "relevance_score": 94
        },
        {
            "jd_text": "Cybersecurity Consultant. Skills: Security Assessments, Penetration Testing, Compliance Audits, Risk Management, Client Advisory. Experience: 5+ years.",
            "resume_text": "Cybersecurity Analyst, 3 years. Monitored SIEM. Some vulnerability scanning. No consulting or penetration testing.",
            "relevance_score": 65
        },
        {
            "jd_text": "Cybersecurity Consultant. Skills: Security Assessments, Penetration Testing, Compliance Audits, Risk Management, Client Advisory. Experience: 5+ years.",
            "resume_text": "IT Auditor, 8 years. Focused on financial audits. Some IT controls. No security assessments.",
            "relevance_score": 30
        },

        # Example: Technical Product Manager
        {
            "jd_text": "Technical Product Manager. Skills: Product Management, Technical Specifications, Software Development Lifecycle, API Products, Agile, Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Technical Product Manager, 6 years. Managed technical products from concept to launch. Wrote detailed technical specifications. Deep understanding of SDLC and API products. Worked in Agile. Strong stakeholder management.",
            "relevance_score": 95
        },
        {
            "jd_text": "Technical Product Manager. Skills: Product Management, Technical Specifications, Software Development Lifecycle, API Products, Agile, Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Software Engineer, 7 years. Built APIs. Some product interest. No formal product management.",
            "relevance_score": 60
        },
        {
            "jd_text": "Technical Product Manager. Skills: Product Management, Technical Specifications, Software Development Lifecycle, API Products, Agile, Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Marketing Manager, 8 years. Managed marketing campaigns. No technical product expertise.",
            "relevance_score": 25
        },

        # Example: Supply Chain Director
        {
            "jd_text": "Supply Chain Director. Skills: Global Supply Chain Management, S&OP, Logistics, Procurement, Strategic Planning, Team Leadership, ERP Systems. Experience: 10+ years.",
            "resume_text": "Supply Chain Director, 12 years. Oversaw global supply chain operations. Led S&OP processes. Optimized logistics and procurement. Developed strategic plans. Managed large teams. Expert in SAP ERP.",
            "relevance_score": 97
        },
        {
            "jd_text": "Supply Chain Director. Skills: Global Supply Chain Management, S&OP, Logistics, Procurement, Strategic Planning, Team Leadership, ERP Systems. Experience: 10+ years.",
            "resume_text": "Supply Chain Manager, 7 years. Managed specific supply chain functions. No director-level strategic planning.",
            "relevance_score": 70
        },
        {
            "jd_text": "Supply Chain Director. Skills: Global Supply Chain Management, S&OP, Logistics, Procurement, Strategic Planning, Team Leadership, ERP Systems. Experience: 10+ years.",
            "resume_text": "Operations Manager, 15 years. Managed manufacturing operations. No end-to-end supply chain leadership.",
            "relevance_score": 40
        },

        # Example: Senior Electrical Engineer
        {
            "jd_text": "Senior Electrical Engineer. Skills: Power Electronics, Analog/Digital Circuit Design, PCB Layout (Altium), Embedded Systems, EMI/EMC. Experience: 5+ years.",
            "resume_text": "Senior Electrical Engineer, 6 years. Designed complex power electronics and mixed-signal circuits. Proficient in PCB layout using Altium. Developed embedded systems. Expertise in EMI/EMC compliance.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Electrical Engineer. Skills: Power Electronics, Analog/Digital Circuit Design, PCB Layout (Altium), Embedded Systems, EMI/EMC. Experience: 5+ years.",
            "resume_text": "Electrical Engineer, 3 years. Designed basic circuits. Some PCB experience. No senior-level expertise.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Electrical Engineer. Skills: Power Electronics, Analog/Digital Circuit Design, PCB Layout (Altium), Embedded Systems, EMI/EMC. Experience: 5+ years.",
            "resume_text": "Software Engineer, 8 years. Worked on backend systems. No hardware or electrical engineering.",
            "relevance_score": 20
        },

        # Example: Construction Project Manager
        {
            "jd_text": "Construction Project Manager. Skills: Construction Management, Project Scheduling (Primavera/MS Project), Budget Control, Site Supervision, Risk Management. Experience: 7+ years.",
            "resume_text": "Construction Project Manager, 8 years. Managed large-scale construction projects. Developed and maintained project schedules in Primavera. Controlled project budgets. Supervised construction sites. Mitigated risks effectively.",
            "relevance_score": 95
        },
        {
            "jd_text": "Construction Project Manager. Skills: Construction Management, Project Scheduling (Primavera/MS Project), Budget Control, Site Supervision, Risk Management. Experience: 7+ years.",
            "resume_text": "Construction Foreman, 10 years. Supervised crews on site. No project management or budgeting.",
            "relevance_score": 50
        },
        {
            "jd_text": "Construction Project Manager. Skills: Construction Management, Project Scheduling (Primavera/MS Project), Budget Control, Site Supervision, Risk Management. Experience: 7+ years.",
            "resume_text": "Civil Engineer, 5 years. Designed structures. No construction management.",
            "relevance_score": 30
        },

        # Example: Senior Research Scientist (Chemistry)
        {
            "jd_text": "Senior Research Scientist (Chemistry). Skills: Organic Synthesis, Reaction Optimization, Spectroscopy (NMR, Mass Spec), Drug Discovery, Publications. Experience: PhD + 5 years research.",
            "resume_text": "Senior Research Scientist, PhD + 6 years research. Led organic synthesis projects. Optimized complex reactions. Expert in NMR and Mass Spec. Contributed significantly to drug discovery. Extensive publications.",
            "relevance_score": 98
        },
        {
            "jd_text": "Senior Research Scientist (Chemistry). Skills: Organic Synthesis, Reaction Optimization, Spectroscopy (NMR, Mass Spec), Drug Discovery, Publications. Experience: PhD + 5 years research.",
            "resume_text": "Research Chemist, 3 years. Performed synthesis. Some analytical skills. No senior-level leadership or drug discovery focus.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Research Scientist (Chemistry). Skills: Organic Synthesis, Reaction Optimization, Spectroscopy (NMR, Mass Spec), Drug Discovery, Publications. Experience: PhD + 5 years research.",
            "resume_text": "Lab Manager, 10 years. Managed lab operations. No research or synthesis.",
            "relevance_score": 20
        },

        # Example: Clinical Project Manager
        {
            "jd_text": "Clinical Project Manager. Skills: Clinical Trial Management, ICH-GCP, CRO Management, Budget Oversight, Regulatory Submissions, Pharma/Biotech. Experience: 5+ years.",
            "resume_text": "Clinical Project Manager, 6 years. Managed phase II/III clinical trials. Ensured ICH-GCP compliance. Oversaw CROs and budgets. Prepared regulatory submissions. Strong pharma/biotech experience.",
            "relevance_score": 96
        },
        {
            "jd_text": "Clinical Project Manager. Skills: Clinical Trial Management, ICH-GCP, CRO Management, Budget Oversight, Regulatory Submissions, Pharma/Biotech. Experience: 5+ years.",
            "resume_text": "Clinical Research Coordinator, 3 years. Assisted with trial operations. No project management or budget oversight.",
            "relevance_score": 60
        },
        {
            "jd_text": "Clinical Project Manager. Skills: Clinical Trial Management, ICH-GCP, CRO Management, Budget Oversight, Regulatory Submissions, Pharma/Biotech. Experience: 5+ years.",
            "resume_text": "Project Manager (IT), 8 years. Managed IT projects. No clinical trial domain.",
            "relevance_score": 30
        },

        # Example: Special Education Teacher
        {
            "jd_text": "Special Education Teacher. Skills: Individualized Education Programs (IEPs), Differentiated Instruction, Behavior Management, Student Assessment, Collaboration. Experience: 2+ years.",
            "resume_text": "Special Education Teacher, 3 years. Developed and implemented IEPs. Provided differentiated instruction. Expert in behavior management. Conducted student assessments. Collaborated with parents and therapists.",
            "relevance_score": 95
        },
        {
            "jd_text": "Special Education Teacher. Skills: Individualized Education Programs (IEPs), Differentiated Instruction, Behavior Management, Student Assessment, Collaboration. Experience: 2+ years.",
            "resume_text": "General Education Teacher, 5 years. Taught regular classes. Some differentiated instruction. No IEP experience.",
            "relevance_score": 60
        },
        {
            "jd_text": "Special Education Teacher. Skills: Individualized Education Programs (IEPs), Differentiated Instruction, Behavior Management, Student Assessment, Collaboration. Experience: 2+ years.",
            "resume_text": "Counselor, 7 years. Provided student counseling. No classroom teaching or IEPs.",
            "relevance_score": 35
        },

        # Example: Financial Planning & Analysis (FP&A) Manager
        {
            "jd_text": "FP&A Manager. Skills: Financial Planning, Forecasting, Budgeting, Variance Analysis, Business Partnering, Financial Modeling, SAP/Hyperion. Experience: 7+ years.",
            "resume_text": "FP&A Manager, 8 years. Led financial planning cycles. Developed accurate forecasts and budgets. Performed detailed variance analysis. Acted as a key business partner. Expert in financial modeling. Proficient in SAP and Hyperion.",
            "relevance_score": 96
        },
        {
            "jd_text": "FP&A Manager. Skills: Financial Planning, Forecasting, Budgeting, Variance Analysis, Business Partnering, Financial Modeling, SAP/Hyperion. Experience: 7+ years.",
            "resume_text": "Senior Financial Analyst, 4 years. Assisted with forecasting. Some modeling. No managerial or business partnering leadership.",
            "relevance_score": 70
        },
        {
            "jd_text": "FP&A Manager. Skills: Financial Planning, Forecasting, Budgeting, Variance Analysis, Business Partnering, Financial Modeling, SAP/Hyperion. Experience: 7+ years.",
            "resume_text": "Accountant, 10 years. Managed general ledger. No FP&A experience.",
            "relevance_score": 30
        },

        # Example: IT Security Architect
        {
            "jd_text": "IT Security Architect. Skills: Security Architecture, Enterprise Security, Threat Modeling, Cloud Security, Identity & Access Management (IAM), Security Frameworks (NIST, TOGAF). Experience: 8+ years.",
            "resume_text": "IT Security Architect, 9 years. Designed and implemented enterprise security architectures. Conducted threat modeling. Expertise in cloud security and IAM. Applied NIST and TOGAF frameworks.",
            "relevance_score": 97
        },
        {
            "jd_text": "IT Security Architect. Skills: Security Architecture, Enterprise Security, Threat Modeling, Cloud Security, Identity & Access Management (IAM), Security Frameworks (NIST, TOGAF). Experience: 8+ years.",
            "resume_text": "Cybersecurity Engineer, 5 years. Implemented security controls. Some cloud security. No architecture design.",
            "relevance_score": 65
        },
        {
            "jd_text": "IT Security Architect. Skills: Security Architecture, Enterprise Security, Threat Modeling, Cloud Security, Identity & Access Management (IAM), Security Frameworks (NIST, TOGAF). Experience: 8+ years.",
            "resume_text": "Network Engineer, 10 years. Managed network infrastructure. Some security awareness. No security architecture focus.",
            "relevance_score": 40
        },

        # Example: Senior Mobile App Developer (Android)
        {
            "jd_text": "Senior Mobile App Developer (Android). Skills: Kotlin, Java, Android SDK, MVVM/MVI, RESTful APIs, Performance Optimization, Unit Testing. Experience: 5+ years.",
            "resume_text": "Senior Android Developer, 6 years. Developed high-performance Android apps in Kotlin and Java. Applied MVVM architecture. Integrated RESTful APIs. Focused on performance optimization and unit testing.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Mobile App Developer (Android). Skills: Kotlin, Java, Android SDK, MVVM/MVI, RESTful APIs, Performance Optimization, Unit Testing. Experience: 5+ years.",
            "resume_text": "Mobile App Developer (iOS), 4 years. Proficient in Swift. No Android experience.",
            "relevance_score": 30
        },
        {
            "jd_text": "Senior Mobile App Developer (Android). Skills: Kotlin, Java, Android SDK, MVVM/MVI, RESTful APIs, Performance Optimization, Unit Testing. Experience: 5+ years.",
            "resume_text": "Fullstack Web Developer, 7 years. Built web apps. Some mobile web. No native Android development.",
            "relevance_score": 45
        },

        # Example: Business Operations Manager
        {
            "jd_text": "Business Operations Manager. Skills: Operations Management, Process Optimization, P&L Management, Team Leadership, Cross-functional Collaboration, CRM. Experience: 7+ years.",
            "resume_text": "Business Operations Manager, 8 years. Oversaw daily operations. Drove process optimization initiatives. Managed P&L for business unit. Led large operations teams. Fostered cross-functional collaboration. Proficient in Salesforce CRM.",
            "relevance_score": 95
        },
        {
            "jd_text": "Business Operations Manager. Skills: Operations Management, Process Optimization, P&L Management, Team Leadership, Cross-functional Collaboration, CRM. Experience: 7+ years.",
            "resume_text": "Operations Coordinator, 4 years. Assisted with daily tasks. No managerial or P&L experience.",
            "relevance_score": 60
        },
        {
            "jd_text": "Business Operations Manager. Skills: Operations Management, Process Optimization, P&L Management, Team Leadership, Cross-functional Collaboration, CRM. Experience: 7+ years.",
            "resume_text": "Project Manager, 10 years. Managed projects. No direct operations management.",
            "relevance_score": 35
        },

        # Example: Senior Marketing Analyst
        {
            "jd_text": "Senior Marketing Analyst. Skills: Marketing Analytics, A/B Testing, SQL, Python (Pandas), Data Visualization (Tableau/Looker), Customer Segmentation. Experience: 4+ years.",
            "resume_text": "Senior Marketing Analyst, 5 years. Led marketing analytics initiatives. Designed and analyzed complex A/B tests. Proficient in SQL and Python for data manipulation. Created advanced dashboards in Tableau. Developed customer segmentation models.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Marketing Analyst. Skills: Marketing Analytics, A/B Testing, SQL, Python (Pandas), Data Visualization (Tableau/Looker), Customer Segmentation. Experience: 4+ years.",
            "resume_text": "Marketing Specialist, 2 years. Ran social media campaigns. Some analytics exposure. No advanced SQL or Python.",
            "relevance_score": 55
        },
        {
            "jd_text": "Senior Marketing Analyst. Skills: Marketing Analytics, A/B Testing, SQL, Python (Pandas), Data Visualization (Tableau/Looker), Customer Segmentation. Experience: 4+ years.",
            "resume_text": "Data Analyst, 6 years. Strong in SQL and Python. No specific marketing domain knowledge.",
            "relevance_score": 70
        },

        # Example: Supply Chain Analyst (Entry Level)
        {
            "jd_text": "Supply Chain Analyst (Entry Level). Skills: Excel, Data Analysis, Logistics, Inventory Basics, Communication. Experience: 0-2 years.",
            "resume_text": "Recent graduate in Supply Chain Management. Strong Excel skills. Completed projects involving logistics and basic inventory analysis. Good communication.",
            "relevance_score": 80
        },
        {
            "jd_text": "Supply Chain Analyst (Entry Level). Skills: Excel, Data Analysis, Logistics, Inventory Basics, Communication. Experience: 0-2 years.",
            "resume_text": "Retail Associate, 4 years. Managed store inventory. No formal supply chain education or data analysis.",
            "relevance_score": 20
        },
        {
            "jd_text": "Supply Chain Analyst (Entry Level). Skills: Excel, Data Analysis, Logistics, Inventory Basics, Communication. Experience: 0-2 years.",
            "resume_text": "Business Student, internship in marketing. Some Excel. No supply chain focus.",
            "relevance_score": 10
        },

        # Example: Senior Mechanical Engineer
        {
            "jd_text": "Senior Mechanical Engineer. Skills: Advanced CAD (SolidWorks/CATIA), FEA, Design for Manufacturing (DFM), Fluid Dynamics, Project Leadership. Experience: 6+ years.",
            "resume_text": "Senior Mechanical Engineer, 7 years. Expert in SolidWorks and CATIA for complex designs. Performed advanced FEA. Applied DFM principles. Strong in fluid dynamics. Led engineering design projects.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Mechanical Engineer. Skills: Advanced CAD (SolidWorks/CATIA), FEA, Design for Manufacturing (DFM), Fluid Dynamics, Project Leadership. Experience: 6+ years.",
            "resume_text": "Mechanical Design Engineer, 3 years. Proficient in SolidWorks. Some FEA. No senior-level leadership or advanced concepts.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Mechanical Engineer. Skills: Advanced CAD (SolidWorks/CATIA), FEA, Design for Manufacturing (DFM), Fluid Dynamics, Project Leadership. Experience: 6+ years.",
            "resume_text": "Manufacturing Engineer, 10 years. Focused on production. No design or FEA.",
            "relevance_score": 30
        },

        # Example: Electrical Design Engineer
        {
            "jd_text": "Electrical Design Engineer. Skills: Circuit Design, PCB Layout, Altium Designer, Analog/Digital Electronics, Power Management. Experience: 3+ years.",
            "resume_text": "Electrical Design Engineer, 4 years. Designed analog and digital circuits. Performed PCB layout using Altium Designer. Expertise in power management. Developed schematics.",
            "relevance_score": 94
        },
        {
            "jd_text": "Electrical Design Engineer. Skills: Circuit Design, PCB Layout, Altium Designer, Analog/Digital Electronics, Power Management. Experience: 3+ years.",
            "resume_text": "Electrical Technician, 5 years. Assembled and tested circuits. No design experience.",
            "relevance_score": 40
        },
        {
            "jd_text": "Electrical Design Engineer. Skills: Circuit Design, PCB Layout, Altium Designer, Analog/Digital Electronics, Power Management. Experience: 3+ years.",
            "resume_text": "Software Engineer, 6 years. No electrical engineering skills.",
            "relevance_score": 15
        },

        # Example: Structural Engineer (Bridge Design)
        {
            "jd_text": "Structural Engineer (Bridge Design). Skills: Bridge Design, Structural Analysis, AASHTO LRFD, AutoCAD Civil 3D, Bridge Software (MIDAS/CSI Bridge). Experience: 4+ years.",
            "resume_text": "Structural Engineer, 5 years. Specialized in bridge design. Performed complex structural analysis. Proficient in AASHTO LRFD. Used AutoCAD Civil 3D and MIDAS for bridge projects.",
            "relevance_score": 95
        },
        {
            "jd_text": "Structural Engineer (Bridge Design). Skills: Bridge Design, Structural Analysis, AASHTO LRFD, AutoCAD Civil 3D, Bridge Software (MIDAS/CSI Bridge). Experience: 4+ years.",
            "resume_text": "Building Structural Engineer, 6 years. Designed building structures. Some AutoCAD. No bridge specific experience.",
            "relevance_score": 60
        },
        {
            "jd_text": "Structural Engineer (Bridge Design). Skills: Bridge Design, Structural Analysis, AASHTO LRFD, AutoCAD Civil 3D, Bridge Software (MIDAS/CSI Bridge). Experience: 4+ years.",
            "resume_text": "Civil Engineer (Water Resources), 7 years. Focused on water projects. No structural design.",
            "relevance_score": 25
        },

        # Example: Analytical Chemist
        {
            "jd_text": "Analytical Chemist. Skills: Method Development, HPLC, GC-MS, Mass Spectrometry, Data Interpretation, Quality Control. Experience: 3+ years.",
            "resume_text": "Analytical Chemist, 4 years. Developed and validated analytical methods. Expert in HPLC and GC-MS. Performed mass spectrometry. Strong in data interpretation and quality control.",
            "relevance_score": 93
        },
        {
            "jd_text": "Analytical Chemist. Skills: Method Development, HPLC, GC-MS, Mass Spectrometry, Data Interpretation, Quality Control. Experience: 3+ years.",
            "resume_text": "Lab Technician, 2 years. Operated analytical instruments. No method development or data interpretation.",
            "relevance_score": 50
        },
        {
            "jd_text": "Analytical Chemist. Skills: Method Development, HPLC, GC-MS, Mass Spectrometry, Data Interpretation, Quality Control. Experience: 3+ years.",
            "resume_text": "Research Chemist, 5 years. Focused on synthesis. Some analytical exposure. No dedicated analytical role.",
            "relevance_score": 65
        },

        # Example: Clinical Research Associate (CRA)
        {
            "jd_text": "Clinical Research Associate (CRA). Skills: Clinical Monitoring, ICH-GCP, Site Management, Regulatory Documents, Pharma/CRO. Experience: 2+ years.",
            "resume_text": "Clinical Research Associate, 3 years. Conducted on-site clinical monitoring. Ensured ICH-GCP compliance. Managed clinical sites. Reviewed regulatory documents. Experience in both pharma and CRO settings.",
            "relevance_score": 94
        },
        {
            "jd_text": "Clinical Research Associate (CRA). Skills: Clinical Monitoring, ICH-GCP, Site Management, Regulatory Documents, Pharma/CRO. Experience: 2+ years.",
            "resume_text": "Clinical Research Coordinator, 1 year. Assisted with trial setup. No independent monitoring.",
            "relevance_score": 60
        },
        {
            "jd_text": "Clinical Research Associate (CRA). Skills: Clinical Monitoring, ICH-GCP, Site Management, Regulatory Documents, Pharma/CRO. Experience: 2+ years.",
            "resume_text": "Registered Nurse, 5 years. Provided patient care. No clinical research.",
            "relevance_score": 30
        },

        # Example: University Professor (Computer Science)
        {
            "jd_text": "University Professor (Computer Science). Skills: Research, Teaching (Undergraduate/Graduate), Publications, Grant Writing, Algorithms, Data Structures. Experience: PhD + 5 years academia.",
            "resume_text": "Professor of Computer Science, PhD + 6 years academia. Conducted cutting-edge research. Taught undergraduate and graduate courses in Algorithms and Data Structures. Published extensively. Secured research grants.",
            "relevance_score": 97
        },
        {
            "jd_text": "University Professor (Computer Science). Skills: Research, Teaching (Undergraduate/Graduate), Publications, Grant Writing, Algorithms, Data Structures. Experience: PhD + 5 years academia.",
            "resume_text": "Software Engineer, 10 years. Strong in algorithms. No teaching or research publications.",
            "relevance_score": 40
        },
        {
            "jd_text": "University Professor (Computer Science). Skills: Research, Teaching (Undergraduate/Graduate), Publications, Grant Writing, Algorithms, Data Structures. Experience: PhD + 5 years academia.",
            "resume_text": "High School Teacher, 15 years. Taught math. No university-level or research.",
            "relevance_score": 20
        },

        # Example: Financial Auditor
        {
            "jd_text": "Financial Auditor. Skills: Audit Planning, GAAP/IFRS, Internal Controls, Financial Statements, Data Analysis (Excel/ACL), Compliance. Experience: 3+ years.",
            "resume_text": "Financial Auditor, 4 years. Led audit planning. Ensured compliance with GAAP. Evaluated internal controls. Analyzed financial statements using Excel and ACL. Strong in compliance.",
            "relevance_score": 92
        },
        {
            "jd_text": "Financial Auditor. Skills: Audit Planning, GAAP/IFRS, Internal Controls, Financial Statements, Data Analysis (Excel/ACL), Compliance. Experience: 3+ years.",
            "resume_text": "Accountant, 5 years. Prepared financial reports. No audit experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "Financial Auditor. Skills: Audit Planning, GAAP/IFRS, Internal Controls, Financial Statements, Data Analysis (Excel/ACL), Compliance. Experience: 3+ years.",
            "resume_text": "Data Analyst, 3 years. Strong in Excel and SQL. No audit or GAAP knowledge.",
            "relevance_score": 60
        },

        # Example: Marketing Director
        {
            "jd_text": "Marketing Director. Skills: Marketing Strategy, Brand Development, Digital Marketing, Team Leadership, Budget Management, Market Research. Experience: 8+ years.",
            "resume_text": "Marketing Director, 10 years. Developed and executed comprehensive marketing strategies. Led brand development initiatives. Oversaw digital marketing campaigns. Managed large teams and multi-million dollar budgets. Conducted extensive market research.",
            "relevance_score": 96
        },
        {
            "jd_text": "Marketing Director. Skills: Marketing Strategy, Brand Development, Digital Marketing, Team Leadership, Budget Management, Market Research. Experience: 8+ years.",
            "resume_text": "Marketing Manager, 5 years. Managed campaigns. Some team leadership. No director-level strategy.",
            "relevance_score": 70
        },
        {
            "jd_text": "Marketing Director. Skills: Marketing Strategy, Brand Development, Digital Marketing, Team Leadership, Budget Management, Market Research. Experience: 8+ years.",
            "resume_text": "Sales Director, 12 years. Led sales teams. No marketing expertise.",
            "relevance_score": 30
        },

        # Example: Manufacturing Engineer
        {
            "jd_text": "Manufacturing Engineer. Skills: Process Optimization, Lean Manufacturing, Six Sigma, CAD, Production Planning, Quality Control. Experience: 3+ years.",
            "resume_text": "Manufacturing Engineer, 4 years. Optimized manufacturing processes using Lean and Six Sigma. Proficient in CAD. Developed production plans. Implemented quality control measures.",
            "relevance_score": 93
        },
        {
            "jd_text": "Manufacturing Engineer. Skills: Process Optimization, Lean Manufacturing, Six Sigma, CAD, Production Planning, Quality Control. Experience: 3+ years.",
            "resume_text": "Mechanical Design Engineer, 5 years. Focused on product design. Some CAD. No manufacturing process focus.",
            "relevance_score": 55
        },
        {
            "jd_text": "Manufacturing Engineer. Skills: Process Optimization, Lean Manufacturing, Six Sigma, CAD, Production Planning, Quality Control. Experience: 3+ years.",
            "resume_text": "Production Supervisor, 8 years. Managed production lines. No engineering background.",
            "relevance_score": 30
        },

        # Example: Salesforce Administrator
        {
            "jd_text": "Salesforce Administrator. Skills: Salesforce Administration, Sales Cloud, Service Cloud, Reports & Dashboards, User Management, Process Automation (Flow/Process Builder). Experience: 2+ years.",
            "resume_text": "Salesforce Administrator, 3 years. Managed Sales Cloud and Service Cloud. Created complex reports and dashboards. Performed user management. Automated processes with Flow and Process Builder.",
            "relevance_score": 90
        },
        {
            "jd_text": "Salesforce Administrator. Skills: Salesforce Administration, Sales Cloud, Service Cloud, Reports & Dashboards, User Management, Process Automation (Flow/Process Builder). Experience: 2+ years.",
            "resume_text": "CRM Support Specialist, 4 years. Supported CRM users. No Salesforce administration.",
            "relevance_score": 40
        },
        {
            "jd_text": "Salesforce Administrator. Skills: Salesforce Administration, Sales Cloud, Service Cloud, Reports & Dashboards, User Management, Process Automation (Flow/Process Builder). Experience: 2+ years.",
            "resume_text": "Business Analyst, 5 years. Gathered requirements. No Salesforce hands-on.",
            "relevance_score": 20
        },

        # Example: Data Entry Specialist
        {
            "jd_text": "Data Entry Specialist. Skills: Accurate Data Entry, Microsoft Excel, Data Validation, Attention to Detail, Typing Speed. Experience: 1+ years.",
            "resume_text": "Data Entry Specialist, 2 years. Consistently performed accurate data entry. Proficient in Microsoft Excel. Performed data validation. High attention to detail and fast typing speed.",
            "relevance_score": 85
        },
        {
            "jd_text": "Data Entry Specialist. Skills: Accurate Data Entry, Microsoft Excel, Data Validation, Attention to Detail, Typing Speed. Experience: 1+ years.",
            "resume_text": "Administrative Assistant, 3 years. Used Excel for basic tasks. Some data entry. Not a primary focus.",
            "relevance_score": 60
        },
        {
            "jd_text": "Data Entry Specialist. Skills: Accurate Data Entry, Microsoft Excel, Data Validation, Attention to Detail, Typing Speed. Experience: 1+ years.",
            "resume_text": "Data Analyst, 4 years. Focused on complex analysis. Data entry is too basic.",
            "relevance_score": 30
        },

        # Example: Executive Assistant
        {
            "jd_text": "Executive Assistant. Skills: Calendar Management, Travel Coordination, Expense Reporting, Communication, Microsoft Office Suite, Discretion. Experience: 5+ years.",
            "resume_text": "Executive Assistant, 6 years. Managed complex calendars and travel for executives. Handled expense reporting. Excellent communication. Expert in Microsoft Office Suite. Maintained high level of discretion.",
            "relevance_score": 90
        },
        {
            "jd_text": "Executive Assistant. Skills: Calendar Management, Travel Coordination, Expense Reporting, Communication, Microsoft Office Suite, Discretion. Experience: 5+ years.",
            "resume_text": "Administrative Assistant, 3 years. Managed office supplies. Some scheduling. No executive level support.",
            "relevance_score": 50
        },
        {
            "jd_text": "Executive Assistant. Skills: Calendar Management, Travel Coordination, Expense Reporting, Communication, Microsoft Office Suite, Discretion. Experience: 5+ years.",
            "resume_text": "Project Manager, 8 years. Managed projects. No administrative support role.",
            "relevance_score": 20
        },

        # Example: Cloud Solutions Architect (Entry Level)
        {
            "jd_text": "Cloud Solutions Architect (Entry Level). Skills: AWS/Azure/GCP Basics, Cloud Concepts, Networking Fundamentals, Linux, Scripting (Python/Bash). Experience: 0-2 years.",
            "resume_text": "Recent graduate with a Master's in Cloud Computing. Strong understanding of AWS and Azure concepts. Solid networking fundamentals. Proficient in Linux and Python scripting. Completed cloud architecture projects.",
            "relevance_score": 85
        },
        {
            "jd_text": "Cloud Solutions Architect (Entry Level). Skills: AWS/Azure/GCP Basics, Cloud Concepts, Networking Fundamentals, Linux, Scripting (Python/Bash). Experience: 0-2 years.",
            "resume_text": "System Administrator, 5 years. Managed on-premise servers. No cloud experience.",
            "relevance_score": 30
        },
        {
            "jd_text": "Cloud Solutions Architect (Entry Level). Skills: AWS/Azure/GCP Basics, Cloud Concepts, Networking Fundamentals, Linux, Scripting (Python/Bash). Experience: 0-2 years.",
            "resume_text": "Software Developer, 3 years. Built web apps. Some AWS for deployment. No architecture focus.",
            "relevance_score": 60
        },

        # Example: Senior Data Engineer
        {
            "jd_text": "Senior Data Engineer. Skills: Spark, Kafka, Distributed Systems, Data Lake, Data Warehouse, ETL Optimization, Cloud Data Services (AWS Glue/Databricks). Experience: 6+ years.",
            "resume_text": "Senior Data Engineer, 7 years. Designed and built large-scale data pipelines using Spark and Kafka for distributed systems. Managed data lakes and data warehouses. Optimized complex ETL processes. Expert in AWS Glue and Databricks.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior Data Engineer. Skills: Spark, Kafka, Distributed Systems, Data Lake, Data Warehouse, ETL Optimization, Cloud Data Services (AWS Glue/Databricks). Experience: 6+ years.",
            "resume_text": "Data Engineer, 3 years. Built basic ETLs. Some Spark. No distributed systems or optimization.",
            "relevance_score": 70
        },
        {
            "jd_text": "Senior Data Engineer. Skills: Spark, Kafka, Distributed Systems, Data Lake, Data Warehouse, ETL Optimization, Cloud Data Services (AWS Glue/Databricks). Experience: 6+ years.",
            "resume_text": "Software Engineer, 8 years. Built backend services. No data engineering focus.",
            "relevance_score": 35
        },

        # Example: Digital Content Creator
        {
            "jd_text": "Digital Content Creator. Skills: Video Editing (Adobe Premiere), Graphic Design (Photoshop/Illustrator), Copywriting, Social Media, Storytelling. Experience: 2+ years.",
            "resume_text": "Digital Content Creator, 3 years. Produced engaging video content using Adobe Premiere. Designed graphics in Photoshop. Wrote compelling copy for social media. Strong storytelling ability.",
            "relevance_score": 90
        },
        {
            "jd_text": "Digital Content Creator. Skills: Video Editing (Adobe Premiere), Graphic Design (Photoshop/Illustrator), Copywriting, Social Media, Storytelling. Experience: 2+ years.",
            "resume_text": "Graphic Designer, 5 years. Expert in Photoshop. No video or copywriting.",
            "relevance_score": 60
        },
        {
            "jd_text": "Digital Content Creator. Skills: Video Editing (Adobe Premiere), Graphic Design (Photoshop/Illustrator), Copywriting, Social Media, Storytelling. Experience: 2+ years.",
            "resume_text": "Journalist, 8 years. Excellent writer. No visual media creation.",
            "relevance_score": 30
        },

        # Example: Customer Success Manager
        {
            "jd_text": "Customer Success Manager. Skills: Customer Relationship Management, SaaS Product Knowledge, Churn Reduction, Onboarding, Account Management. Experience: 3+ years.",
            "resume_text": "Customer Success Manager, 4 years. Managed key customer relationships for SaaS products. Reduced churn by 15%. Led onboarding processes. Strong in account management.",
            "relevance_score": 92
        },
        {
            "jd_text": "Customer Success Manager. Skills: Customer Relationship Management, SaaS Product Knowledge, Churn Reduction, Onboarding, Account Management. Experience: 3+ years.",
            "resume_text": "Customer Service Rep, 5 years. Handled inquiries. No proactive success management.",
            "relevance_score": 40
        },
        {
            "jd_text": "Customer Success Manager. Skills: Customer Relationship Management, SaaS Product Knowledge, Churn Reduction, Onboarding, Account Management. Experience: 3+ years.",
            "resume_text": "Sales Manager, 7 years. Focused on sales. No post-sales customer success.",
            "relevance_score": 30
        },

        # Example: Regulatory Affairs Specialist
        {
            "jd_text": "Regulatory Affairs Specialist. Skills: FDA Regulations, Medical Device/Pharma, Regulatory Submissions (510(k), PMA), Quality Management Systems. Experience: 4+ years.",
            "resume_text": "Regulatory Affairs Specialist, 5 years. Ensured compliance with FDA regulations for medical devices. Prepared and submitted 510(k) applications. Familiar with quality management systems.",
            "relevance_score": 95
        },
        {
            "jd_text": "Regulatory Affairs Specialist. Skills: FDA Regulations, Medical Device/Pharma, Regulatory Submissions (510(k), PMA), Quality Management Systems. Experience: 4+ years.",
            "resume_text": "Quality Assurance Engineer, 6 years. Focused on product quality. Some regulatory awareness. No direct regulatory submissions.",
            "relevance_score": 60
        },
        {
            "jd_text": "Regulatory Affairs Specialist. Skills: FDA Regulations, Medical Device/Pharma, Regulatory Submissions (510(k), PMA), Quality Management Systems. Experience: 4+ years.",
            "resume_text": "Clinical Research Coordinator, 3 years. Managed trials. No regulatory affairs.",
            "relevance_score": 25
        },

        # Example: UX Writer (Entry Level)
        {
            "jd_text": "UX Writer (Entry Level). Skills: Clear & Concise Writing, Microcopy, User-Centered Design Principles, Editing, Collaboration. Experience: 0-2 years.",
            "resume_text": "Recent Communications graduate. Strong in clear and concise writing. Understands user-centered design principles. Internship in content editing. Eager to learn UX writing.",
            "relevance_score": 80
        },
        {
            "jd_text": "UX Writer (Entry Level). Skills: Clear & Concise Writing, Microcopy, User-Centered Design Principles, Editing, Collaboration. Experience: 0-2 years.",
            "resume_text": "Copywriter, 3 years. Wrote marketing copy. No UX focus.",
            "relevance_score": 40
        },
        {
            "jd_text": "UX Writer (Entry Level). Skills: Clear & Concise Writing, Microcopy, User-Centered Design Principles, Editing, Collaboration. Experience: 0-2 years.",
            "resume_text": "Technical Writer, 5 years. Wrote manuals. No microcopy or UX.",
            "relevance_score": 50
        },

        # Example: Investment Banking Analyst (Entry Level)
        {
            "jd_text": "Investment Banking Analyst (Entry Level). Skills: Financial Modeling, Valuation, Excel, PowerPoint, Corporate Finance, Communication. Experience: 0-2 years.",
            "resume_text": "Recent Finance graduate. Strong financial modeling and valuation skills from coursework. Proficient in Excel and PowerPoint. Solid understanding of corporate finance. Excellent communication.",
            "relevance_score": 85
        },
        {
            "jd_text": "Investment Banking Analyst (Entry Level). Skills: Financial Modeling, Valuation, Excel, PowerPoint, Corporate Finance, Communication. Experience: 0-2 years.",
            "resume_text": "Financial Analyst, 3 years. Focused on budgeting. No investment banking.",
            "relevance_score": 50
        },
        {
            "jd_text": "Investment Banking Analyst (Entry Level). Skills: Financial Modeling, Valuation, Excel, PowerPoint, Corporate Finance, Communication. Experience: 0-2 years.",
            "resume_text": "Accountant, 5 years. No financial modeling or corporate finance.",
            "relevance_score": 15
        },

        # Example: Data Science Intern
        {
            "jd_text": "Data Science Intern. Skills: Python (Pandas), SQL, Basic Machine Learning, Data Visualization, Statistics. Experience: 0 years.",
            "resume_text": "Computer Science student. Strong in Python (Pandas) and SQL. Completed projects using basic machine learning algorithms. Created data visualizations. Good grasp of statistics.",
            "relevance_score": 80
        },
        {
            "jd_text": "Data Science Intern. Skills: Python (Pandas), SQL, Basic Machine Learning, Data Visualization, Statistics. Experience: 0 years.",
            "resume_text": "Marketing Intern. Some Excel. No programming or ML.",
            "relevance_score": 5
        },
        {
            "jd_text": "Data Science Intern. Skills: Python (Pandas), SQL, Basic Machine Learning, Data Visualization, Statistics. Experience: 0 years.",
            "resume_text": "Software Engineering student. Strong in Java. No data science coursework.",
            "relevance_score": 30
        },

        # Example: Junior Software Engineer
        {
            "jd_text": "Junior Software Engineer. Skills: Python/Java, Data Structures, Algorithms, Git, REST APIs, Unit Testing. Experience: 0-2 years.",
            "resume_text": "Recent Computer Science graduate. Proficient in Python, data structures, and algorithms. Familiar with Git and REST APIs. Wrote unit tests for projects. Eager to contribute.",
            "relevance_score": 85
        },
        {
            "jd_text": "Junior Software Engineer. Skills: Python/Java, Data Structures, Algorithms, Git, REST APIs, Unit Testing. Experience: 0-2 years.",
            "resume_text": "IT Support Specialist, 3 years. No programming experience.",
            "relevance_score": 10
        },
        {
            "jd_text": "Junior Software Engineer. Skills: Python/Java, Data Structures, Algorithms, Git, REST APIs, Unit Testing. Experience: 0-2 years.",
            "resume_text": "QA Tester, 5 years. Automated tests. No development role.",
            "relevance_score": 40
        },

        # Example: Junior Marketing Specialist
        {
            "jd_text": "Junior Marketing Specialist. Skills: Social Media, Content Creation, Email Marketing, Google Analytics, Canva. Experience: 0-2 years.",
            "resume_text": "Recent Marketing graduate. Managed social media accounts for student clubs. Created engaging content. Assisted with email marketing campaigns. Basic Google Analytics and Canva skills.",
            "relevance_score": 80
        },
        {
            "jd_text": "Junior Marketing Specialist. Skills: Social Media, Content Creation, Email Marketing, Google Analytics, Canva. Experience: 0-2 years.",
            "resume_text": "Sales Assistant, 1 year. No marketing experience.",
            "relevance_score": 5
        },
        {
            "jd_text": "Junior Marketing Specialist. Skills: Social Media, Content Creation, Email Marketing, Google Analytics, Canva. Experience: 0-2 years.",
            "resume_text": "Graphic Designer, 3 years. Expert in design. No marketing strategy.",
            "relevance_score": 30
        },

        # Example: Junior HR Generalist
        {
            "jd_text": "Junior HR Generalist. Skills: Employee Onboarding, HRIS Data Entry, HR Policy, Recruitment Support, Communication. Experience: 0-2 years.",
            "resume_text": "Recent HR graduate. Assisted with employee onboarding. Proficient in HRIS data entry. Familiar with HR policies. Provided recruitment support. Strong communication skills.",
            "relevance_score": 85
        },
        {
            "jd_text": "Junior HR Generalist. Skills: Employee Onboarding, HRIS Data Entry, HR Policy, Recruitment Support, Communication. Experience: 0-2 years.",
            "resume_text": "Administrative Assistant, 4 years. Managed office tasks. No HR specific experience.",
            "relevance_score": 20
        },
        {
            "jd_text": "Junior HR Generalist. Skills: Employee Onboarding, HRIS Data Entry, HR Policy, Recruitment Support, Communication. Experience: 0-2 years.",
            "resume_text": "Recruitment Coordinator, 1 year. Focused on sourcing. No generalist duties.",
            "relevance_score": 60
        },

        # Example: Junior UX/UI Designer
        {
            "jd_text": "Junior UX/UI Designer. Skills: Figma, Wireframing, Prototyping, User Flows, Basic Usability Testing, UI Principles. Experience: 0-2 years.",
            "resume_text": "Recent Design graduate. Proficient in Figma for wireframing and prototyping. Created user flows. Conducted basic usability testing. Strong understanding of UI principles. Portfolio available.",
            "relevance_score": 88
        },
        {
            "jd_text": "Junior UX/UI Designer. Skills: Figma, Wireframing, Prototyping, User Flows, Basic Usability Testing, UI Principles. Experience: 0-2 years.",
            "resume_text": "Graphic Designer, 3 years. Focused on branding. No UX/UI process.",
            "relevance_score": 30
        },
        {
            "jd_text": "Junior UX/UI Designer. Skills: Figma, Wireframing, Prototyping, User Flows, Basic Usability Testing, UI Principles. Experience: 0-2 years.",
            "resume_text": "Web Developer, 2 years. Built responsive websites. Some UI awareness. No design tools.",
            "relevance_score": 50
        },

        # Example: Junior Cloud Engineer
        {
            "jd_text": "Junior Cloud Engineer. Skills: AWS/Azure Basics, Linux, Scripting (Bash/Python), Networking Fundamentals, Cloud Monitoring. Experience: 0-2 years.",
            "resume_text": "Recent IT graduate. Strong understanding of AWS basics. Proficient in Linux and Bash scripting. Solid networking fundamentals. Familiar with cloud monitoring tools. Completed cloud projects.",
            "relevance_score": 85
        },
        {
            "jd_text": "Junior Cloud Engineer. Skills: AWS/Azure Basics, Linux, Scripting (Bash/Python), Networking Fundamentals, Cloud Monitoring. Experience: 0-2 years.",
            "resume_text": "System Administrator, 5 years. Managed on-premise servers. No cloud experience.",
            "relevance_score": 20
        },
        {
            "jd_text": "Junior Cloud Engineer. Skills: AWS/Azure Basics, Linux, Scripting (Bash/Python), Networking Fundamentals, Cloud Monitoring. Experience: 0-2 years.",
            "resume_text": "DevOps Intern. Some CI/CD. No deep cloud infrastructure.",
            "relevance_score": 60
        },

        # Example: Junior Cybersecurity Analyst
        {
            "jd_text": "Junior Cybersecurity Analyst. Skills: Network Security, Incident Response Basics, Vulnerability Scanning, Security Tools, Linux. Experience: 0-2 years.",
            "resume_text": "Recent Cybersecurity graduate. Strong in network security fundamentals. Familiar with incident response basics and vulnerability scanning. Used various security tools. Proficient in Linux.",
            "relevance_score": 88
        },
        {
            "jd_text": "Junior Cybersecurity Analyst. Skills: Network Security, Incident Response Basics, Vulnerability Scanning, Security Tools, Linux. Experience: 0-2 years.",
            "resume_text": "IT Helpdesk, 3 years. Resolved user issues. Some security awareness. No specific cybersecurity analysis.",
            "relevance_score": 30
        },
        {
            "jd_text": "Junior Cybersecurity Analyst. Skills: Network Security, Incident Response Basics, Vulnerability Scanning, Security Tools, Linux. Experience: 0-2 years.",
            "resume_text": "Software Developer, 4 years. Built secure code. No security operations.",
            "relevance_score": 50
        },

        # Example: Junior Mobile App Developer
        {
            "jd_text": "Junior Mobile App Developer. Skills: iOS/Android SDK, Swift/Kotlin, RESTful APIs, UI/UX Principles, Debugging. Experience: 0-2 years.",
            "resume_text": "Recent Computer Science graduate. Developed mobile apps using iOS SDK and Swift for personal projects. Integrated RESTful APIs. Good understanding of UI/UX principles. Strong debugging skills.",
            "relevance_score": 85
        },
        {
            "jd_text": "Junior Mobile App Developer. Skills: iOS/Android SDK, Swift/Kotlin, RESTful APIs, UI/UX Principles, Debugging. Experience: 0-2 years.",
            "resume_text": "Web Developer, 3 years. Built responsive websites. No native mobile development.",
            "relevance_score": 20
        },
        {
            "jd_text": "Junior Mobile App Developer. Skills: iOS/Android SDK, Swift/Kotlin, RESTful APIs, UI/UX Principles, Debugging. Experience: 0-2 years.",
            "resume_text": "Game Developer, 2 years. Focused on PC games. No mobile platform.",
            "relevance_score": 40
        },

        # Example: Junior Project Manager
        {
            "jd_text": "Junior Project Manager. Skills: Project Coordination, Scheduling, Communication, Stakeholder Support, Risk Identification, Microsoft Office. Experience: 0-2 years.",
            "resume_text": "Recent Business graduate. Assisted with project coordination and scheduling in internships. Strong communication skills. Supported stakeholders. Identified potential risks. Proficient in Microsoft Office.",
            "relevance_score": 80
        },
        {
            "jd_text": "Junior Project Manager. Skills: Project Coordination, Scheduling, Communication, Stakeholder Support, Risk Identification, Microsoft Office. Experience: 0-2 years.",
            "resume_text": "Administrative Assistant, 5 years. Organized office tasks. No project management.",
            "relevance_score": 30
        },
        {
            "jd_text": "Junior Project Manager. Skills: Project Coordination, Scheduling, Communication, Stakeholder Support, Risk Identification, Microsoft Office. Experience: 0-2 years.",
            "resume_text": "Team Lead (non-project), 4 years. Led a small team. No formal project management.",
            "relevance_score": 50
        },
    
        {
            "jd_text": "Senior Data Analyst. Skills: Advanced SQL, Python (Pandas, SciPy), Data Storytelling, Power BI, BigQuery, Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Senior Data Analyst with 6 years experience. Expert in SQL and Python for complex data analysis. Developed interactive dashboards in Power BI. Presented insights to executive stakeholders. Worked with BigQuery.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior Data Analyst. Skills: Advanced SQL, Python (Pandas, SciPy), Data Storytelling, Power BI, BigQuery, Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Data Analyst, 3 years. Proficient in SQL and Excel. Basic Python. No experience with BigQuery or senior stakeholder management.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Data Analyst. Skills: Advanced SQL, Python (Pandas, SciPy), Data Storytelling, Power BI, BigQuery, Stakeholder Management. Experience: 5+ years.",
            "resume_text": "Marketing Analyst, 4 years. Focused on campaign performance. Used Google Analytics. Some Excel. No SQL or Python.",
            "relevance_score": 30
        },

        # Example: Fullstack Developer
        {
            "jd_text": "Fullstack Developer. Skills: React, Node.js, Express, MongoDB, AWS Lambda, REST APIs, Git. Experience: 3+ years.",
            "resume_text": "Fullstack Developer with 4 years experience. Built responsive UIs with React. Developed backend APIs with Node.js/Express and MongoDB. Deployed serverless functions on AWS Lambda. Strong Git.",
            "relevance_score": 94
        },
        {
            "jd_text": "Fullstack Developer. Skills: React, Node.js, Express, MongoDB, AWS Lambda, REST APIs, Git. Experience: 3+ years.",
            "resume_text": "Frontend Developer, 5 years. Expert in React. No backend or cloud experience.",
            "relevance_score": 65
        },
        {
            "jd_text": "Fullstack Developer. Skills: React, Node.js, Express, MongoDB, AWS Lambda, REST APIs, Git. Experience: 3+ years.",
            "resume_text": "Backend Java Developer, 6 years. Built Spring Boot microservices. No frontend experience.",
            "relevance_score": 40
        },

        # Example: UX Researcher (Senior)
        {
            "jd_text": "Senior UX Researcher. Skills: Mixed Methods Research, Ethnography, A/B Testing, Statistical Analysis, User Journey Mapping, Workshop Facilitation. Experience: 5+ years.",
            "resume_text": "Lead UX Researcher, 7 years. Conducted mixed-methods research including ethnography and A/B testing. Strong in statistical analysis. Led user journey mapping workshops. Mentored junior researchers.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior UX Researcher. Skills: Mixed Methods Research, Ethnography, A/B Testing, Statistical Analysis, User Journey Mapping, Workshop Facilitation. Experience: 5+ years.",
            "resume_text": "Junior UX Researcher, 2 years. Conducted usability tests. Assisted with surveys. No senior-level experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "Senior UX Researcher. Skills: Mixed Methods Research, Ethnography, A/B Testing, Statistical Analysis, User Journey Mapping, Workshop Facilitation. Experience: 5+ years.",
            "resume_text": "Market Research Analyst, 8 years. Designed and analyzed large-scale surveys. No specific UX research methodologies.",
            "relevance_score": 45
        },

        # Example: Cloud Security Engineer
        {
            "jd_text": "Cloud Security Engineer. Skills: AWS Security, Azure Security, GCP Security, IAM, Network Security, Compliance (NIST, ISO 27001), Scripting (Python). Experience: 4+ years.",
            "resume_text": "Cloud Security Engineer with 5 years. Implemented security controls on AWS and Azure. Configured IAM policies. Strong in network security. Ensured compliance with NIST and ISO 27001. Wrote Python scripts for automation.",
            "relevance_score": 95
        },
        {
            "jd_text": "Cloud Security Engineer. Skills: AWS Security, Azure Security, GCP Security, IAM, Network Security, Compliance (NIST, ISO 27001), Scripting (Python). Experience: 4+ years.",
            "resume_text": "Traditional Security Analyst, 6 years. Managed on-premise firewalls and IDS. No cloud-specific security experience.",
            "relevance_score": 40
        },
        {
            "jd_text": "Cloud Security Engineer. Skills: AWS Security, Azure Security, GCP Security, IAM, Network Security, Compliance (NIST, ISO 27001), Scripting (Python). Experience: 4+ years.",
            "resume_text": "DevOps Engineer, 3 years. Deployed applications to AWS. Some understanding of security groups. No dedicated security role.",
            "relevance_score": 60
        },

        # Example: Machine Learning Engineer
        {
            "jd_text": "Machine Learning Engineer. Skills: Python, TensorFlow/PyTorch, MLOps, Distributed ML, AWS SageMaker, Data Pipelines. Experience: 4+ years.",
            "resume_text": "ML Engineer with 5 years. Developed and deployed ML models in Python using TensorFlow. Built MLOps pipelines. Experience with distributed ML and AWS SageMaker. Managed data pipelines.",
            "relevance_score": 97
        },
        {
            "jd_text": "Machine Learning Engineer. Skills: Python, TensorFlow/PyTorch, MLOps, Distributed ML, AWS SageMaker, Data Pipelines. Experience: 4+ years.",
            "resume_text": "Data Scientist, 3 years. Built models in Python. No MLOps or deployment experience.",
            "relevance_score": 70
        },
        {
            "jd_text": "Machine Learning Engineer. Skills: Python, TensorFlow/PyTorch, MLOps, Distributed ML, AWS SageMaker, Data Pipelines. Experience: 4+ years.",
            "resume_text": "Software Engineer, 6 years. Proficient in Java. No ML experience.",
            "relevance_score": 20
        },

        # Example: Content Strategist
        {
            "jd_text": "Content Strategist. Skills: Content Strategy, SEO, Audience Research, Content Calendar, Copywriting, Analytics. Experience: 4+ years.",
            "resume_text": "Content Strategist, 5 years. Developed and executed comprehensive content strategies. Strong in SEO and audience research. Managed content calendars. Produced high-quality copywriting. Used analytics to measure performance.",
            "relevance_score": 93
        },
        {
            "jd_text": "Content Strategist. Skills: Content Strategy, SEO, Audience Research, Content Calendar, Copywriting, Analytics. Experience: 4+ years.",
            "resume_text": "Social Media Manager, 3 years. Created social media content. Some understanding of audience. No strategic content planning.",
            "relevance_score": 55
        },
        {
            "jd_text": "Content Strategist. Skills: Content Strategy, SEO, Audience Research, Content Calendar, Copywriting, Analytics. Experience: 4+ years.",
            "resume_text": "Public Relations Specialist, 7 years. Managed media relations. Wrote press releases. No digital content strategy.",
            "relevance_score": 30
        },

        # Example: Senior Financial Analyst
        {
            "jd_text": "Senior Financial Analyst. Skills: Financial Planning & Analysis (FP&A), Budgeting, Forecasting, Variance Analysis, SAP/Oracle ERP, Advanced Excel. Experience: 5+ years.",
            "resume_text": "Senior Financial Analyst, 6 years. Led FP&A cycles, including budgeting and forecasting. Performed detailed variance analysis. Expert in SAP ERP and advanced Excel for financial modeling.",
            "relevance_score": 94
        },
        {
            "jd_text": "Senior Financial Analyst. Skills: Financial Planning & Analysis (FP&A), Budgeting, Forecasting, Variance Analysis, SAP/Oracle ERP, Advanced Excel. Experience: 5+ years.",
            "resume_text": "Financial Analyst, 3 years. Assisted with budgeting. Proficient in basic Excel. No FP&A leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior Financial Analyst. Skills: Financial Planning & Analysis (FP&A), Budgeting, Forecasting, Variance Analysis, SAP/Oracle ERP, Advanced Excel. Experience: 5+ years.",
            "resume_text": "Auditor, 7 years. Conducted financial audits. Strong in compliance. No FP&A experience.",
            "relevance_score": 40
        },

        # Example: Talent Acquisition Specialist
        {
            "jd_text": "Talent Acquisition Specialist. Skills: Full-Cycle Recruiting, Sourcing (LinkedIn Recruiter), Applicant Tracking Systems (ATS), Interviewing, Employer Branding. Experience: 3+ years.",
            "resume_text": "Talent Acquisition Specialist, 4 years. Managed full-cycle recruiting for tech roles. Expert in sourcing via LinkedIn Recruiter. Proficient in Greenhouse ATS. Conducted behavioral interviews. Contributed to employer branding.",
            "relevance_score": 92
        },
        {
            "jd_text": "Talent Acquisition Specialist. Skills: Full-Cycle Recruiting, Sourcing (LinkedIn Recruiter), Applicant Tracking Systems (ATS), Interviewing, Employer Branding. Experience: 3+ years.",
            "resume_text": "HR Coordinator, 2 years. Assisted with onboarding and scheduling interviews. No direct recruiting experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "Talent Acquisition Specialist. Skills: Full-Cycle Recruiting, Sourcing (LinkedIn Recruiter), Applicant Tracking Systems (ATS), Interviewing, Employer Branding. Experience: 3+ years.",
            "resume_text": "Sales Manager, 8 years. Led a sales team. Strong negotiation skills. No HR or recruiting experience.",
            "relevance_score": 20
        },

        # Example: Game Developer (Unity)
        {
            "jd_text": "Game Developer (Unity). Skills: Unity 3D, C#, Game Design, Shaders, Physics, Mobile Game Development. Experience: 2+ years.",
            "resume_text": "Game Developer with 3 years experience. Developed mobile games in Unity 3D using C#. Strong in game design principles, shaders, and physics. Published 2 games on app stores.",
            "relevance_score": 95
        },
        {
            "jd_text": "Game Developer (Unity). Skills: Unity 3D, C#, Game Design, Shaders, Physics, Mobile Game Development. Experience: 2+ years.",
            "resume_text": "Web Developer, 4 years. Proficient in JavaScript. Some graphics experience. No game development.",
            "relevance_score": 30
        },
        {
            "jd_text": "Game Developer (Unity). Skills: Unity 3D, C#, Game Design, Shaders, Physics, Mobile Game Development. Experience: 2+ years.",
            "resume_text": "3D Artist, 5 years. Created 3D models and textures. Familiar with game assets. No programming.",
            "relevance_score": 40
        },

        # Example: E-commerce Manager
        {
            "jd_text": "E-commerce Manager. Skills: E-commerce Platform Management (Shopify/Magento), Digital Marketing, SEO, Conversion Rate Optimization (CRO), Analytics. Experience: 4+ years.",
            "resume_text": "E-commerce Manager, 5 years. Managed Shopify store operations. Drove digital marketing campaigns and SEO. Implemented CRO strategies resulting in 15% increase in sales. Expert in Google Analytics.",
            "relevance_score": 93
        },
        {
            "jd_text": "E-commerce Manager. Skills: E-commerce Platform Management (Shopify/Magento), Digital Marketing, SEO, Conversion Rate Optimization (CRO), Analytics. Experience: 4+ years.",
            "resume_text": "Retail Store Manager, 8 years. Managed daily store operations. Some sales experience. No e-commerce platform or digital marketing.",
            "relevance_score": 35
        },
        {
            "jd_text": "E-commerce Manager. Skills: E-commerce Platform Management (Shopify/Magento), Digital Marketing, SEO, Conversion Rate Optimization (CRO), Analytics. Experience: 4+ years.",
            "resume_text": "Digital Marketing Specialist, 3 years. Focused on social media and content. Some SEO. No e-commerce platform management.",
            "relevance_score": 60
        },

        # Example: Data Governance Analyst
        {
            "jd_text": "Data Governance Analyst. Skills: Data Governance Frameworks, Data Quality, Metadata Management, Data Stewardship, SQL, Compliance (GDPR/HIPAA). Experience: 3+ years.",
            "resume_text": "Data Governance Analyst, 4 years. Developed and implemented data governance frameworks. Ensured data quality. Managed metadata. Supported data stewardship. Proficient in SQL. Ensured GDPR compliance.",
            "relevance_score": 92
        },
        {
            "jd_text": "Data Governance Analyst. Skills: Data Governance Frameworks, Data Quality, Metadata Management, Data Stewardship, SQL, Compliance (GDPR/HIPAA). Experience: 3+ years.",
            "resume_text": "Data Analyst, 2 years. Focused on data analysis. Some SQL. No specific data governance experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "Data Governance Analyst. Skills: Data Governance Frameworks, Data Quality, Metadata Management, Data Stewardship, SQL, Compliance (GDPR/HIPAA). Experience: 3+ years.",
            "resume_text": "Compliance Officer, 7 years. Ensured regulatory compliance. No data specific governance.",
            "relevance_score": 40
        },

        # Example: Robotics Engineer
        {
            "jd_text": "Robotics Engineer. Skills: ROS, C++, Python, Robot Kinematics, Control Systems, Sensor Integration. Experience: 3+ years.",
            "resume_text": "Robotics Engineer, 4 years. Developed robot control software in C++ and Python using ROS. Expertise in robot kinematics and control systems. Integrated various sensors (Lidar, Camera).",
            "relevance_score": 96
        },
        {
            "jd_text": "Robotics Engineer. Skills: ROS, C++, Python, Robot Kinematics, Control Systems, Sensor Integration. Experience: 3+ years.",
            "resume_text": "Software Engineer, 5 years. Proficient in Python. No robotics or hardware experience.",
            "relevance_score": 30
        },
        {
            "jd_text": "Robotics Engineer. Skills: ROS, C++, Python, Robot Kinematics, Control Systems, Sensor Integration. Experience: 3+ years.",
            "resume_text": "Mechanical Engineer, 6 years. Designed mechanical systems. Some programming. No robotics specific skills.",
            "relevance_score": 45
        },

        # Example: Technical Account Manager
        {
            "jd_text": "Technical Account Manager. Skills: Client Relationship Management, Technical Support, Product Knowledge (SaaS), Troubleshooting, Communication. Experience: 4+ years.",
            "resume_text": "Technical Account Manager, 5 years. Managed key client relationships for SaaS products. Provided advanced technical support and troubleshooting. Deep product knowledge. Excellent communication.",
            "relevance_score": 92
        },
        {
            "jd_text": "Technical Account Manager. Skills: Client Relationship Management, Technical Support, Product Knowledge (SaaS), Troubleshooting, Communication. Experience: 4+ years.",
            "resume_text": "Customer Service Rep, 3 years. Handled customer inquiries. No technical product knowledge.",
            "relevance_score": 30
        },
        {
            "jd_text": "Technical Account Manager. Skills: Client Relationship Management, Technical Support, Product Knowledge (SaaS), Troubleshooting, Communication. Experience: 4+ years.",
            "resume_text": "Sales Engineer, 6 years. Presented technical solutions to clients. Some sales focus. No post-sales account management.",
            "relevance_score": 70
        },

        # Example: Biomedical Engineer
        {
            "jd_text": "Biomedical Engineer. Skills: Medical Device Design, Biomechanics, Signal Processing, MATLAB, FDA Regulations, Prototyping. Experience: 3+ years.",
            "resume_text": "Biomedical Engineer, 4 years. Designed medical devices. Applied biomechanics principles. Processed biological signals in MATLAB. Familiar with FDA regulations. Built prototypes.",
            "relevance_score": 94
        },
        {
            "jd_text": "Biomedical Engineer. Skills: Medical Device Design, Biomechanics, Signal Processing, MATLAB, FDA Regulations, Prototyping. Experience: 3+ years.",
            "resume_text": "Electrical Engineer, 5 years. Designed circuits. Some signal processing. No medical device or biology focus.",
            "relevance_score": 40
        },
        {
            "jd_text": "Biomedical Engineer. Skills: Medical Device Design, Biomechanics, Signal Processing, MATLAB, FDA Regulations, Prototyping. Experience: 3+ years.",
            "resume_text": "Clinical Research Coordinator, 6 years. Managed clinical trials. No engineering background.",
            "relevance_score": 20
        },

        # Example: Investment Analyst
        {
            "jd_text": "Investment Analyst. Skills: Financial Modeling, Valuation, Equity Research, Portfolio Analysis, Bloomberg Terminal, CFA. Experience: 2+ years.",
            "resume_text": "Investment Analyst, 3 years. Built detailed financial models and valuations. Conducted equity research. Performed portfolio analysis. Proficient with Bloomberg Terminal. CFA Level II candidate.",
            "relevance_score": 93
        },
        {
            "jd_text": "Investment Analyst. Skills: Financial Modeling, Valuation, Equity Research, Portfolio Analysis, Bloomberg Terminal, CFA. Experience: 2+ years.",
            "resume_text": "Financial Planner, 5 years. Advised individual clients on financial goals. No institutional investment analysis.",
            "relevance_score": 50
        },
        {
            "jd_text": "Investment Analyst. Skills: Financial Modeling, Valuation, Equity Research, Portfolio Analysis, Bloomberg Terminal, CFA. Experience: 2+ years.",
            "resume_text": "Accountant, 7 years. Prepared financial statements. No investment analysis.",
            "relevance_score": 25
        },

        # Example: Digital Marketing Analyst
        {
            "jd_text": "Digital Marketing Analyst. Skills: Google Analytics, SEO, SEM, Data Visualization, A/B Testing, Excel, SQL. Experience: 2+ years.",
            "resume_text": "Digital Marketing Analyst, 3 years. Analyzed website performance using Google Analytics. Optimized SEO/SEM campaigns. Created data visualizations. Conducted A/B tests. Proficient in Excel and basic SQL.",
            "relevance_score": 90
        },
        {
            "jd_text": "Digital Marketing Analyst. Skills: Google Analytics, SEO, SEM, Data Visualization, A/B Testing, Excel, SQL. Experience: 2+ years.",
            "resume_text": "Marketing Coordinator, 1 year. Managed social media. Some analytics exposure. No in-depth analysis.",
            "relevance_score": 55
        },
        {
            "jd_text": "Digital Marketing Analyst. Skills: Google Analytics, SEO, SEM, Data Visualization, A/B Testing, Excel, SQL. Experience: 2+ years.",
            "resume_text": "Data Analyst, 4 years. Strong in SQL and Python. No specific digital marketing domain knowledge.",
            "relevance_score": 65
        },

        # Example: Supply Chain Analyst
        {
            "jd_text": "Supply Chain Analyst. Skills: Supply Chain Analytics, Inventory Optimization, Logistics, Data Modeling, Excel, SQL, ERP Systems. Experience: 3+ years.",
            "resume_text": "Supply Chain Analyst, 4 years. Performed in-depth supply chain analytics. Optimized inventory levels. Analyzed logistics data. Built data models in Excel. Used SQL for data extraction from ERP systems.",
            "relevance_score": 93
        },
        {
            "jd_text": "Supply Chain Analyst. Skills: Supply Chain Analytics, Inventory Optimization, Logistics, Data Modeling, Excel, SQL, ERP Systems. Experience: 3+ years.",
            "resume_text": "Operations Coordinator, 2 years. Managed daily operations. Some inventory tracking. No analytical role.",
            "relevance_score": 40
        },
        {
            "jd_text": "Supply Chain Analyst. Skills: Supply Chain Analytics, Inventory Optimization, Logistics, Data Modeling, Excel, SQL, ERP Systems. Experience: 3+ years.",
            "resume_text": "Business Analyst, 5 years. Focused on process improvement. Some data analysis. No supply chain domain expertise.",
            "relevance_score": 50
        },

        # Example: Frontend Engineer (Senior)
        {
            "jd_text": "Senior Frontend Engineer. Skills: React, TypeScript, Redux, Webpack, Performance Optimization, Unit Testing, UI/UX Principles. Experience: 5+ years.",
            "resume_text": "Senior Frontend Engineer, 6 years. Developed complex SPAs with React and TypeScript. Managed state with Redux. Optimized build processes with Webpack. Focused on performance and unit testing. Strong UI/UX principles.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Frontend Engineer. Skills: React, TypeScript, Redux, Webpack, Performance Optimization, Unit Testing, UI/UX Principles. Experience: 5+ years.",
            "resume_text": "Mid-level Frontend Developer, 3 years. Proficient in React. No senior-level experience or performance optimization focus.",
            "relevance_score": 65
        },
        {
            "jd_text": "Senior Frontend Engineer. Skills: React, TypeScript, Redux, Webpack, Performance Optimization, Unit Testing, UI/UX Principles. Experience: 5+ years.",
            "resume_text": "Backend Developer, 7 years. Strong in Python. No frontend experience.",
            "relevance_score": 20
        },

        # Example: Database Developer
        {
            "jd_text": "Database Developer. Skills: SQL, Stored Procedures, Database Design, Performance Tuning, ETL, Data Warehousing. Experience: 3+ years.",
            "resume_text": "Database Developer, 4 years. Wrote complex SQL queries and stored procedures. Designed relational databases. Performed performance tuning. Built ETL processes for data warehousing.",
            "relevance_score": 94
        },
        {
            "jd_text": "Database Developer. Skills: SQL, Stored Procedures, Database Design, Performance Tuning, ETL, Data Warehousing. Experience: 3+ years.",
            "resume_text": "Data Analyst, 2 years. Strong in SQL for querying. No database design or tuning experience.",
            "relevance_score": 60
        },
        {
            "jd_text": "Database Developer. Skills: SQL, Stored Procedures, Database Design, Performance Tuning, ETL, Data Warehousing. Experience: 3+ years.",
            "resume_text": "Application Developer, 5 years. Used ORMs to interact with databases. Limited direct SQL knowledge.",
            "relevance_score": 35
        },

        # Example: Education Program Manager
        {
            "jd_text": "Education Program Manager. Skills: Program Development, Curriculum Design, Stakeholder Engagement, Budget Management, K-12/Higher Ed. Experience: 5+ years.",
            "resume_text": "Education Program Manager, 6 years. Developed and managed educational programs. Designed curriculum for K-12. Engaged with diverse stakeholders. Managed program budgets.",
            "relevance_score": 92
        },
        {
            "jd_text": "Education Program Manager. Skills: Program Development, Curriculum Design, Stakeholder Engagement, Budget Management, K-12/Higher Ed. Experience: 5+ years.",
            "resume_text": "Teacher, 8 years. Taught in a classroom. No program management or curriculum design at a broader level.",
            "relevance_score": 50
        },
        {
            "jd_text": "Education Program Manager. Skills: Program Development, Curriculum Design, Stakeholder Engagement, Budget Management, K-12/Higher Ed. Experience: 5+ years.",
            "resume_text": "Project Coordinator, 3 years. Assisted with project planning. No education specific experience.",
            "relevance_score": 25
        },

        # Example: Environmental Engineer
        {
            "jd_text": "Environmental Engineer. Skills: Environmental Regulations (EPA), Site Assessment, Remediation, Water/Air Quality, CAD. Experience: 3+ years.",
            "resume_text": "Environmental Engineer, 4 years. Conducted site assessments and designed remediation plans. Expert in EPA regulations. Monitored water and air quality. Used CAD for designs.",
            "relevance_score": 94
        },
        {
            "jd_text": "Environmental Engineer. Skills: Environmental Regulations (EPA), Site Assessment, Remediation, Water/Air Quality, CAD. Experience: 3+ years.",
            "resume_text": "Civil Engineer, 5 years. Focused on infrastructure. Some CAD. No environmental specific expertise.",
            "relevance_score": 40
        },
        {
            "jd_text": "Environmental Engineer. Skills: Environmental Regulations (EPA), Site Assessment, Remediation, Water/Air Quality, CAD. Experience: 3+ years.",
            "resume_text": "Lab Technician, 2 years. Performed lab tests. No engineering or regulatory experience.",
            "relevance_score": 15
        },

        # Example: Business Intelligence Developer
        {
            "jd_text": "Business Intelligence Developer. Skills: Power BI, Tableau, SQL, Data Warehousing, ETL, SSIS/SSRS. Experience: 4+ years.",
            "resume_text": "BI Developer, 5 years. Built interactive dashboards in Power BI and Tableau. Designed and managed data warehouses. Developed ETL processes using SSIS. Strong SQL skills.",
            "relevance_score": 95
        },
        {
            "jd_text": "Business Intelligence Developer. Skills: Power BI, Tableau, SQL, Data Warehousing, ETL, SSIS/SSRS. Experience: 4+ years.",
            "resume_text": "Data Analyst, 3 years. Used Tableau for visualization. Some SQL. No data warehousing or ETL development.",
            "relevance_score": 60
        },
        {
            "jd_text": "Business Intelligence Developer. Skills: Power BI, Tableau, SQL, Data Warehousing, ETL, SSIS/SSRS. Experience: 4+ years.",
            "resume_text": "Software Developer, 6 years. Built web applications. No BI or data warehousing experience.",
            "relevance_score": 25
        },

        # Example: Pharmacist
        {
            "jd_text": "Pharmacist. Skills: Medication Dispensing, Patient Counseling, Drug Interactions, Pharmacy Operations, EHR. Experience: 2+ years.",
            "resume_text": "Pharmacist, 3 years. Dispensed medications accurately. Provided patient counseling on drug usage and interactions. Managed pharmacy operations. Proficient in EHR systems.",
            "relevance_score": 93
        },
        {
            "jd_text": "Pharmacist. Skills: Medication Dispensing, Patient Counseling, Drug Interactions, Pharmacy Operations, EHR. Experience: 2+ years.",
            "resume_text": "Pharmacy Technician, 5 years. Assisted pharmacists with dispensing. No counseling or full pharmacist duties.",
            "relevance_score": 45
        },
        {
            "jd_text": "Pharmacist. Skills: Medication Dispensing, Patient Counseling, Drug Interactions, Pharmacy Operations, EHR. Experience: 2+ years.",
            "resume_text": "Nurse, 7 years. Administered medications. No pharmacy specific knowledge.",
            "relevance_score": 30
        },

        # Example: Social Media Manager
        {
            "jd_text": "Social Media Manager. Skills: Social Media Strategy, Content Creation, Community Management, Analytics, Paid Social Campaigns. Experience: 3+ years.",
            "resume_text": "Social Media Manager, 4 years. Developed and executed social media strategies. Created engaging content. Managed online communities. Analyzed performance using analytics tools. Ran successful paid social campaigns.",
            "relevance_score": 94
        },
        {
            "jd_text": "Social Media Manager. Skills: Social Media Strategy, Content Creation, Community Management, Analytics, Paid Social Campaigns. Experience: 3+ years.",
            "resume_text": "Marketing Assistant, 2 years. Assisted with social media posts. No strategic or paid campaign experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "Social Media Manager. Skills: Social Media Strategy, Content Creation, Community Management, Analytics, Paid Social Campaigns. Experience: 3+ years.",
            "resume_text": "Public Relations Specialist, 6 years. Managed media relations. No social media focus.",
            "relevance_score": 25
        },

        # Example: Research Engineer
        {
            "jd_text": "Research Engineer. Skills: R&D, Prototyping, Data Analysis (Python/MATLAB), Experimental Design, Technical Writing. Experience: 4+ years.",
            "resume_text": "Research Engineer, 5 years. Led R&D projects. Developed prototypes. Analyzed experimental data using Python and MATLAB. Designed complex experiments. Wrote technical reports and papers.",
            "relevance_score": 96
        },
        {
            "jd_text": "Research Engineer. Skills: R&D, Prototyping, Data Analysis (Python/MATLAB), Experimental Design, Technical Writing. Experience: 4+ years.",
            "resume_text": "Software Engineer, 3 years. Proficient in Python. No research or experimental design.",
            "relevance_score": 40
        },
        {
            "jd_text": "Research Engineer. Skills: R&D, Prototyping, Data Analysis (Python/MATLAB), Experimental Design, Technical Writing. Experience: 4+ years.",
            "resume_text": "Lab Technician, 7 years. Performed routine lab procedures. No research or engineering role.",
            "relevance_score": 20
        },

        # Example: Solutions Architect
        {
            "jd_text": "Solutions Architect. Skills: Enterprise Architecture, Cloud Solutions (AWS/Azure), Microservices, API Design, Stakeholder Management, Technical Leadership. Experience: 7+ years.",
            "resume_text": "Solutions Architect, 8 years. Designed enterprise-level cloud solutions on AWS and Azure. Architected microservices and robust APIs. Managed complex stakeholder relationships. Provided technical leadership to development teams.",
            "relevance_score": 97
        },
        {
            "jd_text": "Solutions Architect. Skills: Enterprise Architecture, Cloud Solutions (AWS/Azure), Microservices, API Design, Stakeholder Management, Technical Leadership. Experience: 7+ years.",
            "resume_text": "Senior Software Engineer, 5 years. Built microservices. Some AWS experience. No architecture design or leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Solutions Architect. Skills: Enterprise Architecture, Cloud Solutions (AWS/Azure), Microservices, API Design, Stakeholder Management, Technical Leadership. Experience: 7+ years.",
            "resume_text": "Project Manager, 10 years. Managed IT projects. No technical architecture expertise.",
            "relevance_score": 30
        },

        # Example: Technical Trainer
        {
            "jd_text": "Technical Trainer. Skills: Training Delivery, Curriculum Development, Technical Concepts (Software/IT), Adult Learning Principles, Presentation Skills. Experience: 3+ years.",
            "resume_text": "Technical Trainer, 4 years. Delivered engaging technical training sessions on software products. Developed comprehensive curriculum based on adult learning principles. Excellent presentation skills.",
            "relevance_score": 92
        },
        {
            "jd_text": "Technical Trainer. Skills: Training Delivery, Curriculum Development, Technical Concepts (Software/IT), Adult Learning Principles, Presentation Skills. Experience: 3+ years.",
            "resume_text": "Software Developer, 5 years. Strong technical skills. No training or teaching experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "Technical Trainer. Skills: Training Delivery, Curriculum Development, Technical Concepts (Software/IT), Adult Learning Principles, Presentation Skills. Experience: 3+ years.",
            "resume_text": "Customer Service Manager, 7 years. Trained customer service reps. No technical domain knowledge.",
            "relevance_score": 35
        },

        # Example: Data Privacy Officer
        {
            "jd_text": "Data Privacy Officer. Skills: GDPR, CCPA, Data Protection, Privacy Impact Assessments, Compliance, Legal Frameworks. Experience: 5+ years.",
            "resume_text": "Data Privacy Officer, 6 years. Ensured compliance with GDPR and CCPA. Conducted privacy impact assessments. Developed data protection policies. Strong knowledge of legal frameworks.",
            "relevance_score": 95
        },
        {
            "jd_text": "Data Privacy Officer. Skills: GDPR, CCPA, Data Protection, Privacy Impact Assessments, Compliance, Legal Frameworks. Experience: 5+ years.",
            "resume_text": "Compliance Analyst, 3 years. Reviewed company policies. Some regulatory knowledge. No specific data privacy focus.",
            "relevance_score": 60
        },
        {
            "jd_text": "Data Privacy Officer. Skills: GDPR, CCPA, Data Protection, Privacy Impact Assessments, Compliance, Legal Frameworks. Experience: 5+ years.",
            "resume_text": "IT Security Analyst, 4 years. Focused on network security. No privacy law expertise.",
            "relevance_score": 30
        },

        # Example: Quantitative Analyst (Quant)
        {
            "jd_text": "Quantitative Analyst. Skills: Python/R, C++, Statistical Modeling, Time Series Analysis, Financial Markets, Risk Management. Experience: 3+ years.",
            "resume_text": "Quantitative Analyst, 4 years. Developed statistical models in Python and C++. Performed time series analysis for financial markets. Built risk management models.",
            "relevance_score": 96
        },
        {
            "jd_text": "Quantitative Analyst. Skills: Python/R, C++, Statistical Modeling, Time Series Analysis, Financial Markets, Risk Management. Experience: 3+ years.",
            "resume_text": "Data Scientist, 3 years. Strong in Python for ML. No specific financial markets or C++ experience.",
            "relevance_score": 70
        },
        {
            "jd_text": "Quantitative Analyst. Skills: Python/R, C++, Statistical Modeling, Time Series Analysis, Financial Markets, Risk Management. Experience: 3+ years.",
            "resume_text": "Financial Analyst, 5 years. Focused on corporate finance. No quantitative modeling.",
            "relevance_score": 25
        },

        # Example: Product Marketing Manager
        {
            "jd_text": "Product Marketing Manager. Skills: Go-to-Market Strategy, Product Launch, Market Positioning, Messaging, Sales Enablement, Competitive Analysis. Experience: 4+ years.",
            "resume_text": "Product Marketing Manager, 5 years. Developed and executed go-to-market strategies. Led successful product launches. Defined market positioning and messaging. Created sales enablement materials. Conducted competitive analysis.",
            "relevance_score": 94
        },
        {
            "jd_text": "Product Marketing Manager. Skills: Go-to-Market Strategy, Product Launch, Market Positioning, Messaging, Sales Enablement, Competitive Analysis. Experience: 4+ years.",
            "resume_text": "Marketing Specialist, 3 years. Created content and managed social media. No product launch or strategic positioning.",
            "relevance_score": 55
        },
        {
            "jd_text": "Product Marketing Manager. Skills: Go-to-Market Strategy, Product Launch, Market Positioning, Messaging, Sales Enablement, Competitive Analysis. Experience: 4+ years.",
            "resume_text": "Product Manager, 6 years. Defined product features. No marketing focus.",
            "relevance_score": 40
        },

        # Example: Site Reliability Engineer (SRE)
        {
            "jd_text": "Site Reliability Engineer (SRE). Skills: Reliability Engineering, Distributed Systems, Cloud (AWS/GCP), Kubernetes, Prometheus/Grafana, Incident Management. Experience: 5+ years.",
            "resume_text": "SRE with 6 years experience. Focused on reliability engineering for distributed systems on AWS. Expert in Kubernetes. Implemented monitoring with Prometheus/Grafana. Led incident management.",
            "relevance_score": 97
        },
        {
            "jd_text": "Site Reliability Engineer (SRE). Skills: Reliability Engineering, Distributed Systems, Cloud (AWS/GCP), Kubernetes, Prometheus/Grafana, Incident Management. Experience: 5+ years.",
            "resume_text": "DevOps Engineer, 4 years. Built CI/CD pipelines. Used Docker and some AWS. No deep reliability focus.",
            "relevance_score": 70
        },
        {
            "jd_text": "Site Reliability Engineer (SRE). Skills: Reliability Engineering, Distributed Systems, Cloud (AWS/GCP), Kubernetes, Prometheus/Grafana, Incident Management. Experience: 5+ years.",
            "resume_text": "System Administrator, 8 years. Managed Linux servers. No cloud or distributed systems experience.",
            "relevance_score": 30
        },

        # Example: Data Architect
        {
            "jd_text": "Data Architect. Skills: Data Modeling, Data Warehousing, Cloud Data Platforms (Snowflake/Databricks), ETL Architecture, Big Data Technologies, Data Governance. Experience: 7+ years.",
            "resume_text": "Data Architect, 8 years. Designed complex data models and data warehouses. Architected solutions on Snowflake and Databricks. Led ETL architecture. Expertise in big data technologies and data governance.",
            "relevance_score": 98
        },
        {
            "jd_text": "Data Architect. Skills: Data Modeling, Data Warehousing, Cloud Data Platforms (Snowflake/Databricks), ETL Architecture, Big Data Technologies, Data Governance. Experience: 7+ years.",
            "resume_text": "Data Engineer, 5 years. Built ETL pipelines. Some data modeling. No architecture leadership.",
            "relevance_score": 75
        },
        {
            "jd_text": "Data Architect. Skills: Data Modeling, Data Warehousing, Cloud Data Platforms (Snowflake/Databricks), ETL Architecture, Big Data Technologies, Data Governance. Experience: 7+ years.",
            "resume_text": "Business Intelligence Developer, 6 years. Built dashboards. Strong SQL. No data architecture.",
            "relevance_score": 40
        },

        # Example: Technical Sales Engineer
        {
            "jd_text": "Technical Sales Engineer. Skills: Technical Sales, Product Demonstrations, Solution Selling, Client Presentations, CRM (Salesforce), Networking. Experience: 3+ years.",
            "resume_text": "Technical Sales Engineer, 4 years. Drove technical sales cycles. Conducted compelling product demonstrations. Applied solution selling methodologies. Delivered client presentations. Proficient in Salesforce.",
            "relevance_score": 93
        },
        {
            "jd_text": "Technical Sales Engineer. Skills: Technical Sales, Product Demonstrations, Solution Selling, Client Presentations, CRM (Salesforce), Networking. Experience: 3+ years.",
            "resume_text": "Sales Representative, 5 years. Strong in sales. No technical product knowledge or demonstrations.",
            "relevance_score": 50
        },
        {
            "jd_text": "Technical Sales Engineer. Skills: Technical Sales, Product Demonstrations, Solution Selling, Client Presentations, CRM (Salesforce), Networking. Experience: 3+ years.",
            "resume_text": "Software Engineer, 7 years. Strong technical skills. No sales experience.",
            "relevance_score": 20
        },

        # Example: Research Scientist (AI/ML)
        {
            "jd_text": "Research Scientist (AI/ML). Skills: Deep Learning, Reinforcement Learning, Computer Vision, NLP, Python, PyTorch/TensorFlow, Publications. Experience: PhD + 3 years research.",
            "resume_text": "AI Research Scientist, PhD + 4 years research. Expertise in Deep Learning and Reinforcement Learning. Published extensively in Computer Vision and NLP. Implemented models in PyTorch. Strong Python.",
            "relevance_score": 98
        },
        {
            "jd_text": "Research Scientist (AI/ML). Skills: Deep Learning, Reinforcement Learning, Computer Vision, NLP, Python, PyTorch/TensorFlow, Publications. Experience: PhD + 3 years research.",
            "resume_text": "Machine Learning Engineer, 5 years. Deployed ML models. Some deep learning. No research focus or publications.",
            "relevance_score": 70
        },
        {
            "jd_text": "Research Scientist (AI/ML). Skills: Deep Learning, Reinforcement Learning, Computer Vision, NLP, Python, PyTorch/TensorFlow, Publications. Experience: PhD + 3 years research.",
            "resume_text": "Data Scientist, 6 years. Focused on predictive modeling. No deep learning research.",
            "relevance_score": 40
        },

        # Example: Chief Financial Officer (CFO)
        {
            "jd_text": "Chief Financial Officer (CFO). Skills: Financial Strategy, Corporate Finance, M&A, Investor Relations, GAAP, Team Leadership, Board Reporting. Experience: 15+ years.",
            "resume_text": "CFO with 20 years experience. Developed and executed financial strategies. Led M&A activities. Managed investor relations. Ensured GAAP compliance. Led large finance teams and reported to board.",
            "relevance_score": 98
        },
        {
            "jd_text": "Chief Financial Officer (CFO). Skills: Financial Strategy, Corporate Finance, M&A, Investor Relations, GAAP, Team Leadership, Board Reporting. Experience: 15+ years.",
            "resume_text": "Financial Controller, 10 years. Managed financial reporting. Some budgeting. No strategic finance or M&A leadership.",
            "relevance_score": 60
        },
        {
            "jd_text": "Chief Financial Officer (CFO). Skills: Financial Strategy, Corporate Finance, M&A, Investor Relations, GAAP, Team Leadership, Board Reporting. Experience: 15+ years.",
            "resume_text": "Senior Financial Analyst, 7 years. Built financial models. No executive leadership.",
            "relevance_score": 30
        },

        # Example: Director of Engineering
        {
            "jd_text": "Director of Engineering. Skills: Engineering Leadership, Software Architecture, Scalability, Team Management, Budgeting, Agile Methodologies. Experience: 10+ years.",
            "resume_text": "Director of Engineering, 12 years. Led multiple engineering teams. Defined software architecture for scalable systems. Managed departmental budgets. Championed Agile methodologies.",
            "relevance_score": 97
        },
        {
            "jd_text": "Director of Engineering. Skills: Engineering Leadership, Software Architecture, Scalability, Team Management, Budgeting, Agile Methodologies. Experience: 10+ years.",
            "resume_text": "Senior Software Engineer, 8 years. Strong in architecture. Some team lead experience. No director-level management.",
            "relevance_score": 70
        },
        {
            "jd_text": "Director of Engineering. Skills: Engineering Leadership, Software Architecture, Scalability, Team Management, Budgeting, Agile Methodologies. Experience: 10+ years.",
            "resume_text": "Project Manager, 15 years. Managed IT projects. No direct engineering leadership.",
            "relevance_score": 40
        },

        # Example: Head of Marketing
        {
            "jd_text": "Head of Marketing. Skills: Marketing Strategy, Brand Management, Digital Marketing, Team Leadership, P&L Management, Market Analysis. Experience: 10+ years.",
            "resume_text": "Head of Marketing, 12 years. Developed and executed global marketing strategies. Built strong brands. Led digital marketing initiatives. Managed large teams and P&L. Strong market analysis.",
            "relevance_score": 96
        },
        {
            "jd_text": "Head of Marketing. Skills: Marketing Strategy, Brand Management, Digital Marketing, Team Leadership, P&L Management, Market Analysis. Experience: 10+ years.",
            "resume_text": "Marketing Manager, 7 years. Managed campaigns. Some team leadership. No head-of-department experience or P&L.",
            "relevance_score": 65
        },
        {
            "jd_text": "Head of Marketing. Skills: Marketing Strategy, Brand Management, Digital Marketing, Team Leadership, P&L Management, Market Analysis. Experience: 10+ years.",
            "resume_text": "Sales Director, 15 years. Led sales teams. No marketing strategy or brand management.",
            "relevance_score": 35
        },

        # Example: Principal Software Engineer
        {
            "jd_text": "Principal Software Engineer. Skills: System Design, Distributed Systems, Scalability, Mentorship, Code Quality, Performance Optimization. Experience: 8+ years.",
            "resume_text": "Principal Software Engineer, 10 years. Led system design for highly scalable distributed systems. Mentored multiple engineers. Championed code quality and performance optimization initiatives.",
            "relevance_score": 98
        },
        {
            "jd_text": "Principal Software Engineer. Skills: System Design, Distributed Systems, Scalability, Mentorship, Code Quality, Performance Optimization. Experience: 8+ years.",
            "resume_text": "Senior Software Engineer, 6 years. Built complex features. Some design experience. No principal-level leadership or system-wide impact.",
            "relevance_score": 75
        },
        {
            "jd_text": "Principal Software Engineer. Skills: System Design, Distributed Systems, Scalability, Mentorship, Code Quality, Performance Optimization. Experience: 8+ years.",
            "resume_text": "DevOps Engineer, 7 years. Focused on CI/CD. Some scripting. No core software engineering design.",
            "relevance_score": 45
        },

        # Example: Senior HR Business Partner
        {
            "jd_text": "Senior HR Business Partner. Skills: Strategic HR, Organizational Development, Change Management, Leadership Coaching, Workforce Planning, HR Analytics. Experience: 8+ years.",
            "resume_text": "Senior HRBP, 9 years. Partnered with executive leadership on strategic HR initiatives. Drove organizational development and change management. Provided leadership coaching. Led workforce planning and utilized HR analytics.",
            "relevance_score": 95
        },
        {
            "jd_text": "Senior HR Business Partner. Skills: Strategic HR, Organizational Development, Change Management, Leadership Coaching, Workforce Planning, HR Analytics. Experience: 8+ years.",
            "resume_text": "HR Generalist, 5 years. Managed employee relations. Some HRIS experience. No strategic or OD focus.",
            "relevance_score": 60
        },
        {
            "jd_text": "Senior HR Business Partner. Skills: Strategic HR, Organizational Development, Change Management, Leadership Coaching, Workforce Planning, HR Analytics. Experience: 8+ years.",
            "resume_text": "Recruitment Manager, 10 years. Led recruiting teams. No broad HRBP experience.",
            "relevance_score": 30
        },

        # Example: Senior UX Designer
        {
            "jd_text": "Senior UX Designer. Skills: User-Centered Design, Wireframing, Prototyping (Figma/Sketch), Usability Testing, Design Systems, Interaction Design, User Flows. Experience: 5+ years.",
            "resume_text": "Senior UX Designer, 6 years. Led end-to-end user-centered design processes. Created complex wireframes and prototypes in Figma. Conducted extensive usability testing. Contributed to design systems and defined user flows. Expertise in interaction design.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior UX Designer. Skills: User-Centered Design, Wireframing, Prototyping (Figma/Sketch), Usability Testing, Design Systems, Interaction Design, User Flows. Experience: 5+ years.",
            "resume_text": "Mid-level UI Designer, 3 years. Focused on visual design. Some prototyping. No senior UX focus.",
            "relevance_score": 65
        },
        # --- Additional Data for various roles to increase dataset size and diversity ---
        # Example: Data Analyst (Intermediate)
        {
            "jd_text": "Data Analyst. Skills: SQL, Python (Pandas), Tableau, A/B Testing, Statistical Analysis. Experience: 2-4 years.",
            "resume_text": "Data Analyst with 3 years experience. Proficient in SQL and Python for data manipulation. Conducted A/B tests and statistical analysis. Created dashboards in Tableau.",
            "relevance_score": 88
        },
        {
            "jd_text": "Data Analyst. Skills: SQL, Python (Pandas), Tableau, A/B Testing, Statistical Analysis. Experience: 2-4 years.",
            "resume_text": "Business Intelligence Analyst, 5 years. Strong in Power BI and Excel. Some SQL. No Python or A/B testing experience.",
            "relevance_score": 65
        },

        # Example: Software Engineer (Senior)
        {
            "jd_text": "Senior Software Engineer. Skills: Java, Microservices, Spring Cloud, Kafka, Distributed Systems, AWS, Leadership. Experience: 7+ years.",
            "resume_text": "Principal Software Engineer, 9 years. Designed and led development of complex microservices using Java and Spring Cloud. Expertise in Kafka and distributed systems on AWS. Mentored junior engineers.",
            "relevance_score": 97
        },
        {
            "jd_text": "Senior Software Engineer. Skills: Java, Microservices, Spring Cloud, Kafka, Distributed Systems, AWS, Leadership. Experience: 7+ years.",
            "resume_text": "Mid-level Java Developer, 4 years. Built REST APIs. Familiar with basic AWS services. No experience with distributed systems or leadership.",
            "relevance_score": 50
        },

        # Example: Marketing Manager
        {
            "jd_text": "Marketing Manager. Skills: Digital Strategy, Campaign Management, Budgeting, Team Leadership, CRM, Analytics. Experience: 5+ years.",
            "resume_text": "Marketing Manager, 6 years. Developed and executed digital marketing strategies. Managed multi-channel campaigns and budgets. Led a team of 3. Proficient in Salesforce CRM and Google Analytics.",
            "relevance_score": 92
        },
        {
            "jd_text": "Marketing Manager. Skills: Digital Strategy, Campaign Management, Budgeting, Team Leadership, CRM, Analytics. Experience: 5+ years.",
            "resume_text": "Marketing Specialist, 3 years. Executed campaigns. No budgeting or team leadership experience.",
            "relevance_score": 60
        },

        # Example: Financial Accountant
        {
            "jd_text": "Financial Accountant. Skills: GAAP, Financial Statements, General Ledger, Reconciliation, Audit Preparation, ERP (SAP). Experience: 3+ years.",
            "resume_text": "Senior Accountant with 4 years. Prepared financial statements adhering to GAAP. Managed general ledger and reconciliations. Assisted with external audits. Proficient in SAP ERP.",
            "relevance_score": 90
        },
        {
            "jd_text": "Financial Accountant. Skills: GAAP, Financial Statements, General Ledger, Reconciliation, Audit Preparation, ERP (SAP). Experience: 3+ years.",
            "resume_text": "Bookkeeper, 7 years. Handled accounts payable/receivable. No experience with full financial statements or audits.",
            "relevance_score": 45
        },

        # Example: HR Manager
        {
            "jd_text": "HR Manager. Skills: Employee Engagement, Performance Management, Compensation & Benefits, HR Policy, Conflict Resolution. Experience: 5+ years.",
            "resume_text": "HR Manager, 6 years. Drove employee engagement initiatives. Implemented performance management systems. Managed compensation and benefits programs. Expert in conflict resolution.",
            "relevance_score": 94
        },
        {
            "jd_text": "HR Manager. Skills: Employee Engagement, Performance Management, Compensation & Benefits, HR Policy, Conflict Resolution. Experience: 5+ years.",
            "resume_text": "HR Coordinator, 3 years. Assisted with onboarding and record keeping. No managerial experience.",
            "relevance_score": 50
        },

        # Example: Cloud Engineer
        {
            "jd_text": "Cloud Engineer. Skills: AWS, Azure, Infrastructure as Code (Terraform), Scripting (Python/Bash), Monitoring (CloudWatch). Experience: 3+ years.",
            "resume_text": "Cloud Engineer with 4 years. Deployed and managed infrastructure on AWS using Terraform. Wrote Python and Bash scripts for automation. Configured CloudWatch for monitoring.",
            "relevance_score": 93
        },
        {
            "jd_text": "Cloud Engineer. Skills: AWS, Azure, Infrastructure as Code (Terraform), Scripting (Python/Bash), Monitoring (CloudWatch). Experience: 3+ years.",
            "resume_text": "Network Administrator, 5 years. Managed on-premise networks. Basic understanding of cloud concepts. No hands-on cloud deployment.",
            "relevance_score": 35
        },

        # Example: Cybersecurity Analyst
        {
            "jd_text": "Cybersecurity Analyst. Skills: SIEM, Threat Detection, Vulnerability Scanning, Security Operations, Incident Response. Experience: 2+ years.",
            "resume_text": "Cybersecurity Analyst, 3 years. Monitored SIEM for threat detection. Performed vulnerability scans. Participated in incident response. Worked in a SOC environment.",
            "relevance_score": 91
        },
        {
            "jd_text": "Cybersecurity Analyst. Skills: SIEM, Threat Detection, Vulnerability Scanning, Security Operations, Incident Response. Experience: 2+ years.",
            "resume_text": "IT Helpdesk Specialist, 4 years. Resolved user issues. Some security awareness training. No direct cybersecurity experience.",
            "relevance_score": 25
        },

        # Example: Senior Project Manager
        {
            "jd_text": "Senior Project Manager. Skills: PMP, Agile, Scrum, Budget Management, Stakeholder Engagement, Risk Management, Software Development. Experience: 8+ years.",
            "resume_text": "Certified PMP Senior Project Manager, 10 years. Managed large-scale software development projects using Agile/Scrum. Expert in budget and risk management. Strong stakeholder engagement.",
            "relevance_score": 96
        },
        {
            "jd_text": "Senior Project Manager. Skills: PMP, Agile, Scrum, Budget Management, Stakeholder Engagement, Risk Management, Software Development. Experience: 8+ years.",
            "resume_text": "Team Lead, 5 years. Led a small development team. Familiar with Agile. No formal project management certification or large project experience.",
            "relevance_score": 60
        },

        # Example: UX Writer
        {
            "jd_text": "UX Writer. Skills: UX Writing, Content Strategy, Microcopy, Style Guides, User-Centered Design, Figma. Experience: 3+ years.",
            "resume_text": "UX Writer with 4 years. Crafted clear and concise microcopy for digital products. Developed content strategies and style guides. Collaborated with UX designers in Figma.",
            "relevance_score": 90
        },
        {
            "jd_text": "UX Writer. Skills: UX Writing, Content Strategy, Microcopy, Style Guides, User-Centered Design, Figma. Experience: 3+ years.",
            "resume_text": "Copywriter, 5 years. Wrote marketing copy and advertisements. No specific UX writing experience.",
            "relevance_score": 30
        },

        # Example: Data Engineer (Entry Level)
        {
            "jd_text": "Data Engineer (Entry Level). Skills: Python, SQL, ETL, Cloud Basics (AWS/GCP), Data Modeling. Experience: 0-2 years.",
            "resume_text": "Recent Computer Science graduate. Strong in Python and SQL. Completed university projects involving ETL and basic data modeling. Familiar with AWS cloud concepts.",
            "relevance_score": 80
        },
        {
            "jd_text": "Data Engineer (Entry Level). Skills: Python, SQL, ETL, Cloud Basics (AWS/GCP), Data Modeling. Experience: 0-2 years.",
            "resume_text": "Data Entry Clerk, 3 years. Accurate data entry. No programming or data engineering skills.",
            "relevance_score": 10
        },

        # Example: SEO Specialist
        {
            "jd_text": "SEO Specialist. Skills: On-page SEO, Off-page SEO, Keyword Research, Google Analytics, SEMRush/Ahrefs. Experience: 2+ years.",
            "resume_text": "SEO Specialist, 3 years. Improved organic rankings through on-page and off-page SEO. Conducted extensive keyword research. Proficient in Google Analytics and SEMRush.",
            "relevance_score": 92
        },
        {
            "jd_text": "SEO Specialist. Skills: On-page SEO, Off-page SEO, Keyword Research, Google Analytics, SEMRush/Ahrefs. Experience: 2+ years.",
            "resume_text": "Social Media Manager, 4 years. Managed social media campaigns. Some content creation. No specific SEO expertise.",
            "relevance_score": 40
        },

        # Example: Business Systems Analyst
        {
            "jd_text": "Business Systems Analyst. Skills: System Analysis, Requirements Gathering, SQL, ERP Implementation, Stakeholder Communication. Experience: 4+ years.",
            "resume_text": "Business Systems Analyst, 5 years. Performed system analysis and gathered requirements for ERP implementation. Strong SQL skills. Excellent stakeholder communication. Worked on SAP projects.",
            "relevance_score": 93
        },
        {
            "jd_text": "Business Systems Analyst. Skills: System Analysis, Requirements Gathering, SQL, ERP Implementation, Stakeholder Communication. Experience: 4+ years.",
            "resume_text": "IT Support Analyst, 6 years. Provided technical support. Basic understanding of business processes. No system analysis or ERP implementation experience.",
            "relevance_score": 30
        },

        # Example: Embedded Software Engineer
        {
            "jd_text": "Embedded Software Engineer. Skills: C/C++, RTOS, Microcontrollers, Firmware Development, Debugging. Experience: 3+ years.",
            "resume_text": "Embedded Software Engineer, 4 years. Developed firmware in C/C++ for ARM microcontrollers. Experience with RTOS. Strong debugging skills. Worked on IoT devices.",
            "relevance_score": 95
        },
        {
            "jd_text": "Embedded Software Engineer. Skills: C/C++, RTOS, Microcontrollers, Firmware Development, Debugging. Experience: 3+ years.",
            "resume_text": "Web Developer, 5 years. Proficient in Python and JavaScript. No embedded systems experience.",
            "relevance_score": 20
        },

        # Example: Quality Control Inspector
        {
            "jd_text": "Quality Control Inspector. Skills: Quality Inspection, Measurement Tools (Calipers, Micrometers), Blueprint Reading, SPC, Documentation. Experience: 2+ years.",
            "resume_text": "QC Inspector, 3 years. Performed quality inspections using calipers and micrometers. Read complex blueprints. Applied SPC techniques. Maintained accurate documentation.",
            "relevance_score": 90
        },
        {
            "jd_text": "Quality Control Inspector. Skills: Quality Inspection, Measurement Tools (Calipers, Micrometers), Blueprint Reading, SPC, Documentation. Experience: 2+ years.",
            "resume_text": "Production Operator, 7 years. Operated machinery on a manufacturing line. No quality inspection or measurement experience.",
            "relevance_score": 30
        },

        # Example: Research Assistant (Psychology)
        {
            "jd_text": "Research Assistant. Skills: Data Collection, Statistical Software (SPSS/R), Literature Review, Report Writing, Experimental Design. Experience: 1+ years.",
            "resume_text": "Research Assistant, 2 years. Assisted with data collection for psychology experiments. Analyzed data using SPSS. Conducted literature reviews and wrote research reports.",
            "relevance_score": 88
        },
        {
            "jd_text": "Research Assistant. Skills: Data Collection, Statistical Software (SPSS/R), Literature Review, Report Writing, Experimental Design. Experience: 1+ years.",
            "resume_text": "Administrative Assistant, 4 years. Organized office files. No research experience.",
            "relevance_score": 15
        },

        # Example: Elementary School Teacher
        {
            "jd_text": "Elementary School Teacher. Skills: Lesson Planning, Classroom Management, Differentiated Instruction, Child Development, Parent Communication. Experience: 2+ years.",
            "resume_text": "Elementary School Teacher, 3 years. Created engaging lesson plans. Managed diverse classrooms effectively. Implemented differentiated instruction. Strong parent communication.",
            "relevance_score": 95
        },
        {
            "jd_text": "Elementary School Teacher. Skills: Lesson Planning, Classroom Management, Differentiated Instruction, Child Development, Parent Communication. Experience: 2+ years.",
            "resume_text": "High School Teacher, 5 years. Taught specific subjects to older students. No elementary school experience.",
            "relevance_score": 50
        },

        # Example: Financial Advisor
        {
            "jd_text": "Financial Advisor. Skills: Financial Planning, Investment Management, Client Relationship Management, Retirement Planning, Risk Assessment. Experience: 3+ years.",
            "resume_text": "Financial Advisor, 4 years. Developed comprehensive financial plans. Managed investment portfolios. Built strong client relationships. Advised on retirement planning and risk assessment.",
            "relevance_score": 94
        },
        {
            "jd_text": "Financial Advisor. Skills: Financial Planning, Investment Management, Client Relationship Management, Retirement Planning, Risk Assessment. Experience: 3+ years.",
            "resume_text": "Bank Teller, 7 years. Assisted customers with transactions. No financial advisory experience.",
            "relevance_score": 20
        },

        # Example: Social Worker
        {
            "jd_text": "Social Worker. Skills: Case Management, Crisis Intervention, Client Advocacy, Community Resources, Documentation. Experience: 2+ years.",
            "resume_text": "Social Worker, 3 years. Provided case management and crisis intervention. Advocated for clients. Connected clients to community resources. Maintained accurate documentation.",
            "relevance_score": 92
        },
        {
            "jd_text": "Social Worker. Skills: Case Management, Crisis Intervention, Client Advocacy, Community Resources, Documentation. Experience: 2+ years.",
            "resume_text": "Counselor, 5 years. Provided individual therapy. No case management or community resource navigation.",
            "relevance_score": 60
        },

        # Example: Technical Support Engineer
        {
            "jd_text": "Technical Support Engineer. Skills: Software Troubleshooting, Linux, Networking, Ticketing Systems, Customer Communication. Experience: 2+ years.",
            "resume_text": "Technical Support Engineer, 3 years. Troubleshot complex software issues on Linux systems. Basic networking knowledge. Managed tickets in Zendesk. Excellent customer communication.",
            "relevance_score": 90
        },
        {
            "jd_text": "Technical Support Engineer. Skills: Software Troubleshooting, Linux, Networking, Ticketing Systems, Customer Communication. Experience: 2+ years.",
            "resume_text": "Call Center Agent, 4 years. Handled customer inquiries. No technical troubleshooting.",
            "relevance_score": 25
        },
        # --- Software Engineer ---
        {
            "jd_text": "Software Engineer. Skills: Python, Java, Spring Boot, Microservices, REST APIs, Docker, Kubernetes, AWS, CI/CD. Experience: 3+ years.",
            "resume_text": "Experienced Backend Developer with 5 years in Python, Java, Spring Boot. Built scalable microservices and REST APIs. Familiar with Docker and AWS. Participated in CI/CD.",
            "relevance_score": 95
        },
        {
            "jd_text": "Software Engineer. Skills: Python, Java, Spring Boot, Microservices, REST APIs, Docker, Kubernetes, AWS, CI/CD. Experience: 3+ years.",
            "resume_text": "Frontend Developer with 2 years experience in React and JavaScript. Basic understanding of APIs. No backend or DevOps experience.",
            "relevance_score": 55
        },
        {
            "jd_text": "Software Engineer. Skills: Python, Java, Spring Boot, Microservices, REST APIs, Docker, Kubernetes, AWS, CI/CD. Experience: 3+ years.",
            "resume_text": "QA Tester with 7 years experience. Automated test cases with Selenium. Some Java knowledge. No development or cloud experience.",
            "relevance_score": 30
        },
        {
            "jd_text": "Software Engineer. Skills: Python, Java, Spring Boot, Microservices, REST APIs, Docker, Kubernetes, AWS, CI/CD. Experience: 3+ years.",
            "resume_text": "Recent Computer Science graduate. Strong in data structures and algorithms. Basic Python projects. Eager to learn.",
            "relevance_score": 40
        },
        {
            "jd_text": "Software Engineer. Skills: Python, Java, Spring Boot, Microservices, REST APIs, Docker, Kubernetes, AWS, CI/CD. Experience: 3+ years.",
            "resume_text": "Fullstack Developer with 4 years experience. Proficient in Node.js, React, and MongoDB. Used AWS S3. Familiar with Git.",
            "relevance_score": 75
        },

        # --- Data Scientist ---
        {
            "jd_text": "Data Scientist. Skills: Python, R, SQL, Machine Learning, Deep Learning, NLP, Tableau, Spark. Experience: 4+ years.",
            "resume_text": "Lead Data Scientist with 6 years experience. Expert in Python, R, SQL. Developed and deployed ML/DL models. Strong NLP skills. Used Tableau for visualization and Spark for big data.",
            "relevance_score": 98
        },
        {
            "jd_text": "Data Scientist. Skills: Python, R, SQL, Machine Learning, Deep Learning, NLP, Tableau, Spark. Experience: 4+ years.",
            "resume_text": "Data Analyst, 3 years experience. Proficient in SQL and Excel. Basic Python for data cleaning. Interested in ML but no practical experience.",
            "relevance_score": 60
        },
        {
            "jd_text": "Data Scientist. Skills: Python, R, SQL, Machine Learning, Deep Learning, NLP, Tableau, Spark. Experience: 4+ years.",
            "resume_text": "Financial Analyst with 8 years experience in financial modeling. Strong Excel. No programming or ML background.",
            "relevance_score": 15
        },
        {
            "jd_text": "Data Scientist. Skills: Python, R, SQL, Machine Learning, Deep Learning, NLP, Tableau, Spark. Experience: 4+ years.",
            "resume_text": "PhD in Statistics, 1 year post-doc. Strong theoretical background in statistical modeling. Limited industry experience with big data tools.",
            "relevance_score": 70
        },
        {
            "jd_text": "Data Scientist. Skills: Python, R, SQL, Machine Learning, Deep Learning, NLP, Tableau, Spark. Experience: 4+ years.",
            "resume_text": "Business Intelligence Developer, 5 years. Built dashboards with Power BI and QlikView. Strong SQL. Some Python for data extraction.",
            "relevance_score": 78
        },

        # --- Product Manager ---
        {
            "jd_text": "Product Manager. Skills: Product Strategy, Roadmap, Agile, User Stories, Market Research, Jira. Experience: 5+ years.",
            "resume_text": "Senior Product Manager, 7 years. Defined product vision, managed roadmaps, led Agile teams. Excellent at user stories and market analysis. Proficient in Jira.",
            "relevance_score": 92
        },
        {
            "jd_text": "Product Manager. Skills: Product Strategy, Roadmap, Agile, User Stories, Market Research, Jira. Experience: 5+ years.",
            "resume_text": "Project Manager, 6 years experience. Managed projects using Waterfall. Good communication. No specific product ownership experience.",
            "relevance_score": 45
        },
        {
            "jd_text": "Product Manager. Skills: Product Strategy, Roadmap, Agile, User Stories, Market Research, Jira. Experience: 5+ years.",
            "resume_text": "UX Designer with 4 years experience. Focused on user research and prototyping. Collaborated with product teams. Familiar with Agile.",
            "relevance_score": 65
        },

        # --- Marketing Specialist ---
        {
            "jd_text": "Marketing Specialist. Skills: Digital Marketing, SEO, SEM, Social Media, Content Creation, Google Analytics. Experience: 2+ years.",
            "resume_text": "Digital Marketing professional, 3 years experience. Managed SEO/SEM campaigns, created social media content. Strong in Google Analytics. HubSpot CRM experience.",
            "relevance_score": 90
        },
        {
            "jd_text": "Marketing Specialist. Skills: Digital Marketing, SEO, SEM, Social Media, Content Creation, Google Analytics. Experience: 2+ years.",
            "resume_text": "Sales Assistant with 1 year experience. Good at customer communication. No digital marketing skills.",
            "relevance_score": 20
        },
        {
            "jd_text": "Marketing Specialist. Skills: Digital Marketing, SEO, SEM, Social Media, Content Creation, Google Analytics. Experience: 2+ years.",
            "resume_text": "Recent marketing graduate. Strong theoretical knowledge of digital marketing. Internship in content creation. Eager to apply skills.",
            "relevance_score": 60
        },

        # --- Financial Analyst ---
        {
            "jd_text": "Financial Analyst. Skills: Financial Modeling, Valuation, Excel, SQL, Financial Reporting. Experience: 3+ years.",
            "resume_text": "Experienced Financial Analyst with 4 years. Built complex financial models, performed valuations. Expert in Excel. Prepared quarterly financial reports.",
            "relevance_score": 88
        },
        {
            "jd_text": "Financial Analyst. Skills: Financial Modeling, Valuation, Excel, SQL, Financial Reporting. Experience: 3+ years.",
            "resume_text": "Accountant with 5 years experience. Handled payroll and bookkeeping. Basic Excel. No financial modeling experience.",
            "relevance_score": 40
        },
        {
            "jd_text": "Financial Analyst. Skills: Financial Modeling, Valuation, Excel, SQL, Financial Reporting. Experience: 3+ years.",
            "resume_text": "Investment Banking Associate, 2 years experience. Strong in corporate finance and M&A. Used advanced Excel. Some valuation experience.",
            "relevance_score": 75
        },

        # --- Human Resources Generalist ---
        {
            "jd_text": "Human Resources Generalist. Skills: Employee Relations, Talent Acquisition, HRIS, Benefits Administration, Compliance. Experience: 3+ years.",
            "resume_text": "HR Generalist with 4 years experience. Managed employee relations, supported recruitment, administered benefits. Proficient in Workday HRIS.",
            "relevance_score": 90
        },
        {
            "jd_text": "Human Resources Generalist. Skills: Employee Relations, Talent Acquisition, HRIS, Benefits Administration, Compliance. Experience: 3+ years.",
            "resume_text": "Office Manager, 10 years experience. Handled administrative tasks and some basic payroll. No dedicated HR experience.",
            "relevance_score": 25
        },
        {
            "jd_text": "Human Resources Generalist. Skills: Employee Relations, Talent Acquisition, HRIS, Benefits Administration, Compliance. Experience: 3+ years.",
            "resume_text": "Recruitment Coordinator, 2 years experience. Sourced candidates and managed applicant tracking system. Good communication.",
            "relevance_score": 65
        },

        # --- UX/UI Designer ---
        {
            "jd_text": "UX/UI Designer. Skills: User Research, Wireframing, Prototyping, Figma, Adobe XD, Usability Testing. Experience: 2+ years.",
            "resume_text": "UX/UI Designer with 3 years experience. Conducted user research, created wireframes and prototypes in Figma. Strong in usability testing. Built design systems.",
            "relevance_score": 93
        },
        {
            "jd_text": "UX/UI Designer. Skills: User Research, Wireframing, Prototyping, Figma, Adobe XD, Usability Testing. Experience: 2+ years.",
            "resume_text": "Graphic Designer, 5 years experience. Expert in Photoshop and Illustrator. No specific UX/UI process or research experience.",
            "relevance_score": 35
        },
        {
            "jd_text": "UX/UI Designer. Skills: User Research, Wireframing, Prototyping, Figma, Adobe XD, Usability Testing. Experience: 2+ years.",
            "resume_text": "Web Developer with 2 years experience in HTML, CSS, JavaScript. Designed responsive layouts. Some UI awareness.",
            "relevance_score": 50
        },

        # --- Cloud Architect ---
        {
            "jd_text": "Cloud Architect. Skills: AWS, Azure, GCP, Cloud Architecture, Terraform, Kubernetes, Network Security. Experience: 7+ years.",
            "resume_text": "Principal Cloud Architect with 10 years experience. Designed complex cloud solutions on AWS and Azure. Expert in Terraform and Kubernetes. Strong network security background.",
            "relevance_score": 97
        },
        {
            "jd_text": "Cloud Architect. Skills: AWS, Azure, GCP, Cloud Architecture, Terraform, Kubernetes, Network Security. Experience: 7+ years.",
            "resume_text": "System Administrator, 5 years experience. Managed on-premise servers. Basic understanding of AWS. No architecture or IaC experience.",
            "relevance_score": 40
        },
        {
            "jd_text": "Cloud Architect. Skills: AWS, Azure, GCP, Cloud Architecture, Terraform, Kubernetes, Network Security. Experience: 7+ years.",
            "resume_text": "DevOps Engineer, 4 years experience. Built CI/CD pipelines. Used Docker and some AWS services. Interested in architecture.",
            "relevance_score": 70
        },

        # --- Cybersecurity Engineer ---
        {
            "jd_text": "Cybersecurity Engineer. Skills: Incident Response, Vulnerability Management, SIEM, Firewalls, Python Scripting. Experience: 4+ years.",
            "resume_text": "Cybersecurity Engineer, 5 years experience. Led incident response, managed vulnerabilities. Proficient with SIEM tools and firewalls. Wrote Python scripts for automation.",
            "relevance_score": 94
        },
        {
            "jd_text": "Cybersecurity Engineer. Skills: Incident Response, Vulnerability Management, SIEM, Firewalls, Python Scripting. Experience: 4+ years.",
            "resume_text": "Network Administrator, 6 years experience. Configured routers and switches. Basic security awareness. No incident response or advanced security tools experience.",
            "relevance_score": 30
        },
        {
            "jd_text": "Cybersecurity Engineer. Skills: Incident Response, Vulnerability Management, SIEM, Firewalls, Python Scripting. Experience: 4+ years.",
            "resume_text": "IT Auditor, 7 years experience. Conducted security audits and compliance checks. Familiar with NIST framework. No hands-on engineering.",
            "relevance_score": 60
        },

        # --- DevOps Specialist ---
        {
            "jd_text": "DevOps Specialist. Skills: CI/CD, Docker, Kubernetes, Jenkins, Terraform, Ansible, AWS, Python. Experience: 4+ years.",
            "resume_text": "DevOps Lead with 7 years experience. Implemented CI/CD pipelines with Jenkins, managed infrastructure with Terraform/Ansible on AWS. Expert in Docker and Kubernetes. Strong Python scripting.",
            "relevance_score": 96
        },
        {
            "jd_text": "DevOps Specialist. Skills: CI/CD, Docker, Kubernetes, Jenkins, Terraform, Ansible, AWS, Python. Experience: 4+ years.",
            "resume_text": "Software Engineer, 3 years. Used Git and some basic Docker. No CI/CD or infrastructure automation experience.",
            "relevance_score": 50
        },
        {
            "jd_text": "DevOps Specialist. Skills: CI/CD, Docker, Kubernetes, Jenkins, Terraform, Ansible, AWS, Python. Experience: 4+ years.",
            "resume_text": "Cloud Engineer, 2 years. Focused on AWS infrastructure. Some scripting experience. Interested in automation.",
            "relevance_score": 70
        },

        # --- Mobile App Developer (iOS) ---
        {
            "jd_text": "Mobile App Developer (iOS). Skills: Swift, SwiftUI, iOS SDK, Xcode, RESTful APIs, UI/UX. Experience: 3+ years.",
            "resume_text": "Senior iOS Developer, 5 years experience. Built multiple apps with Swift and SwiftUI. Strong in iOS SDK and Xcode. Integrated RESTful APIs. Focused on great UI/UX.",
            "relevance_score": 95
        },
        {
            "jd_text": "Mobile App Developer (iOS). Skills: Swift, SwiftUI, iOS SDK, Xcode, RESTful APIs, UI/UX. Experience: 3+ years.",
            "resume_text": "Android Developer, 4 years experience in Kotlin. No iOS experience.",
            "relevance_score": 30
        },
        {
            "jd_text": "Mobile App Developer (iOS). Skills: Swift, SwiftUI, iOS SDK, Xcode, RESTful APIs, UI/UX. Experience: 3+ years.",
            "resume_text": "Fullstack Web Developer, 6 years. Proficient in React Native for cross-platform. Some native iOS exposure.",
            "relevance_score": 70
        },

        # --- Business Development Manager ---
        {
            "jd_text": "Business Development Manager. Skills: Sales Strategy, Lead Generation, Negotiation, CRM (Salesforce), Client Relationships. Experience: 5+ years.",
            "resume_text": "Business Development Manager with 6 years experience. Consistently exceeded sales targets. Strong in lead generation and client relationship management. Proficient in Salesforce.",
            "relevance_score": 90
        },
        {
            "jd_text": "Business Development Manager. Skills: Sales Strategy, Lead Generation, Negotiation, CRM (Salesforce), Client Relationships. Experience: 5+ years.",
            "resume_text": "Marketing Coordinator, 2 years experience. Supported marketing campaigns. No direct sales or business development experience.",
            "relevance_score": 25
        },
        {
            "jd_text": "Business Development Manager. Skills: Sales Strategy, Lead Generation, Negotiation, CRM (Salesforce), Client Relationships. Experience: 5+ years.",
            "resume_text": "Sales Manager, 8 years experience. Managed a sales team. Strong negotiation skills. Used HubSpot CRM.",
            "relevance_score": 80
        },

        # --- Project Coordinator ---
        {
            "jd_text": "Project Coordinator. Skills: Project Planning, Scheduling, Communication, Microsoft Office, Jira. Experience: 2+ years.",
            "resume_text": "Project Coordinator with 3 years experience. Assisted in project planning and scheduling. Excellent communication. Used Microsoft Project and Jira.",
            "relevance_score": 85
        },
        {
            "jd_text": "Project Coordinator. Skills: Project Planning, Scheduling, Communication, Microsoft Office, Jira. Experience: 2+ years.",
            "resume_text": "Administrative Assistant, 5 years experience. Organized meetings and managed calendars. No formal project management experience.",
            "relevance_score": 30
        },
        {
            "jd_text": "Project Coordinator. Skills: Project Planning, Scheduling, Communication, Microsoft Office, Jira. Experience: 2+ years.",
            "resume_text": "Recent graduate with strong organizational skills. Internship in event planning. Familiar with basic office software.",
            "relevance_score": 55
        },

        # --- Technical Writer ---
        {
            "jd_text": "Technical Writer. Skills: Technical Documentation, API Documentation, Content Management Systems, Markdown, XML. Experience: 3+ years.",
            "resume_text": "Technical Writer, 4 years experience. Wrote user manuals and API documentation. Proficient in Confluence and Markdown. Experience with DITA XML.",
            "relevance_score": 92
        },
        {
            "jd_text": "Technical Writer. Skills: Technical Documentation, API Documentation, Content Management Systems, Markdown, XML. Experience: 3+ years.",
            "resume_text": "Content Writer, 5 years experience. Wrote blog posts and marketing copy. No technical documentation experience.",
            "relevance_score": 35
        },
        {
            "jd_text": "Technical Writer. Skills: Technical Documentation, API Documentation, Content Management Systems, Markdown, XML. Experience: 3+ years.",
            "resume_text": "Software Developer with 7 years experience. Wrote clean code and internal documentation. No formal technical writing role.",
            "relevance_score": 60
        },

        # --- Sales Representative ---
        {
            "jd_text": "Sales Representative. Skills: Sales, Lead Generation, Cold Calling, Negotiation, CRM. Experience: 1+ years.",
            "resume_text": "Sales Rep with 2 years experience. Consistently met sales targets. Strong in lead generation and cold calling. Used HubSpot CRM.",
            "relevance_score": 88
        },
        {
            "jd_text": "Sales Representative. Skills: Sales, Lead Generation, Cold Calling, Negotiation, CRM. Experience: 1+ years.",
            "resume_text": "Customer Service Rep, 3 years experience. Handled customer inquiries. No direct sales experience.",
            "relevance_score": 20
        },
        {
            "jd_text": "Sales Representative. Skills: Sales, Lead Generation, Cold Calling, Negotiation, CRM. Experience: 1+ years.",
            "resume_text": "Retail Sales Associate, 4 years experience. Assisted customers on sales floor. Good communication. No B2B sales.",
            "relevance_score": 50
        },

        # --- Customer Support Specialist ---
        {
            "jd_text": "Customer Support Specialist. Skills: Customer Service, Problem Solving, Troubleshooting, CRM, Communication. Experience: 2+ years.",
            "resume_text": "Customer Support Specialist, 3 years experience. Resolved complex customer issues via phone/email. Proficient in Zendesk CRM. Excellent communication.",
            "relevance_score": 90
        },
        {
            "jd_text": "Customer Support Specialist. Skills: Customer Service, Problem Solving, Troubleshooting, CRM, Communication. Experience: 2+ years.",
            "resume_text": "Retail Associate, 4 years experience. Assisted customers in store. No technical troubleshooting or CRM experience.",
            "relevance_score": 30
        },
        {
            "jd_text": "Customer Support Specialist. Skills: Customer Service, Problem Solving, Troubleshooting, CRM, Communication. Experience: 2+ years.",
            "resume_text": "Technical Support Engineer, 5 years. Provided IT support. Strong troubleshooting. Some customer interaction.",
            "relevance_score": 75
        },

        # --- Operations Coordinator ---
        {
            "jd_text": "Operations Coordinator. Skills: Process Improvement, Logistics, Inventory Management, Data Entry, Excel. Experience: 2+ years.",
            "resume_text": "Operations Coordinator with 3 years experience. Streamlined logistics processes. Managed inventory. Proficient in Excel and ERP systems.",
            "relevance_score": 85
        },
        {
            "jd_text": "Operations Coordinator. Skills: Process Improvement, Logistics, Inventory Management, Data Entry, Excel. Experience: 2+ years.",
            "resume_text": "Administrative Assistant, 5 years. Handled scheduling and office supplies. No operations or logistics experience.",
            "relevance_score": 25
        },
        {
            "jd_text": "Operations Coordinator. Skills: Process Improvement, Logistics, Inventory Management, Data Entry, Excel. Experience: 2+ years.",
            "resume_text": "Supply Chain Analyst, 1 year experience. Focused on data analysis for supply chain. Some process mapping.",
            "relevance_score": 65
        },

        # --- Supply Chain Manager ---
        {
            "jd_text": "Supply Chain Manager. Skills: Supply Chain Optimization, Logistics, Inventory Planning, Demand Forecasting, SAP. Experience: 5+ years.",
            "resume_text": "Supply Chain Manager, 6 years experience. Optimized end-to-end supply chain. Expert in logistics and inventory planning. Used SAP for demand forecasting.",
            "relevance_score": 94
        },
        {
            "jd_text": "Supply Chain Manager. Skills: Supply Chain Optimization, Logistics, Inventory Planning, Demand Forecasting, SAP. Experience: 5+ years.",
            "resume_text": "Warehouse Supervisor, 8 years experience. Managed warehouse operations. No strategic supply chain planning experience.",
            "relevance_score": 40
        },
        {
            "jd_text": "Supply Chain Manager. Skills: Supply Chain Optimization, Logistics, Inventory Planning, Demand Forecasting, SAP. Experience: 5+ years.",
            "resume_text": "Procurement Specialist, 4 years experience. Managed vendor relationships and purchasing. Familiar with ERP systems.",
            "relevance_score": 70
        },

        # --- Mechanical Design Engineer ---
        {
            "jd_text": "Mechanical Design Engineer. Skills: CAD (SolidWorks), FEA, Product Design, Thermodynamics, GD&T. Experience: 3+ years.",
            "resume_text": "Mechanical Design Engineer, 4 years experience. Designed components in SolidWorks. Performed FEA. Strong in product design and GD&T principles.",
            "relevance_score": 93
        },
        {
            "jd_text": "Mechanical Design Engineer. Skills: CAD (SolidWorks), FEA, Product Design, Thermodynamics, GD&T. Experience: 3+ years.",
            "resume_text": "Manufacturing Engineer, 5 years experience. Focused on production processes. Basic CAD knowledge. No design or FEA experience.",
            "relevance_score": 35
        },
        {
            "jd_text": "Mechanical Design Engineer. Skills: CAD (SolidWorks), FEA, Product Design, Thermodynamics, GD&T. Experience: 3+ years.",
            "resume_text": "Recent Mechanical Engineering graduate. Strong in CAD software. Completed projects involving design and analysis. Eager to learn industry tools.",
            "relevance_score": 60
        },

        # --- Electrical Systems Engineer ---
        {
            "jd_text": "Electrical Systems Engineer. Skills: Circuit Design, Embedded Systems, PCB Layout, Microcontrollers, Signal Processing. Experience: 4+ years.",
            "resume_text": "Electrical Engineer, 5 years experience. Designed complex circuits. Developed embedded systems firmware. Proficient in PCB layout and microcontrollers.",
            "relevance_score": 95
        },
        {
            "jd_text": "Electrical Systems Engineer. Skills: Circuit Design, Embedded Systems, PCB Layout, Microcontrollers, Signal Processing. Experience: 4+ years.",
            "resume_text": "Software Engineer, 3 years experience. Worked on web applications. No hardware or embedded systems experience.",
            "relevance_score": 20
        },
        {
            "jd_text": "Electrical Systems Engineer. Skills: Circuit Design, Embedded Systems, PCB Layout, Microcontrollers, Signal Processing. Experience: 4+ years.",
            "resume_text": "Controls Engineer, 6 years experience. Designed control systems using PLCs. Some experience with circuit diagrams.",
            "relevance_score": 65
        },

        # --- Civil Structural Engineer ---
        {
            "jd_text": "Civil Structural Engineer. Skills: Structural Analysis, AutoCAD, Revit, Concrete Design, Steel Design. Experience: 3+ years.",
            "resume_text": "Structural Engineer with 4 years experience. Performed structural analysis for buildings. Proficient in AutoCAD and Revit. Designed concrete and steel structures.",
            "relevance_score": 92
        },
        {
            "jd_text": "Civil Structural Engineer. Skills: Structural Analysis, AutoCAD, Revit, Concrete Design, Steel Design. Experience: 3+ years.",
            "resume_text": "Civil Engineer, 5 years experience. Focused on transportation projects. Basic AutoCAD. No structural design experience.",
            "relevance_score": 30
        },
        {
            "jd_text": "Civil Structural Engineer. Skills: Structural Analysis, AutoCAD, Revit, Concrete Design, Steel Design. Experience: 3+ years.",
            "resume_text": "Construction Project Manager, 8 years experience. Managed construction sites. Read blueprints. Some understanding of structural elements.",
            "relevance_score": 55
        },

        # --- Research Chemist ---
        {
            "jd_text": "Research Chemist. Skills: Organic Synthesis, Analytical Techniques (HPLC, GC-MS), Spectroscopy, Lab Safety. Experience: 3+ years.",
            "resume_text": "Research Chemist, 4 years experience. Performed organic synthesis. Expert in HPLC and GC-MS. Conducted various spectroscopic analyses. Strong adherence to lab safety.",
            "relevance_score": 94
        },
        {
            "jd_text": "Research Chemist. Skills: Organic Synthesis, Analytical Techniques (HPLC, GC-MS), Spectroscopy, Lab Safety. Experience: 3+ years.",
            "resume_text": "Lab Technician, 2 years experience. Prepared samples and maintained equipment. No independent research or synthesis experience.",
            "relevance_score": 30
        },
        {
            "jd_text": "Research Chemist. Skills: Organic Synthesis, Analytical Techniques (HPLC, GC-MS), Spectroscopy, Lab Safety. Experience: 3+ years.",
            "resume_text": "Quality Control Chemist, 5 years experience. Performed routine analytical testing. Familiar with lab instruments. No research background.",
            "relevance_score": 60
        },

        # --- Clinical Biologist ---
        {
            "jd_text": "Clinical Biologist. Skills: Molecular Biology, Cell Culture, PCR, Flow Cytometry, Data Analysis. Experience: 3+ years.",
            "resume_text": "Clinical Biologist, 4 years experience. Performed molecular biology experiments. Proficient in cell culture and PCR. Used flow cytometry for analysis. Strong data analysis skills.",
            "relevance_score": 93
        },
        {
            "jd_text": "Clinical Biologist. Skills: Molecular Biology, Cell Culture, PCR, Flow Cytometry, Data Analysis. Experience: 3+ years.",
            "resume_text": "Medical Technologist, 5 years experience. Performed routine lab tests. No research or advanced molecular biology techniques.",
            "relevance_score": 25
        },
        {
            "jd_text": "Clinical Biologist. Skills: Molecular Biology, Cell Culture, PCR, Flow Cytometry, Data Analysis. Experience: 3+ years.",
            "resume_text": "Biomedical Researcher, 2 years experience. Focused on genetics. Some cell culture experience. Eager to work in clinical setting.",
            "relevance_score": 70
        },

        # --- Registered Nurse (ER) ---
        {
            "jd_text": "Registered Nurse (ER). Skills: Emergency Patient Care, Triage, Medication Administration, BLS/ACLS, EHR. Experience: 2+ years.",
            "resume_text": "ER Nurse with 3 years experience. Provided emergency patient care and performed triage. Proficient in medication administration. BLS/ACLS certified. Experienced with Epic EHR.",
            "relevance_score": 95
        },
        {
            "jd_text": "Registered Nurse (ER). Skills: Emergency Patient Care, Triage, Medication Administration, BLS/ACLS, EHR. Experience: 2+ years.",
            "resume_text": "Pediatric Nurse, 5 years experience. Worked in a children's hospital. No emergency room specific experience.",
            "relevance_score": 60
        },
        {
            "jd_text": "Registered Nurse (ER). Skills: Emergency Patient Care, Triage, Medication Administration, BLS/ACLS, EHR. Experience: 2+ years.",
            "resume_text": "Paramedic, 7 years experience. Provided pre-hospital emergency care. Strong in triage and patient assessment. No hospital RN experience.",
            "relevance_score": 75
        },

        # --- High School Math Teacher ---
        {
            "jd_text": "High School Math Teacher. Skills: Lesson Planning, Classroom Management, Algebra, Geometry, Calculus, Student Assessment. Experience: 2+ years.",
            "resume_text": "High School Math Teacher, 3 years experience. Developed engaging lesson plans for Algebra and Geometry. Strong classroom management. Conducted student assessments.",
            "relevance_score": 90
        },
        {
            "jd_text": "High School Math Teacher. Skills: Lesson Planning, Classroom Management, Algebra, Geometry, Calculus, Student Assessment. Experience: 2+ years.",
            "resume_text": "Elementary School Teacher, 7 years experience. Taught various subjects to young children. No high school specific math teaching experience.",
            "relevance_score": 40
        },
        {
            "jd_text": "High School Math Teacher. Skills: Lesson Planning, Classroom Management, Algebra, Geometry, Calculus, Student Assessment. Experience: 2+ years.",
            "resume_text": "Math Tutor, 5 years experience. Tutored high school and college students in various math subjects. Strong subject matter expertise. No formal classroom teaching.",
            "relevance_score": 70
        },

        # --- Additional Diverse Examples to Increase Dataset Size and Variety ---
        # Example: Data Engineer
        {
            "jd_text": "Data Engineer. Skills: ETL, Spark, Kafka, SQL, Python, Data Warehousing, AWS. Experience: 4+ years.",
            "resume_text": "Senior Data Engineer, 6 years. Built robust ETL pipelines using Spark and Kafka. Designed data warehouses on AWS Redshift. Proficient in Python and SQL.",
            "relevance_score": 96
        },
        {
            "jd_text": "Data Engineer. Skills: ETL, Spark, Kafka, SQL, Python, Data Warehousing, AWS. Experience: 4+ years.",
            "resume_text": "Database Administrator, 8 years. Managed SQL Server databases. Some scripting. No experience with big data streaming or cloud data platforms.",
            "relevance_score": 45
        },

        # Example: Content Marketing Manager
        {
            "jd_text": "Content Marketing Manager. Skills: Content Strategy, SEO, Copywriting, Social Media, Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Content Marketing Manager, 6 years. Developed and executed content strategies. Strong in SEO and copywriting. Led a small team. Used Google Analytics.",
            "relevance_score": 92
        },
        {
            "jd_text": "Content Marketing Manager. Skills: Content Strategy, SEO, Copywriting, Social Media, Analytics, Team Leadership. Experience: 5+ years.",
            "resume_text": "Journalist, 10 years experience. Excellent writing skills. No marketing or SEO experience.",
            "relevance_score": 30
        },

        # Example: IT Support Specialist
        {
            "jd_text": "IT Support Specialist. Skills: Troubleshooting, Windows, MacOS, Network Basics, Hardware, Customer Service. Experience: 1+ years.",
            "resume_text": "IT Support Technician, 2 years. Resolved hardware/software issues on Windows and MacOS. Basic network troubleshooting. Strong customer service.",
            "relevance_score": 88
        },
        {
            "jd_text": "IT Support Specialist. Skills: Troubleshooting, Windows, MacOS, Network Basics, Hardware, Customer Service. Experience: 1+ years.",
            "resume_text": "Sales Associate, 3 years. Good at interacting with people. No technical skills.",
            "relevance_score": 10
        },

        # Example: Research Scientist (Biotech)
        {
            "jd_text": "Research Scientist. Skills: Cell Biology, Assay Development, qPCR, Microscopy, Data Analysis (R/Python). Experience: PhD + 2 years post-doc.",
            "resume_text": "PhD in Cell Biology, 3 years post-doc. Expertise in cell culture and assay development. Performed qPCR and advanced microscopy. Analyzed data with R.",
            "relevance_score": 95
        },
        {
            "jd_text": "Research Scientist. Skills: Cell Biology, Assay Development, qPCR, Microscopy, Data Analysis (R/Python). Experience: PhD + 2 years post-doc.",
            "resume_text": "Lab Manager, 10 years. Managed lab operations and equipment. No direct research or data analysis responsibility.",
            "relevance_score": 40
        },

        # Example: Civil Engineer (Transportation)
        {
            "jd_text": "Civil Engineer (Transportation). Skills: Road Design, Traffic Modeling, AutoCAD Civil 3D, Stormwater Management, Project Management. Experience: 4+ years.",
            "resume_text": "Transportation Engineer, 5 years. Designed roadways and intersections. Performed traffic modeling. Proficient in Civil 3D. Managed small projects.",
            "relevance_score": 90
        },
        {
            "jd_text": "Civil Engineer (Transportation). Skills: Road Design, Traffic Modeling, AutoCAD Civil 3D, Stormwater Management, Project Management. Experience: 4+ years.",
            "resume_text": "Structural Engineer, 6 years. Designed building structures. Some AutoCAD. No transportation specific experience.",
            "relevance_score": 35
        },

        # Example: UX Researcher
        {
            "jd_text": "UX Researcher. Skills: User Interviews, Usability Testing, Survey Design, Data Analysis, Figma/Sketch. Experience: 3+ years.",
            "resume_text": "UX Researcher, 4 years experience. Conducted extensive user interviews and usability tests. Designed surveys. Analyzed qualitative and quantitative data. Used Figma for prototypes.",
            "relevance_score": 94
        },
        {
            "jd_text": "UX Researcher. Skills: User Interviews, Usability Testing, Survey Design, Data Analysis, Figma/Sketch. Experience: 3+ years.",
            "resume_text": "Market Researcher, 5 years experience. Designed consumer surveys and analyzed market trends. No specific UX research methodology.",
            "relevance_score": 60
        },

        # Example: Database Administrator
        {
            "jd_text": "Database Administrator. Skills: SQL Server, Oracle, Database Tuning, Backup/Recovery, Performance Monitoring. Experience: 5+ years.",
            "resume_text": "Senior DBA, 7 years experience. Managed SQL Server and Oracle databases. Expert in performance tuning and backup strategies. Implemented disaster recovery plans.",
            "relevance_score": 95
        },
        {
            "jd_text": "Database Administrator. Skills: SQL Server, Oracle, Database Tuning, Backup/Recovery, Performance Monitoring. Experience: 5+ years.",
            "resume_text": "Software Developer, 4 years. Wrote SQL queries for applications. No administrative or tuning experience.",
            "relevance_score": 40
        },

        # Example: Technical Project Manager
        {
            "jd_text": "Technical Project Manager. Skills: Agile, Scrum, Software Development Lifecycle, Risk Management, Stakeholder Communication. Experience: 7+ years.",
            "resume_text": "Technical Project Manager, 8 years. Led multiple software development projects using Agile/Scrum. Expert in risk management and stakeholder communication. Delivered on time and budget.",
            "relevance_score": 96
        },
        {
            "jd_text": "Technical Project Manager. Skills: Agile, Scrum, Software Development Lifecycle, Risk Management, Stakeholder Communication. Experience: 7+ years.",
            "resume_text": "Business Analyst, 5 years. Gathered requirements and created documentation. Familiar with Agile. No direct project management leadership.",
            "relevance_score": 60
        },

        # Example: Financial Controller
        {
            "jd_text": "Financial Controller. Skills: Financial Reporting, Budgeting, Forecasting, GAAP, Internal Controls, Team Leadership. Experience: 8+ years.",
            "resume_text": "Financial Controller, 10 years. Oversaw financial reporting, budgeting, and forecasting. Ensured GAAP compliance. Implemented strong internal controls. Managed finance team.",
            "relevance_score": 95
        },
        {
            "jd_text": "Financial Controller. Skills: Financial Reporting, Budgeting, Forecasting, GAAP, Internal Controls, Team Leadership. Experience: 8+ years.",
            "resume_text": "Senior Accountant, 6 years. Prepared journal entries and reconciliations. Some exposure to financial statements. No management experience.",
            "relevance_score": 50
        },

        # Example: HR Business Partner
        {
            "jd_text": "HR Business Partner. Skills: Strategic HR, Talent Management, Change Management, Employee Relations, Coaching. Experience: 6+ years.",
            "resume_text": "HR Business Partner, 7 years. Partnered with leadership on strategic HR initiatives. Drove talent management and change management programs. Expert in employee relations and coaching.",
            "relevance_score": 93
        },
        {
            "jd_text": "HR Business Partner. Skills: Strategic HR, Talent Management, Change Management, Employee Relations, Coaching. Experience: 6+ years.",
            "resume_text": "HR Administrator, 4 years. Handled HR operations and data entry. No strategic or advisory role.",
            "relevance_score": 40
        },

        # Example: Software Quality Assurance Engineer
        {
            "jd_text": "Software Quality Assurance Engineer. Skills: Test Automation, Selenium, JUnit/Pytest, CI/CD, Agile, Bug Tracking. Experience: 3+ years.",
            "resume_text": "QA Automation Engineer, 4 years. Developed test automation frameworks using Selenium and Pytest. Integrated tests into CI/CD pipelines. Proficient in Jira for bug tracking. Agile environment.",
            "relevance_score": 94
        },
        {
            "jd_text": "Software Quality Assurance Engineer. Skills: Test Automation, Selenium, JUnit/Pytest, CI/CD, Agile, Bug Tracking. Experience: 3+ years.",
            "resume_text": "Manual Tester, 5 years. Performed manual functional testing. No automation experience.",
            "relevance_score": 30
        },

        # Example: Network Engineer
        {
            "jd_text": "Network Engineer. Skills: Cisco, Juniper, Routing, Switching, Firewalls, VPN, Network Monitoring. Experience: 4+ years.",
            "resume_text": "Senior Network Engineer, 6 years. Designed and implemented enterprise networks using Cisco and Juniper. Expert in routing, switching, and firewall configurations. Managed VPNs and network monitoring.",
            "relevance_score": 96
        },
        {
            "jd_text": "Network Engineer. Skills: Cisco, Juniper, Routing, Switching, Firewalls, VPN, Network Monitoring. Experience: 4+ years.",
            "resume_text": "IT Support Specialist, 3 years. Basic network troubleshooting. No advanced routing or firewall experience.",
            "relevance_score": 35
        },

        # Example: Data Analyst (Entry Level)
        {
            "jd_text": "Data Analyst (Entry Level). Skills: Excel, SQL, Data Cleaning, Basic Statistics, Communication. Experience: 0-2 years.",
            "resume_text": "Recent graduate with a degree in Business Analytics. Strong Excel and SQL skills from coursework. Completed a project on data cleaning. Eager to start career.",
            "relevance_score": 80
        },
        {
            "jd_text": "Data Analyst (Entry Level). Skills: Excel, SQL, Data Cleaning, Basic Statistics, Communication. Experience: 0-2 years.",
            "resume_text": "Customer Service Representative, 5 years. Excellent communication. No data analysis skills.",
            "relevance_score": 10
        },

        # Example: Marketing Coordinator
        {
            "jd_text": "Marketing Coordinator. Skills: Event Planning, Social Media, Email Marketing, Content Support, CRM. Experience: 1+ years.",
            "resume_text": "Marketing Coordinator, 2 years. Assisted with event planning and social media campaigns. Supported email marketing. Used HubSpot CRM.",
            "relevance_score": 85
        },
        {
            "jd_text": "Marketing Coordinator. Skills: Event Planning, Social Media, Email Marketing, Content Support, CRM. Experience: 1+ years.",
            "resume_text": "Administrative Assistant, 3 years. Organized office events. No marketing experience.",
            "relevance_score": 20
        },

        # Example: Business Analyst (IT Focus)
        {
            "jd_text": "Business Analyst (IT). Skills: Requirements Gathering, Process Mapping, SQL, Agile, JIRA, Stakeholder Management. Experience: 3+ years.",
            "resume_text": "IT Business Analyst, 4 years. Gathered requirements for software projects. Mapped business processes. Proficient in SQL and JIRA. Worked in Agile teams.",
            "relevance_score": 92
        },
        {
            "jd_text": "Business Analyst (IT). Skills: Requirements Gathering, Process Mapping, SQL, Agile, JIRA, Stakeholder Management. Experience: 3+ years.",
            "resume_text": "Financial Analyst, 5 years. Performed financial reporting. Some data analysis. No IT project experience.",
            "relevance_score": 40
        },

        # Example: Mechanical Engineer (Entry Level)
        {
            "jd_text": "Mechanical Engineer (Entry Level). Skills: CAD (SolidWorks), Thermodynamics, Materials Science, Prototyping. Experience: 0-2 years.",
            "resume_text": "Recent Mechanical Engineering graduate. Strong SolidWorks skills from university projects. Coursework in Thermodynamics and Materials Science. Built a prototype in capstone project.",
            "relevance_score": 80
        },
        {
            "jd_text": "Mechanical Engineer (Entry Level). Skills: CAD (SolidWorks), Thermodynamics, Materials Science, Prototyping. Experience: 0-2 years.",
            "resume_text": "Manufacturing Technician, 5 years. Operated machinery. No design or engineering degree.",
            "relevance_score": 15
        },

        # Example: Registered Nurse (General)
        {
            "jd_text": "Registered Nurse. Skills: Patient Care, Medication Administration, Electronic Health Records, Communication, Critical Thinking. Experience: 1+ years.",
            "resume_text": "Registered Nurse, 2 years. Provided direct patient care. Administered medications. Documented in EHR. Strong communication and critical thinking.",
            "relevance_score": 90
        },
        {
            "jd_text": "Registered Nurse. Skills: Patient Care, Medication Administration, Electronic Health Records, Communication, Critical Thinking. Experience: 1+ years.",
            "resume_text": "Certified Nursing Assistant, 5 years. Assisted patients with daily care. No medication administration or EHR experience.",
            "relevance_score": 40
        },
    ] # Placeholder for your synthetic data

    if not synthetic_data:
        print("Error: synthetic_data list is empty. Please populate it with training data.")
    else:
        print(f"Generated {len(synthetic_data)} synthetic data points for training.")

        # Prepare data for training
        X = []  # Features
        y = []  # Relevance scores

        for entry in synthetic_data:
            jd_text = entry["jd_text"]
            resume_text = entry["resume_text"]
            relevance_score = entry["relevance_score"]

            features = create_features(jd_text, resume_text, jd_embedding_model, resume_embedding_model)
            X.append(features)
            y.append(relevance_score)

        X = np.array(X)
        y = np.array(y)

        print(f"Features created. X shape: {X.shape}, y shape: {y.shape}")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300], # Number of trees in the forest
            'max_depth': [10, 20, None],     # Maximum depth of the tree
            'min_samples_leaf': [1, 2, 4]    # Minimum number of samples required to be at a leaf node
        }

        # Initialize RandomForestRegressor
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)

        # Initialize GridSearchCV
        # cv=3 means 3-fold cross-validation
        # scoring='r2' means optimize for R-squared
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')

        print("Starting GridSearchCV for hyperparameter tuning...")
        grid_search.fit(X_train, y_train)

        # Get the best model from GridSearchCV
        model = grid_search.best_estimator_
        print("RandomForestRegressor trained with best hyperparameters.")
        print(f"Best parameters found: {grid_search.best_params_}")

        # Evaluate the best model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Model Evaluation:")
        print(f"  Mean Squared Error (MSE): {mse:.2f}")
        print(f"  R-squared (R2): {r2:.2f}")

        # Save the trained model
        joblib.dump(model, MODEL_SAVE_PATH)
        print(f"Model saved successfully to {MODEL_SAVE_PATH}")
