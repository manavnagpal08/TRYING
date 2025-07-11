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
from sentence_transformers import SentenceTransformer
import nltk
import collections
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse # For encoding mailto links

# Import T5 specific libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Download NLTK stopwords data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet') # For lemmatization
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger') # For POS tagging
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet # To map POS tags for WordNetLemmatizer


# --- Load Embedding + ML Model ---
@st.cache_resource
def load_ml_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ml_model = joblib.load("ml_screening_model.pkl")
        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory.")
        return None, None

# --- Load T5 Model ---
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
model, ml_model = load_ml_model()
t5_tokenizer, t5_model = load_t5_model()

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# --- Helper function to map NLTK POS tags to WordNetLemmatizer tags ---
def get_wordnet_pos(treebank_tag):
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


# --- Stop Words List (Using NLTK) ---
NLTK_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
# Add your comprehensive custom stop words list here.
# For brevity, I'm using a placeholder but you should paste your large list from the previous response.
# This list is crucial. If you don't paste it, many generic words will remain.
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
    "fcsp", "fcc", "fcnsp", "fct", "fcp", "fcs", "fce", "fcn", "fcnp", "fcnse",
    # General words that might appear in "skills" section but aren't actual skills
    "skills", "skill", "improve", "improving", "ability", "abilities", "knowledge", "proficient",
    "expertise", "experience", "experienced", "background", "capabilities", "competencies",
    "competency", "develop", "developing", "developed", "learn", "learning", "mastery",
    "understanding", "areas", "area", "technical", "soft", "communication", "leadership",
    "problem-solving", "critical-thinking", "adaptability", "creativity", "teamwork", "collaboration",
    "interpersonal", "organizational", "management", "strategic", "tactical", "operational",
    "excellent", "strong", "good", "basic", "intermediate", "advanced", "proficient",
    "demonstrated", "proven", "track record", "results", "driven", "achievements", "accomplishments",
    "responsibilities", "duties", "tasks", "roles", "role", "key", "summary", "profile",
    "objective", "education", "certifications", "awards", "honors", "publications", "interests",
    "references", "portfolio", "contact", "phone", "email", "linkedin", "github", "website",
    "address", "city", "state", "zip", "country", "national", "international", "global",
    "remote", "hybrid", "onsite", "full-time", "part-time", "contract", "freelance", "internship",
    "volunteer", "education", "degree", "bachelor", "master", "phd", "university", "college",
    "institute", "school", "major", "minor", "gpa", "course", "courses", "class", "classes",
    "project", "projects", "thesis", "dissertation", "research", "paper", "papers", "journal",
    "journals", "conference", "conferences", "presentation", "presentations", "workshop", "workshops",
    "seminar", "seminars", "training", "trainings", "certification", "certifications", "license",
    "licenses", "award", "awards", "honor", "honors", "distinction", "distinctions", "scholarship",
    "scholarships", "fellowship", "fellowships", "grant", "grants", "patent", "patents",
    "publication", "publications", "article", "articles", "book", "books", "chapter", "chapters",
    "report", "reports", "manual", "manuals", "guide", "guides", "documentation", "documentations",
    "technical report", "white paper", "case study", "case studies", "solution architect",
    "data scientist", "machine learning engineer", "software developer", "devops engineer",
    "cloud engineer", "cybersecurity analyst", "product manager", "project manager",
    "business analyst", "marketing manager", "sales manager", "hr manager", "financial analyst",
    "accountant", "auditor", "consultant", "director", "manager", "lead", "senior", "junior",
    "associate", "specialist", "coordinator", "assistant", "intern", "engineer", "analyst",
    "architect", "strategist", "expert", "professional", "consultant", "advisor", "officer",
    "executive", "president", "vice president", "ceo", "cto", "cfo", "coo", "chief", "head",
    "group", "division", "department", "unit", "section", "team", "office", "company", "corporation",
    "inc", "ltd", "llc", "corp", "group", "holdings", "solutions", "services", "technologies",
    "systems", "consulting", "advisory", "management", "financial", "digital", "global",
    "international", "national", "regional", "local", "public", "private", "government",
    "non-profit", "startup", "mid-size", "enterprise", "fortune", "global", "innovative",
    "cutting-edge", "leading", "pioneering", "transformative", "disruptive", "scalable",
    "robust", "reliable", "secure", "efficient", "effective", "optimized", "automated",
    "integrated", "seamless", "user-friendly", "intuitive", "responsive", "dynamic",
    "interactive", "engaging", "compelling", "impactful", "sustainable", "ethical",
    "compliant", "governance", "risk", "compliance", "regulatory", "standard", "standards",
    "best practices", "methodology", "methodologies", "process", "processes", "procedure",
    "procedures", "guideline", "guidelines", "framework", "frameworks", "tool", "tools",
    "technology", "technologies", "platform", "platforms", "solution", "solutions",
    "system", "systems", "architecture", "design", "development", "implementation",
    "deployment", "maintenance", "support", "operations", "monitoring", "analysis",
    "reporting", "visualization", "dashboard", "dashboards", "metrics", "kpis", "performance",
    "optimization", "automation", "integration", "migration", "transformation", "upgrade",
    "update", "patch", "patches", "troubleshooting", "debugging", "testing", "quality",
    "assurance", "control", "auditing", "compliance", "security", "privacy", "data",
    "information", "analytics", "intelligence", "insight", "insights", "strategy",
    "planning", "execution", "management", "leadership", "mentoring", "coaching",
    "training", "development", "recruitment", "hiring", "onboarding", "retention",
    "employee", "engagement", "relations", "compensation", "benefits", "payroll",
    "hr", "human resources", "talent acquisition", "talent management", "workforce",
    "diversity", "inclusion", "equity", "belonging", "csr", "sustainability", "environmental",
    "social", "governance", "ethics", "integrity", "professionalism", "communication",
    "presentation", "negotiation", "collaboration", "teamwork", "interpersonal",
    "problem solving", "critical thinking", "analytical", "creative", "innovative",
    "adaptable", "flexible", "resilient", "organized", "detail-oriented", "proactive",
    "self-starter", "independent", "results-driven", "client-facing", "stakeholder management",
    "vendor management", "budget management", "cost reduction", "process improvement",
    "standardization", "quality management", "project management", "program management",
    "portfolio management", "agile", "scrum", "kanban", "waterfall", "lean", "six sigma",
    "pmp", "prince2", "itil", "cobit", "cism", "cissp", "ceh", "security+", "network+",
    "a+", "linux+", "ccna", "ccnp", "ccie", "aws", "azure", "gcp", "certified",
    "developer", "architect", "sysops", "administrator", "specialty", "professional",
    "expert", "master", "principal", "distinguished", "fellow", "senior staff", "staff",
    "junior staff", "associate staff", "intern", "co-op", "trainee", "apprentice",
    "volunteer", "pro-bono", "freelance", "contract", "temp", "full-time", "part-time",
    "casual", "seasonal", "gig", "remote", "hybrid", "onsite", "in-office", "field-based",
    "travel", "relocation", "visa sponsorship", "eligible to work", "right to work",
    "driver's license", "car", "own transport", "flexible hours", "on-call", "shift work",
    "overtime", "weekend work", "public holidays", "bank holidays", "paid leave",
    "unpaid leave", "sick leave", "maternity leave", "paternity leave", "parental leave",
    "bereavement leave", "sabbatical", "retirement", "pension", "superannuation",
    "health insurance", "dental insurance", "vision insurance", "life insurance",
    "disability insurance", "critical illness", "employee assistance program", "eap",
    "wellness program", "gym membership", "subsidized meals", "company car", "mobile phone",
    "laptop", "home office allowance", "training budget", "professional development",
    "mentorship", "coaching", "career progression", "internal mobility", "job rotation",
    "secondment", "tuition reimbursement", "education assistance", "student loan repayment",
    "childcare vouchers", "cycle to work", "share options", "stock options", "equity",
    "bonus", "commission", "profit share", "salary", "wage", "remuneration", "package",
    "compensation", "benefits", "perks", "allowances", "expenses", "reimbursement",
    "tax-efficient", "salary sacrifice", "pension contributions", "medical aid",
    "401k", "403b", "457", "ira", "roth ira", "sep ira", "simple ira", "espp", "rsu",
    "ltdi", "stdi", "adr", "arbitration", "mediation", "grievance", "disciplinary",
    "code of conduct", "ethics policy", "confidentiality agreement", "nda", "non-compete",
    "non-solicitation", "ip assignment", "offer letter", "contract of employment",
    "employee handbook", "company policy", "procedure manual", "compliance training",
    "health and safety", "hse", "ohs", "osh", "ergonomics", "fire safety", "first aid",
    "incident reporting", "accident investigation", "risk assessment", "hazard identification",
    "safe work procedures", "emergency preparedness", "business continuity", "disaster recovery",
    "crisis management", "crisis communication", "public relations", "media relations",
    "investor relations", "shareholder relations", "government relations", "lobbying",
    "community relations", "corporate social responsibility", "csr report", "sustainability report",
    "environmental impact", "carbon footprint", "waste management", "recycling", "renewable energy",
    "green initiatives", "eco-friendly", "fair trade", "ethical sourcing", "supply chain ethics",
    "human rights", "labor practices", "child labor", "forced labor", "modern slavery",
    "equal opportunity", "affirmative action", "diversity and inclusion", "unconscious bias",
    "harassment prevention", "discrimination prevention", "grievance procedure", "whistleblowing",
    "internal audit", "external audit", "financial audit", "operational audit", "compliance audit",
    "it audit", "security audit", "quality audit", "environmental audit", "social audit",
    "due diligence", "mergers and acquisitions", "m&a", "divestitures", "joint ventures",
    "strategic alliances", "partnerships", "outsourcing", "insourcing", "offshoring",
    "nearshoring", "reshoring", "vendor management", "supplier relationship management",
    "contract negotiation", "contract management", "procurement", "purchasing", "sourcing",
    "logistics", "supply chain", "inventory management", "warehouse management",
    "transportation management", "fleet management", "route optimization", "demand planning",
    "forecasting", "production planning", "manufacturing execution system", "mes",
    "enterprise resource planning", "erp", "customer relationship management", "crm",
    "supply chain management", "scm", "human capital management", "hcm", "financial management",
    "accounting software", "payroll software", "hr software", "crm software", "erp software",
    "project management software", "collaboration tools", "communication tools",
    "video conferencing", "web conferencing", "document management", "content management",
    "knowledge management", "business intelligence", "bi", "data warehousing", "data lakes",
    "data marts", "etl", "data integration", "data governance", "data quality", "data migration",
    "data modeling", "data architecture", "database administration", "dba", "sql", "nosql",
    "data science", "machine learning", "deep learning", "artificial intelligence", "ai",
    "natural language processing", "nlp", "computer vision", "cv", "predictive analytics",
    "prescriptive analytics", "descriptive analytics", "statistical analysis", "data mining",
    "big data", "hadoop", "spark", "kafka", "tableau", "power bi", "qlikview", "excel",
    "r", "python", "sas", "spss", "matlab", "stata", "azure machine learning",
    "aws sagemaker", "google ai platform", "tensorflow", "pytorch", "keras", "scikit-learn",
    "xgboost", "lightgbm", "catboost", "statsmodels", "numpy", "pandas",
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
    "fcsp", "fcc", "fcnsp", "fct", "fcp", "fcs", "fce", "fcn", "fcnp", "fcnse",
    # General words that might appear in "skills" section but aren't actual skills
    "skills", "skill", "improve", "improving", "ability", "abilities", "knowledge", "proficient",
    "expertise", "experience", "experienced", "background", "capabilities", "competencies",
    "competency", "develop", "developing", "developed", "learn", "learning", "mastery",
    "understanding", "areas", "area", "technical", "soft", "communication", "leadership",
    "problem-solving", "critical-thinking", "adaptability", "creativity", "teamwork", "collaboration",
    "interpersonal", "organizational", "management", "strategic", "tactical", "operational",
    "excellent", "strong", "good", "basic", "intermediate", "advanced", "proficient",
    "demonstrated", "proven", "track record", "results", "driven", "achievements", "accomplishments",
    "responsibilities", "duties", "tasks", "roles", "role", "key", "summary", "profile",
    "objective", "education", "certifications", "awards", "honors", "publications", "interests",
    "references", "portfolio", "contact", "phone", "email", "linkedin", "github", "website",
    "address", "city", "state", "zip", "country", "national", "international", "global",
    "remote", "hybrid", "onsite", "full-time", "part-time", "contract", "freelance", "internship",
    "volunteer", "education", "degree", "bachelor", "master", "phd", "university", "college",
    "institute", "school", "major", "minor", "gpa", "course", "courses", "class", "classes",
    "project", "projects", "thesis", "dissertation", "research", "paper", "papers", "journal",
    "journals", "conference", "conferences", "presentation", "presentations", "workshop", "workshops",
    "seminar", "seminars", "training", "trainings", "certification", "certifications", "license",
    "licenses", "award", "awards", "honor", "honors", "distinction", "distinctions", "scholarship",
    "scholarships", "fellowship", "fellowships", "grant", "grants", "patent", "patents",
    "publication", "publications", "article", "articles", "book", "books", "chapter", "chapters",
    "report", "reports", "manual", "manuals", "guide", "guides", "documentation", "documentations",
    "technical report", "white paper", "case study", "case studies", "solution architect",
    "data scientist", "machine learning engineer", "software developer", "devops engineer",
    "cloud engineer", "cybersecurity analyst", "product manager", "project manager",
    "business analyst", "marketing manager", "sales manager", "hr manager", "financial analyst",
    "accountant", "auditor", "consultant", "director", "manager", "lead", "senior", "junior",
    "associate", "specialist", "coordinator", "assistant", "intern", "engineer", "analyst",
    "architect", "strategist", "expert", "professional", "consultant", "advisor", "officer",
    "executive", "president", "vice president", "ceo", "cto", "cfo", "coo", "chief", "head",
    "group", "division", "department", "unit", "section", "team", "office", "company", "corporation",
    "inc", "ltd", "llc", "corp", "group", "holdings", "solutions", "services", "technologies",
    "systems", "consulting", "advisory", "management", "financial", "digital", "global",
    "international", "national", "regional", "local", "public", "private", "government",
    "non-profit", "startup", "mid-size", "enterprise", "fortune", "global", "innovative",
    "cutting-edge", "leading", "pioneering", "transformative", "disruptive", "scalable",
    "robust", "reliable", "secure", "efficient", "effective", "optimized", "automated",
    "integrated", "seamless", "user-friendly", "intuitive", "responsive", "dynamic",
    "interactive", "engaging", "compelling", "impactful", "sustainable", "ethical",
    "compliant", "governance", "risk", "compliance", "regulatory", "standard", "standards",
    "best practices", "methodology", "methodologies", "process", "processes", "procedure",
    "procedures", "guideline", "guidelines", "framework", "frameworks", "tool", "tools",
    "technology", "technologies", "platform", "platforms", "solution", "solutions",
    "system", "systems", "architecture", "design", "development", "implementation",
    "deployment", "maintenance", "support", "operations", "monitoring", "analysis",
    "reporting", "visualization", "dashboard", "dashboards", "metrics", "kpis", "performance",
    "optimization", "automation", "integration", "migration", "transformation", "upgrade",
    "update", "patch", "patches", "troubleshooting", "debugging", "testing", "quality",
    "assurance", "control", "auditing", "compliance", "security", "privacy", "data",
    "information", "analytics", "intelligence", "insight", "insights", "strategy",
    "planning", "execution", "management", "leadership", "mentoring", "coaching",
    "training", "development", "recruitment", "hiring", "onboarding", "retention",
    "employee", "engagement", "relations", "compensation", "benefits", "payroll",
    "hr", "human resources", "talent acquisition", "talent management", "workforce",
    "diversity", "inclusion", "equity", "belonging", "csr", "sustainability", "environmental",
    "social", "governance", "ethics", "integrity", "professionalism", "communication",
    "presentation", "negotiation", "collaboration", "teamwork", "interpersonal",
    "problem solving", "critical thinking", "analytical", "creative", "innovative",
    "adaptable", "flexible", "resilient", "organized", "detail-oriented", "proactive",
    "self-starter", "independent", "results-driven", "client-facing", "stakeholder management",
    "vendor management", "budget management", "cost reduction", "process improvement",
    "standardization", "quality management", "project management", "program management",
    "portfolio management", "agile", "scrum", "kanban", "waterfall", "lean", "six sigma",
    "pmp", "prince2", "itil", "cobit", "cism", "cissp", "ceh", "security+", "network+",
    "a+", "linux+", "ccna", "ccnp", "ccie", "aws", "azure", "gcp", "certified",
    "developer", "architect", "sysops", "administrator", "specialty", "professional",
    "expert", "master", "principal", "distinguished", "fellow", "senior staff", "staff",
    "junior staff", "associate staff", "intern", "co-op", "trainee", "apprentice",
    "volunteer", "pro-bono", "freelance", "contract", "temp", "full-time", "part-time",
    "casual", "seasonal", "gig", "remote", "hybrid", "onsite", "in-office", "field-based",
    "travel", "relocation", "visa sponsorship", "eligible to work", "right to work",
    "driver's license", "car", "own transport", "flexible hours", "on-call", "shift work",
    "overtime", "weekend work", "public holidays", "bank holidays", "paid leave",
    "unpaid leave", "sick leave", "maternity leave", "paternity leave", "parental leave",
    "bereavement leave", "sabbatical", "retirement", "pension", "superannuation",
    "health insurance", "dental insurance", "vision insurance", "life insurance",
    "disability insurance", "critical illness", "employee assistance program", "eap",
    "wellness program", "gym membership", "subsidized meals", "company car", "mobile phone",
    "laptop", "home office allowance", "training budget", "professional development",
    "mentorship", "coaching", "career progression", "internal mobility", "job rotation",
    "secondment", "tuition reimbursement", "education assistance", "student loan repayment",
    "childcare vouchers", "cycle to work", "share options", "stock options", "equity",
    "bonus", "commission", "profit share", "salary", "wage", "remuneration", "package",
    "compensation", "benefits", "perks", "allowances", "expenses", "reimbursement",
    "tax-efficient", "salary sacrifice", "pension contributions", "medical aid",
    "401k", "403b", "457", "ira", "roth ira", "sep ira", "simple ira", "espp", "rsu",
    "ltdi", "stdi", "adr", "arbitration", "mediation", "grievance", "disciplinary",
    "code of conduct", "ethics policy", "confidentiality agreement", "nda", "non-compete",
    "non-solicitation", "ip assignment", "offer letter", "contract of employment",
    "employee handbook", "company policy", "procedure manual", "compliance training",
    "health and safety", "hse", "ohs", "osh", "ergonomics", "fire safety", "first aid",
    "incident reporting", "accident investigation", "risk assessment", "hazard identification",
    "safe work procedures", "emergency preparedness", "business continuity", "disaster recovery",
    "crisis management", "crisis communication", "public relations", "media relations",
    "investor relations", "shareholder relations", "government relations", "lobbying",
    "community relations", "corporate social responsibility", "csr report", "sustainability report",
    "environmental impact", "carbon footprint", "waste management", "recycling", "renewable energy",
    "green initiatives", "eco-friendly", "fair trade", "ethical sourcing", "supply chain ethics",
    "human rights", "labor practices", "child labor", "forced labor", "modern slavery",
    "equal opportunity", "affirmative action", "diversity and inclusion", "unconscious bias",
    "harassment prevention", "discrimination prevention", "grievance procedure", "whistleblowing",
    "internal audit", "external audit", "financial audit", "operational audit", "compliance audit",
    "it audit", "security audit", "quality audit", "environmental audit", "social audit",
    "due diligence", "mergers and acquisitions", "m&a", "divestitures", "joint ventures",
    "strategic alliances", "partnerships", "outsourcing", "insourcing", "offshoring",
    "nearshoring", "reshoring", "vendor management", "supplier relationship management",
    "contract negotiation", "contract management", "procurement", "purchasing", "sourcing",
    "logistics", "supply chain", "inventory management", "warehouse management",
    "transportation management", "fleet management", "route optimization", "demand planning",
    "forecasting", "production planning", "manufacturing execution system", "mes",
    "enterprise resource planning", "erp", "customer relationship management", "crm",
    "supply chain management", "scm", "human capital management", "hcm", "financial management",
    "accounting software", "payroll software", "hr software", "crm software", "erp software",
    "project management software", "collaboration tools", "communication tools",
    "video conferencing", "web conferencing", "document management", "content management",
    "knowledge management", "business intelligence", "bi", "data warehousing", "data lakes",
    "data marts", "etl", "data integration", "data governance", "data quality", "data migration",
    "data modeling", "data architecture", "database administration", "dba", "sql", "nosql",
    "data science", "machine learning", "deep learning", "artificial intelligence", "ai",
    "natural language processing", "nlp", "computer vision", "cv", "predictive analytics",
    "prescriptive analytics", "descriptive analytics", "statistical analysis", "data mining",
    "big data", "hadoop", "spark", "kafka", "tableau", "power bi", "qlikview", "excel",
    "r", "python", "sas", "spss", "matlab", "stata", "azure machine learning",
    "aws sagemaker", "google ai platform", "tensorflow", "pytorch", "keras", "scikit-learn",
    "xgboost", "lightgbm", "catboost", "statsmodels", "numpy", "pandas",
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
