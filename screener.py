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


# --- Stop Words List (Using NLTK) ---
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
    "xgboost", "lightgbm", "catboost", "r studio", "jupyter", "databricks", "snowflake",
    "redshift", "synapse", "bigquery", "azure data lake", "aws s3", "google cloud storage",
    "relational database", "mongodb", "cassandra", "redis", "elasticsearch", "kafka streams",
    "apache flink", "apache spark", "apache kafka", "apache hadoop", "data bricks", "dbt",
    "airflow", "luigi", "prefect", "mlflow", "kubeflow", "docker", "kubernetes", "ansible",
    "terraform", "jenkins", "gitlab ci/cd", "github actions", "azure devops", "aws codepipeline",
    "google cloud build", "ci/cd", "devops", "site reliability engineering", "sre",
    "infrastructure as code", "iac", "cloud computing", "aws", "azure", "gcp", "private cloud",
    "hybrid cloud", "multi-cloud", "serverless", "lambda", "functions", "cloud functions",
    "api gateway", "microservices", "service mesh", "istio", "linkerd", "grpc", "restful apis",
    "soap", "queues", "messaging", "rabbitmq", "activemq", "apache kafka", "azure service bus",
    "aws sqs", "aws sns", "google cloud pub/sub", "version control", "git", "github", "gitlab",
    "bitbucket", "svn", "mercurial", "jiras", "confluence", "slack", "microsoft teams",
    "zoom", "google meet", "webex", "skype", "email", "outlook", "gmail", "calendaring",
    "scheduling", "microsoft office", "google workspace", "g suite", "microsoft 365",
    "visio", "draw.io", "lucidchart", "powerpoint", "keynote", "google slides",
    "word", "google docs", "excel", "google sheets", "project management", "agile tools",
    "jira", "trello", "asana", "monday.com", "smartsheet", "microsoft project",
    "html", "css", "javascript", "typescript", "react", "angular", "vue.js", "svelte",
    "jquery", "bootstrap", "tailwind css", "sass", "less", "webpack", "babel", "npm", "yarn",
    "node.js", "express.js", "python", "django", "flask", "fastapi", "java", "spring boot",
    "c#", ".net core", "php", "laravel", "symphony", "ruby", "rails", "go", "golang",
    "swift", "kotlin", "scala", "clojure", "rust", "frontend", "backend", "fullstack",
    "web development", "mobile development", "ios", "android", "react native", "flutter",
    "xamarin", "native script", "ionic", "progressive web apps", "pwa", "single page application",
    "spa", "rest api design", "graphql api", "micro-frontend", "server-side rendering",
    "client-side rendering", "state management", "redux", "mobx", "vuex", "ngrx",
    "context api", "web sockets", "real-time applications", "authentication", "authorization",
    "oauth", "jwt", "openid connect", "saml", "active directory", "ldap", "okta", "auth0",
    "keycloak", "security testing", "penetration testing", "vulnerability scanning",
    "static code analysis", "dynamic code analysis", "security awareness", "threat modeling",
    "incident response", "disaster recovery", "business continuity", "bcp", "drp",
    "gdpr", "hipaa", "ccpa", "soc 2", "iso 27001", "nist", "pci dss", "cybersecurity frameworks",
    "firewalls", "ids", "ips", "vpn", "endpoint protection", "antivirus", "anti-malware",
    "siem", "soc", "network security", "cloud security", "application security",
    "data security", "identity and access management", "iam", "privileged access management", "pam",
    "encryption", "decryption", "hashing", "digital signatures", "ssl/tls", "pkis",
    "cryptography", "blockchain", "distributed ledger technology", "dlt", "smart contracts",
    "decentralized applications", "dapps", "cryptocurrencies", "nfts", "web3", "metaverse",
    "augmented reality", "ar", "virtual reality", "vr", "mixed reality", "mr",
    "internet of things", "iot", "edge computing", "quantum computing", "robotics",
    "automation", "rpa", "robotic process automation", "chatbots", "virtual assistants",
    "natural language understanding", "nlu", "natural language generation", "nlg",
    "speech recognition", "text-to-speech", "computer vision", "image processing",
    "video analytics", "facial recognition", "object detection", "image classification",
    "sentiment analysis", "entity recognition", "topic modeling", "summarization",
    "translation", "recommendation systems", "recommender systems", "fraud detection",
    "anomaly detection", "predictive maintenance", "demand forecasting", "supply chain optimization",
    "resource allocation", "scheduling", "route optimization", "pricing optimization",
    "customer segmentation", "churn prediction", "lead scoring", "credit scoring",
    "risk assessment", "portfolio optimization", "algorithmic trading", "high-frequency trading",
    "quantitative analysis", "financial modeling", "econometrics", "statistical modeling",
    "actuarial science", "biostatistics", "epidemiology", "clinical trials", "pharmacology",
    "genomics", "bioinformatics", "biotechnology", "life sciences", "healthcare", "pharma",
    "medical devices", "diagnostics", "hospitals", "clinics", "telehealth", "ehealth",
    "fintech", "insurtech", "regtech", "edtech", "proptech", "legaltech", "agritech",
    "foodtech", "traveltech", "gaming", "esports", "media", "entertainment", "publishing",
    "advertising", "marketing", "digital marketing", "seo", "sem", "social media marketing",
    "content marketing", "email marketing", "affiliate marketing", "influencer marketing",
    "event marketing", "public relations", "brand management", "product management",
    "product development", "product lifecycle management", "plm", "go-to-market strategy",
    "market research", "competitor analysis", "swot analysis", "pestel analysis",
    "customer journey", "user experience", "ux", "user interface", "ui", "ux design",
    "ui design", "user research", "usability testing", "wireframing", "prototyping",
    "user flows", "information architecture", "interaction design", "visual design",
    "motion graphics", "graphic design", "web design", "mobile design", "branding",
    "typography", "color theory", "layout", "composition", "illustration", "photography",
    "video production", "animation", "audio production", "sound design", "music production",
    "voice acting", "scriptwriting", "storytelling", "copywriting", "editing", "proofreading",
    "localization", "internationalization", "translation", "transcription", "dubbing",
    "subtitling", "interpreting", "customer service", "customer support", "technical support",
    "help desk", "call center", "field service", "client relations", "account management",
    "sales", "business development", "lead generation", "cold calling", "sales presentations",
    "negotiation", "closing sales", "crm software", "salesforce", "hubspot", "microsoft dynamics",
    "sap crm", "oracle crm", "zendesk sales", "freshsales", "pipedrive", "monday sales crm",
    "zoho crm", "insightly", "agile crm", "bitrix24", "capsule crm", "teamleader",
    "v-tiger", "microsoft access", "filemaker", "tableau", "powerbi", "qlik", "domo",
    "looker", "microstrategy", "cognos", "sas visual analytics", "spotfire", "datawrapper",
    "infogram", "chartio", "metabase", "redash", "supersets", "dash", "bokeh", "plotly",
    "matplotlib", "seaborn", "ggplot2", "d3.js", "chart.js", "highcharts", "echarts",
    "google charts", "microsoft charts", "openpyxl", "pandas", "numpy", "scipy", "statsmodels",
    "numba", "cython", "dask", "modin", "vaex", "polars", "koalas", "pyspark", "sparklyr",
    "datatable", "fastai", "hugging face", "transformers", "pytorch lightning", "keras tuner",
    "mlflow", "neptune.ai", "weights & biases", "comet ml", "tensorboard", "streamlit",
    "dash", "voila", "panel", "gradio", "shinylive", "fastpages", "mkdocs", "sphinx",
    "swagger", "openapi", "postman", "insomnia", "soapui", "jmeter", "locust", "k6",
    "blazemeter", "gatling", "artillery", "cypress", "selenium", "playwright", "puppeteer",
    "jest", "mocha", "chai", "jasmine", "karma", "enzyme", "react testing library",
    "vue test utils", "angular testing", "j-unit", "n-unit", "x-unit", "pytest", "unittest",
    "doctest", "robot framework", "cucumber", "gherkin", "specflow", "behave", "lettuce",
    "gauge", "testrail", "zephyr", "qtest", "xray", "hp alm", "micro focus alm",
    "azure test plans", "aws device farm", "google firebase test lab", "browserstack",
    "sauce labs", "lambdatest", "crossbrowsertesting", "applitools", "percy", "storybook",
    "chromatic", "figma", "sketch", "adobe xd", "invision", "zeplin", "marvel app",
    "framer", "principle", "abstract", "gitlfs", "sourcetree", "github desktop", "fork",
    "sublime merge", "vscode git", "jetbrains git", "command line git", "github cli",
    "gitlab cli", "azure cli", "aws cli", "gcloud cli", "kubectl", "helm", "terraform cli",
    "ansible cli", "packer", "vagrant", "virtualbox", "vmware workstation", "parallels",
    "hyper-v", "proxmox", "kvm", "xen", "openstack", "cloudstack", "eucalyptus", "bare metal",
    "on-premise", "data center", "colo", "hosting", "managed services", "support services",
    "professional services", "consulting services", "training services", "education services",
    "certification services", "advisory services", "staff augmentation", "recruiting services",
    "headhunting", "executive search", "contingency search", "retained search",
    "hr consulting", "talent strategy", "workforce planning", "organizational development",
    "change management", "leadership development", "performance management", "learning & development",
    "employee relations", "industrial relations", "labor law", "employment law", "hr policies",
    "hr procedures", "hr systems", "hr metrics", "hr analytics", "hr dashboards", "hr reporting",
    "compliance management", "regulatory compliance", "audit readiness", "internal controls",
    "sarbanes-oxley", "sox", "dodd-frank", "basel iii", "solvency ii", "ifrs", "gaap",
    "financial reporting", "management accounting", "cost accounting", "budgeting",
    "forecasting", "financial analysis", "variance analysis", "cash flow management",
    "treasury management", "risk management", "credit risk", "market risk", "operational risk",
    "liquidity risk", "cyber risk", "strategic risk", "reputational risk", "legal risk",
    "compliance risk", "enterprise risk management", "erm", "internal audit", "external audit",
    "fraud examination", "forensic accounting", "tax planning", "tax compliance",
    "transfer pricing", "international taxation", "investment analysis", "portfolio management",
    "asset management", "wealth management", "financial planning", "retirement planning",
    "estate planning", "insurance sales", "underwriting", "claims management", "actuarial science",
    "p&c insurance", "life insurance", "health insurance", "reinsurance", "brokerage",
    "financial markets", "equities", "fixed income", "derivatives", "foreign exchange",
    "commodities", "futures", "options", "swaps", "bonds", "stocks", "indices", "etfs",
    "mutual funds", "hedge funds", "private equity", "venture capital", "angel investing",
    "crowdfunding", "m&a advisory", "corporate finance", "project finance", "structured finance",
    "debt financing", "equity financing", "valuation", "due diligence", "deal sourcing",
    "deal execution", "post-merger integration", "post-acquisition integration",
    "divestment", "carve-out", "spin-off", "initial public offering", "ipo", "secondary offering",
    "bond issuance", "debt issuance", "syndicated loans", "leveraged buyouts", "lbo",
    "management buyouts", "mbo", "restructuring", "bankruptcy", "insolvency", "distressed assets",
    "workout", "debt recovery", "credit analysis", "loan origination", "loan servicing",
    "collection", "asset-backed securities", "mortgage-backed securities", "collateralized debt obligations",
    "cdo", "securitization", "financial engineering", "quantitative finance", "model validation",
    "stress testing", "scenario analysis", "value-at-risk", "var", "expected shortfall", "es",
    "monte carlo simulation", "time series analysis", "regression analysis", "machine learning in finance",
    "algorithmic trading", "high-frequency trading", "fintech solutions", "blockchain in finance",
    "robo-advisors", "peer-to-peer lending", "p2p", "digital payments", "mobile payments",
    "cryptocurrency trading", "custody services", "exchange platforms", "defi", "decentralized finance",
    "nft marketplaces", "tokenomics", "smart contract auditing", "cybersecurity in finance",
    "fraud detection", "anti-money laundering", "aml", "know your customer", "kyc",
    "sanctions screening", "transaction monitoring", "regulatory reporting", "basel", "dodd-frank",
    "solvency", "mifid", "emirs", "fdic", "fca", "sec", "finra", "esma", "ecb", "federal reserve",
    "monetary authority of singapore", "mas", "hong kong monetary authority", "hkma",
    "people's bank of china", "pboc", "rbi", "bank of england", "boe", "bank of japan", "boj",
    "bank of canada", "boc", "european central bank", "ecb", "swiss national bank", "snb",
    "australian treasury", "reserve bank of australia", "rba", "new zealand treasury",
    "reserve bank of new zealand", "rbnz", "securities and exchange board of india", "sebi",
    "china securities regulatory commission", "csrc", "japan financial services agency", "jfsa",
    "south korea financial services commission", "fsc", "brazil central bank", "bcb",
    "mexico central bank", "banxico", "south africa reserve bank", "sarb", "nigeria central bank",
    "cbn", "uae central bank", "cbuae", "saudi central bank", "sacc", "qatar central bank", "qcb",
    "egypt central bank", "cbe", "turkey central bank", "cbrt", "russia central bank", "cbrf",
    "poland central bank", "nbp", "czech national bank", "cnb", "hungary central bank", "mnb",
    "romania national bank", "bnrx", "bulgaria national bank", "bnb", "greece central bank", "bog",
    "portugal central bank", "bp", "ireland central bank", "cbi", "belgium national bank", "nbb",
    "netherlands central bank", "dnb", "austria central bank", "oenb", "finland central bank", "bof",
    "sweden central bank", "riksbank", "norway central bank", "norges bank", "denmark central bank",
    "dnb", "iceland central bank", "cbis", "greenland central bank", "gl", "faroes central bank",
    "fb", "malta central bank", "cbm", "cyprus central bank", "cbc", "luxembourg central bank", "bcl",
    "liechtenstein central bank", "fma", "monaco central bank", "bdm", "san marino central bank",
    "bcs", "vatican city central bank", "vcb", "andorra central bank", "anc", "kosovo central bank",
    "cbk", "montenegro central bank", "cbcg", "albania central bank", "bsa", "macedonia central bank",
    "nbrm", "serbia national bank", "nbs", "bosnia and herzegovina central bank", "cbbh",
    "croatia national bank", "hnb", "slovenia central bank", "bsi", "slovakia central bank", "nbs",
    "estonia central bank", "eesti pank", "latvia central bank", "banka latvijas", "lithuania central bank",
    "lietuvos bankas", "belarus central bank", "nbrb", "ukraine national bank", "nbu",
    "moldova national bank", "bnm", "georgia national bank", "nbg", "armenia central bank", "cba",
    "azerbaijan central bank", "cbar", "kazakhstan national bank", "nbk", "kyrgyzstan national bank",
    "nbkr", "uzbekistan central bank", "cbu", "turkmenistan central bank", "cbt", "tajikistan national bank",
    "nbt", "afghanistan central bank", "dab", "iran central bank", "cbi", "iraq central bank", "cbi",
    "syria central bank", "cbs", "lebanon central bank", "bdl", "jordan central bank", "cbj",
    "israel central bank", "boi", "palestine monetary authority", "pma", "egypt central bank", "cbe",
    "libya central bank", "cbl", "tunisia central bank", "bct", "algeria central bank", "baa",
    "morocco central bank", "bam", "mauritania central bank", "bcm", "senegal central bank", "bceao",
    "mali central bank", "bceao", "niger central bank", "bceao", "burkina faso central bank", "bceao",
    "togo central bank", "bceao", "benin central bank", "bceao", "ivory coast central bank", "bceao",
    "guinea-bissau central bank", "bceao", "cape verde central bank", "bccv", "gambia central bank",
    "cbg", "guinea central bank", "bcrg", "sierra leone central bank", "bsl", "liberia central bank",
    "cbl", "ghana central bank", "bog", "nigeria central bank", "cbn", "cameroon central bank", "beac",
    "central african republic central bank", "beac", "chad central bank", "beac", "congo republic central bank",
    "beac", "equatorial guinea central bank", "beac", "gabon central bank", "beac",
    "democratic republic of congo central bank", "bcc", "burundi central bank", "brb", "rwanda central bank",
    "bnk", "uganda central bank", "bou", "kenya central bank", "cbk", "tanzania central bank", "bot",
    "zambia central bank", "boz", "malawi central bank", "rbm", "mozambique central bank", "bdm",
    "zimbabwe central bank", "rbz", "botswana central bank", "bob", "namibia central bank", "bon",
    "south africa reserve bank", "sarb", "lesotho central bank", "cbl", "eswatini central bank", "cbs",
    "angola national bank", "bna", "sao tome and principe central bank", "bcstp", "comoros central bank",
    "bcc", "madagascar central bank", "bfm", "mauritius central bank", "bom", "seychelles central bank",
    "cbs", "djibouti central bank", "cbd", "eritrea central bank", "boe", "ethiopia central bank",
    "nbe", "somalia central bank", "cbs", "sudan central bank", "cbs", "south sudan central bank",
    "cbss", "cuba central bank", "bcc", "dominican republic central bank", "bancentral",
    "haiti central bank", "brh", "jamaica central bank", "boj", "trinidad and tobago central bank",
    "cbtt", "barbados central bank", "cbb", "bahamas central bank", "cbb", "guyana central bank",
    "bog", "suriname central bank", "cbs", "french guiana central bank", "iedom",
    "guadeloupe central bank", "iedom", "martinique central bank", "iedom", "saint pierre and miquelon central bank",
    "iedom", "saint barthélemy central bank", "iedom", "saint martin central bank", "iedom",
    "aruba central bank", "cba", "curaçao and sint maarten central bank", "cbcsm",
    "bolivia central bank", "bcb", "colombia central bank", "bancorep", "ecuador central bank",
    "bce", "peru central bank", "bcrp", "venezuela central bank", "bcv", "argentina central bank",
    "bcra", "brazil central bank", "bcb", "chile central bank", "bcch", "paraguay central bank",
    "bcp", "uruguay central bank", "bcu", "australia reserve bank", "rba", "new zealand reserve bank",
    "rbnz", "fiji reserve bank", "rbf", "papua new guinea central bank", "bpng",
    "solomon islands central bank", "cbsi", "vanuatu reserve bank", "rbv", "new caledonia central bank",
    "ieb", "french polynesia central bank", "iep", "samoa central bank", "cbs", "tonga reserve bank",
    "rbt", "tuvalu central bank", "cbt", "kiribati central bank", "cbsk", "nauru central bank",
    "cbn", "marshall islands central bank", "cbi", "micronesia central bank", "cbm",
    "palau central bank", "cbp", "northern mariana islands central bank", "cbni", "guam central bank",
    "cbg", "american samoa central bank", "cbas", "wallis and futuna central bank", "ieb",
    "singapore monetary authority", "mas", "malaysia central bank", "bnm", "indonesia central bank",
    "bi", "philippines central bank", "bsp", "thailand central bank", "bot", "vietnam central bank",
    "sbv", "cambodia national bank", "nbc", "laos central bank", "bol", "myanmar central bank",
    "cbm", "brunei central bank", "ambd", "east timor central bank", "bctl", "bangladesh central bank",
    "bb", "pakistan central bank", "sbp", "sri lanka central bank", "cbsl", "nepal central bank",
    "nrb", "bhutan central bank", "rbn", "maldives central bank", "mma", "mongolia central bank",
    "bom", "taiwan central bank", "cbc", "hong kong monetary authority", "hkma", "macau monetary authority",
    "amma", "south korea central bank", "bok", "north korea central bank", "cbprk", "japan central bank",
    "boj", "china central bank", "pboc", "mongolia central bank", "bom", "myanmar central bank", "cbm",
    "laos central bank", "bol", "cambodia national bank", "nbc", "vietnam central bank", "sbv",
    "thailand central bank", "bot", "philippines central bank", "bsp", "indonesia central bank", "bi",
    "malaysia central bank", "bnm", "singapore monetary authority", "mas", "brunei central bank", "ambd",
    "east timor central bank", "bctl", "bangladesh central bank", "bb", "pakistan central bank", "sbp",
    "sri lanka central bank", "cbsl", "nepal central bank", "nrb", "bhutan central bank", "rbn",
    "maldives central bank", "mma", "mongolia central bank", "bom", "taiwan central bank", "cbc",
    "hong kong monetary authority", "hkma", "macau monetary authority", "amma", "south korea central bank",
    "bok", "north korea central bank", "cbprk", "japan central bank", "boj", "china central bank", "pboc"
])
STOP_WORDS = NLTK_STOP_WORDS.union(CUSTOM_STOP_WORDS)


# --- Helpers ---
def clean_text(text):
    """Cleans text by removing newlines, extra spaces, and non-ASCII characters."""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip().lower()

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return ''.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        return f"[ERROR] {str(e)}"

def extract_years_of_experience(text):
    """Extracts years of experience from a given text by parsing date ranges or keywords."""
    text = text.lower()
    total_months = 0
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
    """Extracts an email address from the given text."""
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def extract_name(text):
    """
    Attempts to extract a name from the first few lines of the resume text.
    This is a heuristic and might not be perfect for all resume formats.
    """
    lines = text.strip().split('\n')
    if not lines:
        return None

    potential_name_lines = []
    for line in lines[:3]:
        line = line.strip()
        if not re.search(r'[@\d\.\-]', line) and len(line.split()) <= 4 and (line.isupper() or (line and line[0].isupper() and all(word[0].isupper() or not word.isalpha() for word in line.split()))):
            potential_name_lines.append(line)

    if potential_name_lines:
        name = max(potential_name_lines, key=len)
        name = re.sub(r'summary|education|experience|skills|projects|certifications', '', name, flags=re.IGNORECASE).strip()
        if name:
            return name.title()
    return None

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

# --- AI Suggestion Function (Now uses T5 for narrative, ML for score) ---
@st.cache_data(show_spinner="Generating AI Suggestion...")
def generate_ai_suggestion(candidate_name, score, years_exp, semantic_similarity, jd_text, resume_text, matched_keywords, missing_skills):
    """
    Generates a comprehensive AI suggestion combining T5 summary with rule-based assessment,
    including a section on why to hire and company growth.
    """
    overall_fit_phrase = ""
    recommendation_phrase = ""
    strengths = []
    gaps = []
    growth_potential = []

    # Define thresholds
    HIGH_FIT_SCORE = 85
    MODERATE_FIT_SCORE = 65
    HIGH_SEMANTIC = 0.75
    MODERATE_SEMANTIC = 0.40
    HIGH_EXP = 4
    MODERATE_EXP = 2

    # Overall Fit and Recommendation based on scores
    if score >= HIGH_FIT_SCORE and years_exp >= HIGH_EXP and semantic_similarity >= HIGH_SEMANTIC:
        overall_fit_phrase = "Exceptional Match"
        recommendation_phrase = "Strongly Recommended for Interview"
        strengths.append("Their profile demonstrates an **outstanding conceptual and practical alignment** with the job description.")
        strengths.append(f"They possess **extensive relevant experience** ({years_exp:.1f} years), aligning perfectly with the role's demands.")
        growth_potential.append(f"Hiring {candidate_name} would directly accelerate our team's objectives by leveraging their deep expertise in **{' and '.join(matched_keywords[:3]) if matched_keywords else 'key areas'}**.")
        growth_potential.append("Their proven ability in highly relevant fields suggests they can quickly contribute to strategic initiatives and mentor junior staff, fostering internal growth.")

    elif score >= MODERATE_FIT_SCORE and years_exp >= MODERATE_EXP and semantic_similarity >= MODERATE_SEMANTIC:
        overall_fit_phrase = "Good Fit"
        recommendation_phrase = "Recommended for Interview"
        strengths.append("The candidate shows **good overall alignment** with the role, both conceptually and through their experience.")
        if years_exp < HIGH_EXP:
            gaps.append(f"Experience ({years_exp:.1f} years) is slightly below the ideal, suggesting a need to probe depth in specific areas.")
        if semantic_similarity < HIGH_SEMANTIC:
            gaps.append("Their conceptual alignment with the role is fair; consider probing their approach to complex scenarios outlined in the JD.")
        growth_potential.append(f"{candidate_name}'s solid foundation in **{' and '.join(matched_keywords[:2]) if matched_keywords else 'relevant skills'}** indicates they can be a reliable contributor, adding capacity and driving project completion.")
        growth_potential.append("With targeted development, they have the potential to grow into more senior responsibilities, strengthening our long-term team capabilities.")
    else:
        overall_fit_phrase = "Lower Fit"
        recommendation_phrase = "Consider for Further Review / Likely Decline"
        gaps.append("Their overall profile indicates **significant discrepancies** with the job requirements, suggesting a lower overall fit.")
        growth_potential.append(f"{candidate_name}'s current profile may require significant onboarding or skill development in **{' and '.join(missing_skills[:3]) if missing_skills else 'critical areas'}**, which might limit immediate impact on company growth.")

    if years_exp < MODERATE_EXP:
        gaps.append(f"Experience ({years_exp:.1f} years) is notably limited for this role.")
    if semantic_similarity < MODERATE_SEMANTIC:
        gaps.append("A **conceptual gap** exists between their profile and the job description, implying a potential mismatch in understanding or approach.")
    if missing_skills:
        gaps.append(f"Key skills explicitly mentioned in the JD such as **{' and '.join(missing_skills[:3])}** appear to be less prominent in their resume.")
    if matched_keywords:
        strengths.append(f"Strong alignment with keywords including **{' and '.join(matched_keywords[:3])}**.")


    # Generate T5 summary for the resume
    t5_resume_summary = generate_summary_with_t5(resume_text)

    # Combine all parts into the final suggestion
    summary_parts = [f"**Overall Fit:** {overall_fit_phrase}.",]
    if strengths:
        summary_parts.append(f"**Strengths:** {' '.join(strengths)}")
    if gaps:
        summary_parts.append(f"**Areas for Development:** {' '.join(gaps)}")
    summary_parts.append(f"**Resume Summary (T5):** {t5_resume_summary}")
    summary_parts.append(f"**Why Hire & Company Growth:** {' '.join(growth_potential)}")
    summary_parts.append(f"**Recommendation:** {recommendation_phrase}.")


    return " ".join(summary_parts)


def semantic_score(resume_text, jd_text, years_exp):
    """
    Calculates a semantic score using an ML model and provides additional details.
    Falls back to smart_score if the ML model is not loaded or prediction fails.
    Applies STOP_WORDS filtering for keyword analysis (internally, not for display).
    """
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    score = 0.0
    feedback = "Initial assessment."
    semantic_similarity = 0.0

    if ml_model is None or model is None:
        st.warning("ML models not loaded. Providing basic score and generic feedback.")
        resume_words = {word for word in re.findall(r'\b\w+\b', resume_clean) if word not in STOP_WORDS}
        jd_words = {word for word in re.findall(r'\b\w+\b', jd_clean) if word not in STOP_WORDS}
        
        overlap_count = len(resume_words & jd_words)
        total_jd_words = len(jd_words)
        
        basic_score = (overlap_count / total_jd_words) * 70 if total_jd_words > 0 else 0
        basic_score += min(years_exp * 5, 30) # Add up to 30 for experience
        score = round(min(basic_score, 100), 2)
        
        feedback = "Due to missing ML models, a detailed AI suggestion cannot be provided. Basic score derived from keyword overlap. Manual review is highly recommended."
        
        return score, feedback, 0.0 # Return 0 for semantic similarity if ML not available


    try:
        jd_embed = model.encode(jd_clean) # This gives 384 features for all-MiniLM-L6-v2
        resume_embed = model.encode(resume_clean) # This gives 384 features for all-MiniLM-L6-v2

        semantic_similarity = cosine_similarity(jd_embed.reshape(1, -1), resume_embed.reshape(1, -1))[0][0]
        semantic_similarity = float(np.clip(semantic_similarity, 0, 1))

        resume_words_filtered = {word for word in re.findall(r'\b\w+\b', resume_clean) if word not in STOP_WORDS}
        jd_words_filtered = {word for word in re.findall(r'\b\w+\b', jd_clean) if word not in STOP_WORDS}
        keyword_overlap_count = len(resume_words_filtered & jd_words_filtered)
        
        years_exp_for_model = float(years_exp) if years_exp is not None else 0.0

        # Calculate jd_coverage_percentage before creating features for the model
        if len(jd_words_filtered) > 0:
            jd_coverage_percentage = (keyword_overlap_count / len(jd_words_filtered)) * 100
        else:
            jd_coverage_percentage = 0.0

        # Construct the 772-feature vector
        # 384 (JD embed) + 384 (Resume embed) + 1 (years_exp) + 1 (keyword_overlap) + 1 (semantic_similarity) + 1 (jd_coverage_percentage) = 772
        features = np.concatenate([
            jd_embed,
            resume_embed,
            [years_exp_for_model],
            [keyword_overlap_count],
            [semantic_similarity],
            [jd_coverage_percentage]
        ])

        predicted_score = ml_model.predict([features])[0]

        # Blending logic (can be adjusted)
        blended_score = (predicted_score * 0.6) + \
                        (jd_coverage_percentage * 0.1) + \
                        (semantic_similarity * 100 * 0.3)

        if semantic_similarity > 0.7 and years_exp >= 3:
            blended_score += 5

        score = float(np.clip(blended_score, 0, 100))
        
        return round(score, 2), "AI suggestion will be generated...", round(semantic_similarity, 2)


    except Exception as e:
        st.warning(f"Error during semantic scoring, falling back to basic: {e}")
        resume_words = {word for word in re.findall(r'\b\w+\b', resume_clean) if word not in STOP_WORDS}
        jd_words = {word for word in re.findall(r'\b\w+\b', jd_clean) if word not in STOP_WORDS}
        
        overlap_count = len(resume_words & jd_words)
        total_jd_words = len(jd_words)
        
        basic_score = (overlap_count / total_jd_words) * 70 if total_jd_words > 0 else 0
        basic_score += min(years_exp * 5, 30) # Add up to 30 for experience
        score = round(min(basic_score, 100), 2)

        feedback = "Due to an error in core AI model, a detailed AI suggestion cannot be provided. Basic score derived. Manual review is highly recommended."

        return score, feedback, 0.0 # Return 0 for semantic similarity on fallback


# --- Email Generation Function ---
def create_mailto_link(recipient_email, candidate_name, job_title="Job Opportunity", sender_name="Recruiting Team"):
    """
    Generates a mailto: link with pre-filled subject and body for inviting a candidate.
    """
    subject = urllib.parse.quote(f"Invitation for Interview - {job_title} - {candidate_name}")
    body = urllib.parse.quote(f"""Dear {candidate_name},

We were very impressed with your profile and would like to invite you for an interview for the {job_title} position.

Best regards,

The {sender_name}""")
    return f"mailto:{recipient_email}?subject={subject}&body={body}"

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
        jd_path = job_roles[jd_option]
        if jd_path and os.path.exists(jd_path):
            with open(jd_path, "r", encoding="utf-8") as f:
                jd_text = f.read()
    
    if jd_text:
        with st.expander("📝 View Loaded Job Description"):
            st.text_area("Job Description Content", jd_text, height=200, disabled=True, label_visibility="collapsed")
        
        # Add T5 Job Description Summary
        if t5_model and t5_tokenizer:
            with st.expander("T5 Job Description Summary"):
                jd_summary_text = generate_summary_with_t5(jd_text)
                st.write(jd_summary_text)
        else:
            st.warning("T5 model not loaded, JD summarization unavailable.")


with col2:
    cutoff = st.slider("📈 **Minimum Score Cutoff (%)**", 0, 100, 75, help="Candidates scoring below this percentage will be flagged for closer review or considered less suitable.")
    min_experience = st.slider("💼 **Minimum Experience Required (Years)**", 0, 15, 2, help="Candidates with less than this experience will be noted.")
    st.markdown("---")
    st.info("Once criteria are set, upload resumes below to begin screening.")

resume_files = st.file_uploader("📄 **Upload Resumes (PDF)**", type="pdf", accept_multiple_files=True, help="Upload one or more PDF resumes for screening.")

df = pd.DataFrame()

if jd_text and resume_files:
    # --- Job Description Keyword Cloud ---
    st.markdown("---")
    st.markdown("## ☁️ Job Description Keyword Cloud")
    st.caption("Visualizing the most frequent and important keywords from the Job Description (common words filtered out).")
    jd_words_for_cloud = " ".join([word for word in re.findall(r'\b\w+\b', clean_text(jd_text)) if word not in STOP_WORDS])
    if jd_words_for_cloud:
        wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(jd_words_for_cloud)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free up memory
    else:
        st.info("No significant keywords to display for the Job Description. Please ensure your JD has sufficient content.")
    st.markdown("---")

    results = []
    resume_text_map = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(resume_files):
        status_text.text(f"Processing {file.name} ({i+1}/{len(resume_files)})...")
        progress_bar.progress((i + 1) / len(resume_files))

        text = extract_text_from_pdf(file)
        if text.startswith("[ERROR]"):
            st.error(f"Failed to process {file.name}: {text.replace('[ERROR] ', '')}")
            continue

        exp = extract_years_of_experience(text)
        email = extract_email(text)
        candidate_name = extract_name(text) or file.name.replace('.pdf', '').replace('_', ' ').title()

        # Calculate Matched Keywords and Missing Skills
        resume_clean_for_keywords = clean_text(text)
        jd_clean_for_keywords = clean_text(jd_text)

        # Filter out stop words for keyword analysis
        resume_words_set = {word for word in re.findall(r'\b\w+\b', resume_clean_for_keywords) if word not in STOP_WORDS}
        jd_words_set = {word for word in re.findall(r'\b\w+\b', jd_clean_for_keywords) if word not in STOP_WORDS}

        matched_keywords_list = sorted(list(resume_words_set.intersection(jd_words_set)))
        
        # Corrected: missing_skills should be JD words not in resume words
        missing_skills_list = sorted(list(jd_words_set.difference(resume_words_set)))
        
        score, _, semantic_similarity = semantic_score(text, jd_text, exp)
        
        detailed_ai_suggestion = generate_ai_suggestion(
            candidate_name=candidate_name,
            score=score,
            years_exp=exp,
            semantic_similarity=semantic_similarity,
            jd_text=jd_text,
            resume_text=text,
            matched_keywords=matched_keywords_list,
            missing_skills=missing_skills_list
        )

        results.append({
            "File Name": file.name,
            "Candidate Name": candidate_name,
            "Score (%)": score,
            "Years Experience": exp,
            "Email": email or "Not Found",
            "AI Suggestion": detailed_ai_suggestion,
            "Matched Keywords": ", ".join(matched_keywords_list),
            "Missing Skills": ", ".join(missing_skills_list),
            "Semantic Similarity": semantic_similarity,
            "Resume Raw Text": text
        })
        resume_text_map[file.name] = text
    
    progress_bar.empty()
    status_text.empty()


    df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False).reset_index(drop=True)

    st.session_state['screening_results'] = results
    
    # Save results to CSV for analytics.py to use
    df.to_csv("results.csv", index=False)


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

    # --- TOP CANDIDATE AI RECOMMENDATION (Game Changer Feature) ---
    st.markdown("## 👑 Top Candidate AI Recommendation")
    st.caption("A concise, AI-powered assessment for the most suitable candidate, focusing on their potential impact.")
    
    if not df.empty:
        top_candidate = df.iloc[0]
        st.markdown(f"### **{top_candidate['Candidate Name']}**")
        st.markdown(f"**Score:** {top_candidate['Score (%)']:.2f}% | **Experience:** {top_candidate['Years Experience']:.1f} years | **Semantic Similarity:** {top_candidate['Semantic Similarity']:.2f}")
        st.markdown(f"**AI Assessment:** {top_candidate['AI Suggestion']}") # This now includes the growth section
        
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


    # === AI Recommendation for Shortlisted Candidates (Streamlined) ===
    st.markdown("## 🌟 Shortlisted Candidates Overview")
    st.caption("Candidates meeting your score and experience criteria, with their AI-generated summaries.")

    shortlisted_candidates = df[(df['Score (%)'] >= cutoff) & (df['Years Experience'] >= min_experience)]

    if not shortlisted_candidates.empty:
        st.success(f"**{len(shortlisted_candidates)}** candidate(s) meet your specified criteria (Score ≥ {cutoff}%, Experience ≥ {min_experience} years).")
        
        display_shortlisted_summary_cols = [
            'Candidate Name',
            'Score (%)',
            'Years Experience',
            'Semantic Similarity',
            'Email',
            'AI Suggestion' # Individual AI suggestion for each
        ]
        
        st.dataframe(
            shortlisted_candidates[display_shortlisted_summary_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score (%)": st.column_config.ProgressColumn(
                    "Score (%)",
                    help="Matching score against job requirements",
                    format="%f",
                    min_value=0,
                    max_value=100,
                ),
                "Years Experience": st.column_config.NumberColumn(
                    "Years Experience",
                    help="Total years of professional experience",
                    format="%.1f years",
                ),
                "Semantic Similarity": st.column_config.NumberColumn(
                    "Semantic Similarity",
                    help="Conceptual similarity between JD and Resume (higher is better)",
                    format="%.2f",
                    min_value=0,
                    max_value=1
                ),
                "AI Suggestion": st.column_config.Column(
                    "AI Suggestion",
                    help="AI's concise overall assessment and recommendation",
                    width="large" # Make it wider to show more text
                )
            }
        )
        st.info("For individual detailed AI assessments and action steps, please refer to the table above or the Analytics Dashboard.")

    else:
        st.warning("No candidates met the defined screening criteria (score cutoff and minimum experience). You might consider adjusting the sliders or reviewing the uploaded resumes/JD.")

    st.markdown("---")

    # Add a 'Tag' column for quick categorization
    df['Tag'] = df.apply(lambda row: "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3 else (
        "✅ Good Fit" if row['Score (%)'] >= 75 else "⚠️ Needs Review"), axis=1)

    st.markdown("## 📋 Comprehensive Candidate Results Table")
    st.caption("Full details for all processed resumes, including individual AI suggestions. **For deep dive analytics and keyword breakdowns, refer to the Analytics Dashboard.**")
    
    comprehensive_cols = [
        'Candidate Name',
        'Score (%)',
        'Years Experience',
        'Semantic Similarity',
        'Tag',
        'Email',
        'AI Suggestion', # Ensure this is included
        'Matched Keywords',
        'Missing Skills',
    ]
    
    existing_comprehensive_cols = [col for col in comprehensive_cols if col in df.columns]

    st.dataframe(
        df[existing_comprehensive_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score (%)": st.column_config.ProgressColumn(
                "Score (%)",
                help="Matching score against job requirements",
                format="%f",
                min_value=0,
                max_value=100,
            ),
            "Years Experience": st.column_config.NumberColumn(
                "Years Experience",
                help="Total years of professional experience",
                format="%.1f years",
            ),
            "Semantic Similarity": st.column_config.NumberColumn(
                "Semantic Similarity",
                help="Conceptual similarity between JD and Resume (higher is better)",
                format="%.2f",
                min_value=0,
                max_value=1
            ),
            "AI Suggestion": st.column_config.Column(
                "AI Suggestion",
                help="AI's concise overall assessment and recommendation",
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
else:
    st.info("Upload a Job Description and Resumes to begin screening.")

# --- About Section ---
st.sidebar.title("About ScreenerPro")
st.sidebar.info(
    "ScreenerPro is an AI-powered application designed to streamline the resume screening "
    "process. It leverages a custom-trained Machine Learning model, a Sentence Transformer for "
    "semantic understanding, and a fine-tuned T5 model for insightful AI suggestions and summarization.\n\n"
    "Upload job descriptions and resumes, and let AI assist you in identifying the best-fit candidates!"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Developed by [Manav Nagpal](https://www.linkedin.com/in/manav-nagpal-b03a74211/)"
)
