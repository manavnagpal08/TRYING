import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
import joblib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
import collections
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse # For encoding mailto links

# Import T5 specific libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- NO NLTK IMPORTS OR DOWNLOADS HERE ---
# All NLTK related code (imports, downloads, lemmatizer, get_wordnet_pos) are removed.
# Skill extraction is now purely regex-based.

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

# --- MASTER SKILLS DICTIONARY ---
# This is your comprehensive list of all potential skills.
# Please paste your skills here when ready.
ALL_SKILLS_MASTER = {
        # Product & Project Management
    "Product Strategy", "Roadmap Development", "Agile Methodologies", "Scrum", "Kanban", "Jira", "Trello",
    "Feature Prioritization", "OKRs", "KPIs", "Stakeholder Management", "A/B Testing", "User Stories", "Epics",
    "Product Lifecycle", "Sprint Planning", "Project Charter", "Gantt Charts", "MVP", "Backlog Grooming",
    "Risk Management", "Change Management", "Program Management", "Portfolio Management", "PMP", "CSM",

    # Software Development & Engineering
    "Python", "Java", "JavaScript", "C++", "C#", "Go", "Ruby", "PHP", "Swift", "Kotlin", "TypeScript",
    "HTML5", "CSS3", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring Boot", "Express.js",
    "Git", "GitHub", "GitLab", "Bitbucket", "REST APIs", "GraphQL", "Microservices", "System Design",
    "Unit Testing", "Integration Testing", "End-to-End Testing", "Test Automation", "CI/CD", "Docker", "Kubernetes",
    "Serverless", "AWS Lambda", "Azure Functions", "Google Cloud Functions", "WebSockets", "Kafka", "RabbitMQ",
    "Redis", "SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Cassandra", "Elasticsearch", "Neo4j",
    "Data Structures", "Algorithms", "Object-Oriented Programming", "Functional Programming", "Bash Scripting",
    "Shell Scripting", "DevOps", "DevSecOps", "SRE", "CloudFormation", "Terraform", "Ansible", "Puppet", "Chef",
    "Jenkins", "CircleCI", "GitHub Actions", "Azure DevOps", "Jira", "Confluence", "Swagger", "OpenAPI",

    # Data Science & AI/ML
    "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "Reinforcement Learning",
    "Scikit-learn", "TensorFlow", "PyTorch", "Keras", "XGBoost", "LightGBM", "Data Cleaning", "Feature Engineering",
    "Model Evaluation", "Statistical Modeling", "Time Series Analysis", "Predictive Modeling", "Clustering",
    "Classification", "Regression", "Neural Networks", "Convolutional Networks", "Recurrent Networks",
    "Transformers", "LLMs", "Prompt Engineering", "Generative AI", "MLOps", "Data Munging", "A/B Testing",
    "Experiment Design", "Hypothesis Testing", "Bayesian Statistics", "Causal Inference", "Graph Neural Networks",

    # Data Analytics & BI
    "SQL", "Python (Pandas, NumPy)", "R", "Excel (Advanced)", "Tableau", "Power BI", "Looker", "Qlik Sense",
    "Google Data Studio", "Dax", "M Query", "ETL", "ELT", "Data Warehousing", "Data Lake", "Data Modeling",
    "Business Intelligence", "Data Visualization", "Dashboarding", "Report Generation", "Google Analytics",
    "BigQuery", "Snowflake", "Redshift", "Data Governance", "Data Quality", "Statistical Analysis",
    "Requirements Gathering", "Data Storytelling",

    # Cloud & Infrastructure
    "AWS", "Azure", "Google Cloud Platform", "GCP", "Cloud Architecture", "Hybrid Cloud", "Multi-Cloud",
    "Virtualization", "VMware", "Hyper-V", "Linux Administration", "Windows Server", "Networking", "TCP/IP",
    "DNS", "VPN", "Firewalls", "Load Balancing", "CDN", "Monitoring", "Logging", "Alerting", "Prometheus",
    "Grafana", "Splunk", "ELK Stack", "Cloud Security", "IAM", "VPC", "Storage (S3, Blob, GCS)", "Databases (RDS, Azure SQL)",
    "Container Orchestration", "Infrastructure as Code", "IaC",

    # UI/UX & Design
    "Figma", "Adobe XD", "Sketch", "Photoshop", "Illustrator", "InDesign", "User Research", "Usability Testing",
    "Wireframing", "Prototyping", "UI Design", "UX Design", "Interaction Design", "Information Architecture",
    "Design Systems", "Accessibility", "Responsive Design", "User Flows", "Journey Mapping", "Design Thinking",
    "Visual Design", "Motion Graphics",

    # Marketing & Sales
    "Digital Marketing", "SEO", "SEM", "Content Marketing", "Email Marketing", "Social Media Marketing",
    "Google Ads", "Facebook Ads", "LinkedIn Ads", "Marketing Automation", "HubSpot", "Salesforce Marketing Cloud",
    "CRM", "Lead Generation", "Sales Strategy", "Negotiation", "Account Management", "Market Research",
    "Campaign Management", "Conversion Rate Optimization", "CRO", "Brand Management", "Public Relations",
    "Copywriting", "Content Creation", "Analytics (Google Analytics, SEMrush, Ahrefs)",

    # Finance & Accounting
    "Financial Modeling", "Valuation", "Financial Reporting", "GAAP", "IFRS", "Budgeting", "Forecasting",
    "Variance Analysis", "Auditing", "Taxation", "Accounts Payable", "Accounts Receivable", "Payroll",
    "QuickBooks", "SAP FICO", "Oracle Financials", "Cost Accounting", "Management Accounting", "Treasury Management",
    "Investment Analysis", "Risk Analysis", "Compliance (SOX, AML)",

    # Human Resources (HR)
    "Talent Acquisition", "Recruitment", "Onboarding", "Employee Relations", "HRIS (Workday, SuccessFactors)",
    "Compensation & Benefits", "Performance Management", "Workforce Planning", "HR Policies", "Labor Law",
    "Training & Development", "Diversity & Inclusion", "Conflict Resolution", "Employee Engagement",

    # Customer Service & Support
    "Customer Relationship Management", "CRM", "Zendesk", "ServiceNow", "Intercom", "Live Chat", "Ticketing Systems",
    "Issue Resolution", "Technical Support", "Customer Success", "Client Retention", "Communication Skills",

    # General Business & Soft Skills (often paired with technical skills)
    "Strategic Planning", "Business Development", "Vendor Management", "Process Improvement", "Operations Management",
    "Project Coordination", "Public Speaking", "Presentation Skills", "Cross-functional Collaboration",
    "Problem Solving", "Critical Thinking", "Analytical Skills", "Adaptability", "Time Management",
    "Organizational Skills", "Attention to Detail", "Leadership", "Mentorship", "Team Leadership",
    "Decision Making", "Negotiation", "Client Management", "Stakeholder Communication", "Active Listening",
    "Creativity", "Innovation", "Research", "Data Analysis", "Report Writing", "Documentation",
    "Microsoft Office Suite", "Google Workspace", "Slack", "Zoom", "Confluence", "SharePoint",
    "Cybersecurity", "Information Security", "Risk Assessment", "Compliance", "GDPR", "HIPAA", "ISO 27001",
    "Penetration Testing", "Vulnerability Management", "Incident Response", "Security Audits", "Forensics",
    "Threat Intelligence", "SIEM", "Firewall Management", "Endpoint Security", "Identity and Access Management",
    "IAM", "Cryptography", "Network Security", "Application Security", "Cloud Security",

    # Specific Certifications/Tools often treated as skills
    "PMP", "CSM", "AWS Certified", "Azure Certified", "GCP Certified", "CCNA", "CISSP", "CISM", "CompTIA Security+",
    "ITIL", "Lean Six Sigma", "CFA", "CPA", "SHRM-CP", "PHR", "CEH", "OSCP", "Splunk", "ServiceNow", "Salesforce",
    "Workday", "SAP", "Oracle", "Microsoft Dynamics", "NetSuite", "Adobe Creative Suite", "Canva", "Mailchimp",
    "Hootsuite", "Buffer", "SEMrush", "Ahrefs", "Moz", "Screaming Frog", "JMeter", "Postman", "SoapUI",
    "Git", "SVN", "Perforce", "Confluence", "Jira", "Asana", "Trello", "Monday.com", "Miro", "Lucidchart",
    "Visio", "MS Project", "Primavera", "AutoCAD", "SolidWorks", "MATLAB", "LabVIEW", "Simulink", "ANSYS",
    "CATIA", "NX", "Revit", "ArcGIS", "QGIS", "OpenCV", "NLTK", "SpaCy", "Gensim", "Hugging Face Transformers",
    "Docker Compose", "Helm", "Ansible Tower", "SaltStack", "Chef InSpec", "Terraform Cloud", "Vault",
    "Consul", "Nomad", "Prometheus", "Grafana", "Alertmanager", "Loki", "Tempo", "Jaeger", "Zipkin",
    "Fluentd", "Logstash", "Kibana", "Grafana Loki", "Datadog", "New Relic", "AppDynamics", "Dynatrace",
    "Nagios", "Zabbix", "Icinga", "PRTG", "SolarWinds", "Wireshark", "Nmap", "Metasploit", "Burp Suite",
    "OWASP ZAP", "Nessus", "Qualys", "Rapid7", "Tenable", "CrowdStrike", "SentinelOne", "Palo Alto Networks",
    "Fortinet", "Cisco Umbrella", "Okta", "Auth0", "Keycloak", "Ping Identity", "Active Directory",
    "LDAP", "OAuth", "JWT", "OpenID Connect", "SAML", "MFA", "SSO", "PKI", "TLS/SSL", "VPN", "IDS/IPS",
    "DLP", "CASB", "SOAR", "XDR", "EDR", "MDR", "GRC", "GDPR Compliance", "HIPAA Compliance", "PCI DSS Compliance",
    "ISO 27001 Compliance", "NIST Framework", "COBIT", "ITIL Framework", "Scrum Master", "Product Owner",
    "Agile Coach", "Release Management", "Change Control", "Configuration Management", "Asset Management",
    "Service Desk", "Incident Management", "Problem Management", "Change Management", "Release Management",
    "Service Level Agreements", "SLAs", "Operational Level Agreements", "OLAs", "Underpinning Contracts", "UCs",
    "Knowledge Management", "Continual Service Improvement", "CSI", "Service Catalog", "Service Portfolio",
    "Relationship Management", "Supplier Management", "Financial Management for IT Services",
    "Demand Management", "Capacity Management", "Availability Management", "Information Security Management",
    "Supplier Relationship Management", "Contract Management", "Procurement Management", "Quality Management",
    "Test Management", "Defect Management", "Requirements Management", "Scope Management", "Time Management",
    "Cost Management", "Quality Management", "Resource Management", "Communications Management",
    "Risk Management", "Procurement Management", "Stakeholder Management", "Integration Management",
    "Project Charter", "Project Plan", "Work Breakdown Structure", "WBS", "Gantt Chart", "Critical Path Method",
    "CPM", "Earned Value Management", "EVM", "PERT", "CPM", "Crashing", "Fast Tracking", "Resource Leveling",
    "Resource Smoothing", "Agile Planning", "Scrum Planning", "Kanban Planning", "Sprint Backlog",
    "Product Backlog", "User Story Mapping", "Relative Sizing", "Planning Poker", "Velocity", "Burndown Chart",
    "Burnup Chart", "Cumulative Flow Diagram", "CFD", "Value Stream Mapping", "VSM", "Lean Principles",
    "Six Sigma", "Kaizen", "Kanban", "Total Quality Management", "TQM", "Statistical Process Control", "SPC",
    "Control Charts", "Pareto Analysis", "Fishbone Diagram", "5 Whys", "FMEA", "Root Cause Analysis", "RCA",
    "Corrective Actions", "Preventive Actions", "CAPA", "Non-conformance Management", "Audit Management",
    "Document Control", "Record Keeping", "Training Management", "Calibration Management", "Supplier Quality Management",
    "Customer Satisfaction Measurement", "Net Promoter Score", "NPS", "Customer Effort Score", "CES",
    "Customer Satisfaction Score", "CSAT", "Voice of Customer", "VOC", "Complaint Handling", "Warranty Management",
    "Returns Management", "Service Contracts", "Service Agreements", "Maintenance Management", "Field Service Management",
    "Asset Management", "Enterprise Asset Management", "EAM", "Computerized Maintenance Management System", "CMMS",
    "Geographic Information Systems", "GIS", "GPS", "Remote Sensing", "Image Processing", "CAD", "CAM", "CAE",
    "FEA", "CFD", "PLM", "PDM", "ERP", "CRM", "SCM", "HRIS", "BI", "Analytics", "Data Science", "Machine Learning",
    "Deep Learning", "NLP", "Computer Vision", "AI", "Robotics", "Automation", "IoT", "Blockchain", "Cybersecurity",
    "Cloud Computing", "Big Data", "Data Warehousing", "ETL", "Data Modeling", "Data Governance", "Data Quality",
    "Data Migration", "Data Integration", "Data Virtualization", "Data Lakehouse", "Data Mesh", "Data Fabric",
    "Data Catalog", "Data Lineage", "Metadata Management", "Master Data Management", "MDM",
    "Customer Data Platform", "CDP", "Digital Twin", "Augmented Reality", "AR", "Virtual Reality", "VR",
    "Mixed Reality", "MR", "Extended Reality", "XR", "Game Development", "Unity", "Unreal Engine", "C# (Unity)",
    "C++ (Unreal Engine)", "Game Design", "Level Design", "Character Design", "Environment Design",
    "Animation (Game)", "Rigging", "Texturing", "Shading", "Lighting", "Rendering", "Game Physics",
    "Game AI", "Multiplayer Networking", "Game Monetization", "Game Analytics", "Playtesting",
    "Game Publishing", "Streaming (Gaming)", "Community Management (Gaming)",
    "Game Art", "Game Audio", "Sound Design (Game)", "Music Composition (Game)", "Voice Acting (Game)",
    "Narrative Design", "Storytelling (Game)", "Dialogue Writing", "World Building", "Lore Creation",
    "Game Scripting", "Modding", "Game Engine Development", "Graphics Programming", "Physics Programming",
    "AI Programming (Game)", "Network Programming (Game)", "Tools Programming (Game)", "UI Programming (Game)",
    "Shader Development", "VFX (Game)", "Technical Art", "Technical Animation", "Technical Design",
    "Build Engineering (Game)", "Release Engineering (Game)", "Live Operations (Game)", "Game Balancing",
    "Economy Design (Game)", "Progression Systems (Game)", "Retention Strategies (Game)", "Monetization Strategies (Game)",
    "User Acquisition (Game)", "Marketing (Game)", "PR (Game)", "Community Management (Game)",
    "Customer Support (Game)", "Localization (Game)", "Quality Assurance (Game)", "Game Testing",
    "Compliance (Game)", "Legal (Game)", "Finance (Game)", "HR (Game)", "Business Development (Game)",
    "Partnerships (Game)", "Licensing (Game)", "Brand Management (Game)", "IP Management (Game)",
    "Esports Event Management", "Esports Team Management", "Esports Coaching", "Esports Broadcasting",
    "Esports Sponsorship", "Esports Marketing", "Esports Analytics", "Esports Operations",
    "Esports Content Creation", "Esports Journalism", "Esports Law", "Esports Finance", "Esports HR",
    "Esports Business Development", "Esports Partnerships", "Esports Licensing", "Esports Brand Management",
    "Esports IP Management", "Esports Event Planning", "Esports Production", "Esports Broadcasting",
    "Esports Commentating", "Esports Analysis", "Esports Coaching", "Esports Training", "Esports Recruitment",
    "Esports Scouting", "Esports Player Management", "Esports Team Management", "Esports Organization Management",
    "Esports League Management", "Esports Tournament Management", "Esports Venue Management Software",
    "Esports Sponsorship Management Software", "Esports Marketing Automation Software",
    "Esports Content Management Systems", "Esports Social Media Management Tools",
    "Esports PR Tools", "Esports Brand Monitoring Tools", "Esports Community Management Software",
    "Esports Fan Engagement Platforms", "Esports Merchandise Management Software",
    "Esports Ticketing Platforms", "Esports Hospitality Management Software",
    "Esports Logistics Management Software", "Esports Security Management Software",
    "Esports Legal Management Software", "Esports Finance Management Software",
    "Esports HR Management Software", "Esports Business Operations Software",
    "Esports Data Analytics Software", "Esports Performance Analysis Software",
    "Esports Coaching Software", "Esports Training Platforms", "Esports Scouting Tools",
    "Esports Player Databases", "Esports Team Databases", "Esports Organization Databases",
    "Esports League Databases", "Esports Tournament Platforms", "Esports Venue Management Software",
    "Esports Sponsorship Management Software", "Esports Marketing Automation Software",
    "Esports Content Management Systems", "Esports Social Media Management Tools",
    "Esports PR Tools", "Esports Brand Monitoring Tools", "Esports Community Management Software",
    "Esports Fan Engagement Platforms", "Esports Merchandise Management Software",
    "Esports Ticketing Platforms", "Esports Hospitality Management Software",
    "Esports Logistics Management Software", "Esports Security Management Software",
    "Esports Legal Management Software", "Esports Finance Management Software",
    "Esports HR Management Software", "Esports Business Operations Software",
    "Esports Data Analytics Software", "Esports Performance Analysis Software",
    "Esports Coaching Software", "Esports Training Platforms", "Esports Scouting Tools",
    "Esports Player Databases", "Esports Team Databases", "Esports Organization Databases",
    "Esports League Databases", "Esports Tournament Platforms", "Esports Venue Management Software",
    "Esports Sponsorship Management Software", "Esports Marketing Automation Software",
    "Esports Content Management Systems", "Esports Social Media Management Tools",
    "Esports PR Tools", "Esports Brand Monitoring Tools", "Esports Community Management Software",
    "Esports Fan Engagement Platforms", "Esports Merchandise Management Software",
    "Esports Ticketing Platforms", "Esports Hospitality Management Software",
    "Esports Logistics Management Software", "Esports Security Management Software",
    "Esports Legal Management Software", "Esports Finance Management Software",
    "Esports HR Management Software", "Esports Business Operations Software",
    "Esports Data Analytics Software", "Esports Performance Analysis Software",
    "Esports Coaching Software", "Esports Training Platforms", "Esports Scouting Tools",
    "Esports Player Databases", "Esports Team Databases", "Esports Organization Databases",
    "Esports League Datab"
}


# Convert all_skills_master to a set for faster lookup and uniform case
# Also create a list of sorted skills (longest first) for multi-word matching
ALL_SKILLS_MASTER_SET = {skill.lower() for skill in ALL_SKILLS_MASTER}
SORTED_MASTER_SKILLS = sorted(list(ALL_SKILLS_MASTER_SET), key=len, reverse=True)


# --- Custom Stop Words List (Expanded and NLTK-free) ---
# This list is crucial for filtering out common words that are not skills.
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
    # General words that might appear in "skills" section but aren't actual skills (this list is extensive)
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
    "associate", "specialist", "coordinator", "assistant", "intern", "co-op", "trainee", "apprentice",
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
    "natural language processing", "nlp", "computer vision", "image processing",
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
    "reserve bank of new zealand", "rbnz", "fiji reserve bank", "rbf", "papua new guinea central bank",
    "bpng", "solomon islands central bank", "cbsi", "vanuatu reserve bank", "rbv", "new caledonia central bank",
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
    "boj", "china central bank", "pboc"
])


# --- Skill Extraction Function (NLTK-free) ---
def extract_skills_from_text(text):
    """
    Extracts relevant skills from a given text using multi-word matching and
    stop word filtering, without NLTK.
    """
    text_lower = text.lower()
    extracted_skills = set()
    
    # First, try to match multi-word skills (longest first)
    processed_text = text_lower # Use a mutable copy for marking out matches
    for skill_phrase in SORTED_MASTER_SKILLS:
        # Only consider multi-word skills or single-word skills not in CUSTOM_STOP_WORDS
        if ' ' in skill_phrase or skill_phrase not in CUSTOM_STOP_WORDS:
            pattern = r'\b' + re.escape(skill_phrase) + r'\b'
            if re.search(pattern, processed_text):
                extracted_skills.add(skill_phrase)
                # Replace matched skill with placeholders to avoid re-matching its components
                processed_text = re.sub(pattern, ' ' * len(skill_phrase), processed_text)

    # Now process the remaining text for single words
    single_words = re.findall(r'\b[a-z]+\b', processed_text) 

    for word in single_words:
        # Basic filtering: short or stop words
        if len(word) < 2 or word in CUSTOM_STOP_WORDS:
            continue
        
        # Check if the word is in our master skill list
        if word in ALL_SKILLS_MASTER_SET:
            extracted_skills.add(word)
            
    return list(extracted_skills)


# --- Core PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return ''.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        return f"[ERROR] {str(e)}"

# --- Word Cloud Generation ---
def generate_wordcloud(skills_list):
    """Generates and displays a word cloud from a list of skills."""
    if skills_list:
        skills_text = " ".join(skills_list)
        wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(skills_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        plt.close(plt.gcf()) # Close the figure to free up memory
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
@st.cache_data(show_spinner="Generating T5 Summary...")
def generate_summary_with_t5(text, max_length=150):
    """Summarizes a resume using the loaded T5 model."""
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

# --- Embedding Generation for ML Model ---
def get_job_description_embedding(job_description, model):
    """Generates a numerical embedding for a job description using SentenceTransformer."""
    if model:
        return model.encode(job_description)
    return None

# --- ML Match Prediction ---
# This function is now integrated into semantic_score, but kept for clarity if needed elsewhere.
def predict_match_ml(resume_embedding, job_description_embedding, ml_model):
    """
    Predicts the match probability between a resume and job description
    using a pre-trained machine learning model.
    """
    if ml_model and resume_embedding is not None and job_description_embedding is not None:
        # Concatenate embeddings to form the feature vector for the ML model
        combined_features = np.concatenate((resume_embedding, job_description_embedding)).reshape(1, -1)
        # Get probability of being a match (class 1)
        # This assumes ml_model has a predict_proba method (e.g., RandomForestClassifier, XGBClassifier)
        prediction_proba = ml_model.predict_proba(combined_features)[0][1] 
        return prediction_proba
    return 0.0 # Return 0 if models/embeddings are not available


# --- Helpers for Resume Parsing (from screener (4).py) ---
def clean_text_for_parsing(text):
    """Cleans text by removing newlines, extra spaces, and non-ASCII characters for general parsing."""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip().lower()

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
    # Check first few lines for potential name
    for line in lines[:3]:
        line = line.strip()
        # Filter out lines that look like emails, numbers, or too long
        if not re.search(r'[@\d\.\-]', line) and len(line.split()) <= 4 and (line.isupper() or (line and line[0].isupper() and all(word[0].isupper() or not word.isalpha() for word in line.split()))):
            potential_name_lines.append(line)

    if potential_name_lines:
        # Choose the longest potential name as it's often the full name
        name = max(potential_name_lines, key=len)
        # Remove common resume section headers if they were accidentally picked up
        name = re.sub(r'summary|education|experience|skills|projects|certifications', '', name, flags=re.IGNORECASE).strip()
        if name:
            return name.title() # Capitalize each word
    return None

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


def semantic_score(resume_text, jd_text, years_exp, candidate_name="Unknown Candidate"):
    """
    Calculates a semantic score using an ML model and provides additional details.
    Falls back to smart_score if the ML model is not loaded or prediction fails.
    Applies CUSTOM_STOP_WORDS filtering for keyword analysis (internally, not for display).
    """
    # Use the clean_text_for_parsing for consistent text cleaning for semantic score
    jd_clean = clean_text_for_parsing(jd_text)
    resume_clean = clean_text_for_parsing(resume_text)

    score = 0.0
    feedback = "Initial assessment." # This feedback is now replaced by generate_ai_suggestion
    semantic_similarity = 0.0

    if ml_model is None or model is None:
        st.warning("ML models not loaded. Providing basic score and generic feedback.")
        resume_words = {word for word in re.findall(r'\b\w+\b', resume_clean) if word not in CUSTOM_STOP_WORDS}
        jd_words = {word for word in re.findall(r'\b\w+\b', jd_clean) if word not in CUSTOM_STOP_WORDS}
        
        overlap_count = len(resume_words & jd_words)
        total_jd_words = len(jd_words)
        
        basic_score = (overlap_count / total_jd_words) * 70 if total_jd_words > 0 else 0
        basic_score += min(years_exp * 5, 30) # Add up to 30 for experience
        score = round(min(basic_score, 100), 2)
        
        # The feedback here is for the score, not the AI suggestion
        feedback = "Basic score derived from keyword overlap due to missing ML models."
        
        return score, feedback, 0.0 # Return 0 for semantic similarity if ML not available


    try:
        # Add checks for minimal text length before encoding
        MIN_TEXT_LENGTH_FOR_EMBEDDING = 50 # Arbitrary threshold, adjust if needed
        
        if len(jd_clean) < MIN_TEXT_LENGTH_FOR_EMBEDDING:
            st.warning(f"Job Description text is too short ({len(jd_clean)} chars) after cleaning. Cannot generate meaningful JD embedding for ML model.")
            # Fallback for JD if text is too short
            return 0.0, "JD text too short for ML model.", 0.0

        if len(resume_clean) < MIN_TEXT_LENGTH_FOR_EMBEDDING:
            st.warning(f"Resume text for {candidate_name} is too short ({len(resume_clean)} chars) after cleaning. Cannot generate meaningful resume embedding for ML model.")
            # Fallback for Resume if text is too short
            return 0.0, "Resume text too short for ML model.", 0.0

        jd_embed = model.encode(jd_clean) # This gives 384 features for all-MiniLM-L6-v2
        resume_embed = model.encode(resume_clean) # This gives 384 features for all-MiniLM-L6-v2

        # Ensure embeddings are 1D arrays of expected size
        if jd_embed.shape[0] != 384 or resume_embed.shape[0] != 384:
            st.error(f"Embedding dimension mismatch. Expected 384, got JD: {jd_embed.shape[0]}, Resume: {resume_embed.shape[0]}.")
            return 0.0, "Embedding dimension mismatch.", 0.0

        semantic_similarity = cosine_similarity(jd_embed.reshape(1, -1), resume_embed.reshape(1, -1))[0][0]
        semantic_similarity = float(np.clip(semantic_similarity, 0, 1))

        resume_words_filtered = {word for word in re.findall(r'\b\w+\b', resume_clean) if word not in CUSTOM_STOP_WORDS}
        jd_words_filtered = {word for word in re.findall(r'\b\w+\b', jd_clean) if word not in CUSTOM_STOP_WORDS}
        keyword_overlap_count = len(resume_words_filtered & jd_words_filtered)
        
        years_exp_for_model = float(years_exp) if years_exp is not None else 0.0

        # Calculate jd_coverage_percentage before creating features for the model
        if len(jd_words_filtered) > 0:
            jd_coverage_percentage = (keyword_overlap_count / len(jd_words_filtered)) * 100
        else:
            jd_coverage_percentage = 0.0

        # Construct the feature vector for the ML model
        # The ML model expects a specific number of features based on how it was trained.
        # [JD_EMBEDDING (384), RESUME_EMBEDDING (384), YEARS_EXP (1), KEYWORD_OVERLAP_COUNT (1), SEMANTIC_SIMILARITY (1), JD_COVERAGE_PERCENTAGE (1)]
        # Total features: 384 + 384 + 1 + 1 + 1 + 1 = 772 features
        features = np.concatenate([
            jd_embed,
            resume_embed,
            np.array([years_exp_for_model]), # Ensure these are NumPy arrays for concatenation
            np.array([keyword_overlap_count]),
            np.array([semantic_similarity]),
            np.array([jd_coverage_percentage])
        ])

        # Reshape to (1, -1) for single sample prediction
        features = features.reshape(1, -1)

        # Use predict_proba as ml_model is assumed to be a classifier
        predicted_proba = ml_model.predict_proba(features)[0][1] # Probability of class 1 (match)
        predicted_score = predicted_proba * 100 # Convert to percentage

        # Blending logic (can be adjusted)
        # This blending adds rule-based adjustments to the ML model's prediction
        blended_score = (predicted_score * 0.6) + \
                        (jd_coverage_percentage * 0.1) + \
                        (semantic_similarity * 100 * 0.3)

        # Small bonus for high semantic match and good experience
        if semantic_similarity > 0.7 and years_exp >= 3:
            blended_score += 5

        score = float(np.clip(blended_score, 0, 100)) # Ensure score is between 0 and 100
        
        return round(score, 2), "AI suggestion will be generated...", round(semantic_similarity, 2)


    except Exception as e:
        st.error(f"⚠️ Could not predict score with ML model for {candidate_name}. Ensure 'ml_screening_model.pkl' is a classifier with 'predict_proba' and that the feature input matches training. Error: {e}")
        # Fallback to basic keyword overlap score if ML model fails
        resume_words = {word for word in re.findall(r'\b\w+\b', resume_clean) if word not in CUSTOM_STOP_WORDS}
        jd_words = {word for word in re.findall(r'\b\w+\b', jd_clean) if word not in CUSTOM_STOP_WORDS}
        
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

# --- Bulk Email Generation Function ---
def create_bulk_mailto_link(candidate_emails, job_title="Job Opportunity", sender_name="Recruiting Team"):
    """
    Generates a mailto: link for sending bulk emails to multiple candidates.
    """
    if not candidate_emails:
        return None
    
    # Join all emails with a comma
    recipients = ",".join(candidate_emails)
    
    subject = urllib.parse.quote(f"Invitation for Interview - {job_title}")
    
    body = f"""Dear Candidate,

We hope this email finds you well.

We are pleased to inform you that your application for the {job_title} position has been reviewed, and we are very impressed with your profile.
The {sender_name}"""

    return f"mailto:{recipients}?subject={subject}&body={urllib.parse.quote(body)}"


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="ScreenerPro - AI Resume Screener", page_icon="🧠")
st.title("🧠 ScreenerPro – AI-Powered Resume Screener")

# --- Job Description and Controls Section ---
st.markdown("## ⚙️ Define Job Requirements & Screening Criteria")
col1, col2 = st.columns([2, 1])

# Directory for pre-loaded JDs
JD_DATA_DIR = "data" # Ensure you have a 'data' folder in your project root with JD .txt files

# Try to list pre-loaded JDs
job_role_files = {}
try:
    if os.path.exists(JD_DATA_DIR) and os.path.isdir(JD_DATA_DIR):
        for filename in os.listdir(JD_DATA_DIR):
            if filename.endswith(".txt"):
                # Use filename without extension as display name
                display_name = os.path.splitext(filename)[0].replace('_', ' ').title()
                job_role_files[display_name] = os.path.join(JD_DATA_DIR, filename)
    else:
        st.warning(f"'{JD_DATA_DIR}' directory not found. Pre-loaded JDs will not be available. Please create a 'data' folder and place your JD .txt files inside it.")
except Exception as e:
    st.error(f"Error listing JD files: {e}")

job_roles_options = ["Upload New JD"] + sorted(list(job_role_files.keys()))
selected_jd_option = st.selectbox("Select a Pre-Loaded Job Role or Upload Your Own", job_roles_options, key="jd_option_select")

jd_text = ""
jd_file_uploader = None

if selected_jd_option == "Upload New JD":
    jd_file_uploader = st.file_uploader("Upload Job Description (TXT or PDF)", type=["txt", "pdf"], help="Upload a .txt or .pdf file containing the job description.")
    if jd_file_uploader:
        if jd_file_uploader.type == "application/pdf":
            jd_text = extract_text_from_pdf(jd_file_uploader)
        else: # text file
            jd_text = jd_file_uploader.read().decode("utf-8")
else:
    # Load selected pre-loaded JD
    jd_filepath = job_role_files.get(selected_jd_option)
    if jd_filepath and os.path.exists(jd_filepath):
        try:
            with open(jd_filepath, 'r', encoding='utf-8') as f:
                jd_text = f.read()
            st.success(f"Loaded '{selected_jd_option}' Job Description.")
        except Exception as e:
            st.error(f"Error reading pre-loaded JD '{selected_jd_option}': {e}")
    else:
        st.error(f"Pre-loaded JD file for '{selected_jd_option}' not found or accessible.")


with col1:
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

    # Extract JD skills and embedding here, after jd_text is available
    jd_embedding = None
    jd_skills = []
    if jd_text and model:
        jd_skills = extract_skills_from_text(jd_text)
        jd_embedding = get_job_description_embedding(jd_text, model)
    
    if jd_text and not jd_skills:
        st.warning("No specific skills found in the uploaded Job Description. Skill matching might be less effective.")


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
    # MODIFIED: Use extract_skills_from_text to ensure only actual skills are in the word cloud
    jd_words_for_cloud = " ".join([word for word in extract_skills_from_text(jd_text)])
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

        # --- Skill Extraction from Resume ---
        resume_skills = extract_skills_from_text(text)
        # st.write(f"**Extracted Skills:** {', '.join(resume_skills) if resume_skills else 'No specific skills found.'}") # Moved to detailed view

        # --- Calculate Skill Match based on Extracted JD Skills ---
        skill_match_percentage = 0
        matched_skills = set()
        missing_skills = set()

        # The target skills are now directly from the JD (extracted earlier)
        target_skills_from_jd = {skill.lower() for skill in jd_skills} # Ensure JD skills are lowercase for comparison
        
        if target_skills_from_jd: # Only calculate if JD skills were found
            resume_skills_lower = {skill.lower() for skill in resume_skills} # Convert extracted resume skills to lowercase
            
            matched_skills = resume_skills_lower.intersection(target_skills_from_jd)
            missing_skills = target_skills_from_jd.difference(resume_skills_lower)
            
            if len(target_skills_from_jd) > 0: # Avoid division by zero
                skill_match_percentage = (len(matched_skills) / len(target_skills_from_jd)) * 100
            
            # st.write(f"**Skill Match to Job Description:** {skill_match_percentage:.2f}%") # Moved to detailed view
            # if matched_skills: st.write(f"**Matched Skills:** {', '.join(sorted(list(matched_skills)))}")
            # if missing_skills: st.write(f"**Missing Skills from Job Description:** {', '.join(sorted(list(missing_skills)))}")
        # else:
            # st.info("No target skills extracted from the Job Description for matching.") # Moved to overall JD warning

        # --- Generate Embedding for Resume and Predict ML Match ---
        match_proba = 0.0
        semantic_sim_score = 0.0
        if jd_embedding is not None: # This check is sufficient for ML model
            # Call semantic_score which internally uses predict_match_ml
            score_from_semantic, _, semantic_sim_score = semantic_score(text, jd_text, exp, candidate_name)
            match_proba = score_from_semantic # semantic_score now returns the blended score
        else:
            st.info(f"AI Similarity Score not available for {candidate_name} without Job Description embedding.")

        # --- Generate AI Suggestion ---
        detailed_ai_suggestion = generate_ai_suggestion(
            candidate_name=candidate_name,
            score=match_proba, # Pass the final blended score
            years_exp=exp,
            semantic_similarity=semantic_sim_score,
            jd_text=jd_text,
            resume_text=text,
            matched_keywords=sorted(list(matched_skills)), # Pass sorted lists
            missing_skills=sorted(list(missing_skills))
        )

        results.append({
            "File Name": file.name,
            "Candidate Name": candidate_name,
            "Score (%)": match_proba, # Use the blended score
            "Years Experience": exp,
            "Email": email or "Not Found",
            "AI Suggestion": detailed_ai_suggestion,
            "Matched Keywords": ", ".join(sorted(list(matched_skills))),
            "Missing Skills": ", ".join(sorted(list(missing_skills))),
            "Semantic Similarity": semantic_sim_score,
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
        
        # Use the full AI Suggestion here
        st.markdown(f"**AI Assessment:** {top_candidate['AI Suggestion']}") 
        
        # Determine job title for email
        job_title_for_email = "Job Opportunity" 
        if selected_jd_option != "Upload New JD":
            job_title_for_email = selected_jd_option # Use selected JD name
        elif jd_file_uploader and jd_file_uploader.name:
            # Try to infer from uploaded file name
            job_title_for_email = os.path.splitext(jd_file_uploader.name)[0].replace('_', ' ').title()
        elif jd_text: # Fallback to first line of JD text
            first_line = jd_text.strip().split('\n')[0]
            if len(first_line) < 100:
                job_title_for_email = first_line.title()

        if top_candidate['Email'] != "Not Found":
            mailto_link_top = create_mailto_link(
                recipient_email=top_candidate['Email'],
                candidate_name=top_candidate['Candidate Name'],
                job_title=job_title_for_email
            )
            st.markdown(f'<a href="{mailto_link_top}" target="_blank"><button style="background-color:#00cec9;color:white;border:none;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;border-radius:8px;">📧 Invite Top Candidate for Interview</button></a>', unsafe_allow_html=True)
        else:
            st.info(f"Email address not found for {top_candidate['Candidate Name']}. Cannot send automated invitation.")
        
        st.markdown("---")
        st.info("For detailed analytics, matched keywords, and missing skills for ALL candidates, please navigate to the **Comprehensive Candidate Results Table** below.")

    else:
        st.info("No candidates processed yet to determine the top candidate.")


    # === AI Recommendation for Shortlisted Candidates (Streamlined) ===
    st.markdown("## 🌟 Shortlisted Candidates Overview")
    st.caption("Candidates meeting your score and experience criteria, with their AI-generated summaries.")

    shortlisted_candidates = df[(df['Score (%)'] >= cutoff) & (df['Years Experience'] >= min_experience)].copy() # Use .copy() to avoid SettingWithCopyWarning

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
                    format="%.1f", # Changed format to .1f for consistency
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

        # --- Email All Shortlisted Candidates Button ---
        shortlisted_emails = [email for email in shortlisted_candidates['Email'] if email != "Not Found"]
        if shortlisted_emails:
            bulk_mailto_link = create_bulk_mailto_link(
                candidate_emails=shortlisted_emails,
                job_title=job_title_for_email # Use the determined job title
            )
            st.markdown(f'<a href="{bulk_mailto_link}" target="_blank"><button style="background-color:#4CAF50;color:white;border:none;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;border-radius:8px;">📧 Email All Shortlisted Candidates ({len(shortlisted_emails)})</button></a>', unsafe_allow_html=True)
        else:
            st.info("No email addresses found for shortlisted candidates to send bulk invites.")

        st.info("For individual detailed AI assessments and action steps, please refer to the table above or the Comprehensive Candidate Results Table.")

    else:
        st.warning("No candidates met the defined screening criteria (score cutoff and minimum experience). You might consider adjusting the sliders or reviewing the uploaded resumes/JD.")

    st.markdown("---")

    # Add a 'Tag' column for quick categorization
    df['Tag'] = df.apply(lambda row: "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3 else (
        "✅ Good Fit" if row['Score (%)'] >= 75 else "⚠️ Needs Review"), axis=1)

    st.markdown("## 📋 Comprehensive Candidate Results Table")
    st.caption("Full details for all processed resumes, including individual AI suggestions. **For deep dive analytics and keyword breakdowns, refer to the table below.**")
    
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
                format="%.1f", # Changed format to .1f for consistency
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

# --- About Section (Still in sidebar as it's common practice for app info) ---
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
