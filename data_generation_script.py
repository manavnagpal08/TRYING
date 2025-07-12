import pandas as pd
import random
import os

# --- Data Generation Logic (Copied from model_training_code) ---
job_categories = [
    'Software Engineer', 'Data Scientist', 'Marketing Specialist',
    'Project Manager', 'Customer Service', 'Financial Analyst',
    'HR Specialist', 'Graphic Designer', 'Sales Professional',
    'Operations Manager', 'Product Manager', 'Business Analyst',
    'UX Researcher', 'DevOps Engineer', 'Network Administrator',
    'Cybersecurity Analyst', 'Cloud Architect', 'Content Writer',
    'Legal Counsel', 'Research Scientist'
]

category_keywords = {
    'Software Engineer': [
        "Python, Java, C++, JavaScript, Go", "web development, backend, frontend, full stack",
        "cloud platforms (AWS, Azure, GCP), Docker, Kubernetes", "algorithms, data structures, system design",
        "agile methodologies, continuous integration, deployment (CI/CD)", "API development, microservices",
        "mobile development (iOS, Android), Swift, Kotlin", "testing, debugging, code review",
        "version control (Git), software architecture, scalable systems"
    ],
    'Data Scientist': [
        "Python, R, SQL, Scala", "machine learning, deep learning, statistical modeling",
        "data analysis, data visualization, predictive analytics", "Big Data (Spark, Hadoop, Hive)",
        "A/B testing, experimental design", "NLP, computer vision",
        "statistical inference, causal modeling, time series analysis", "ETL processes, data warehousing",
        "Jupyter, TensorFlow, PyTorch, scikit-learn"
    ],
    'Marketing Specialist': [
        "digital marketing, SEO, SEM, content marketing", "social media management, campaign strategy",
        "email marketing, Google Analytics, HubSpot", "brand management, market research",
        "copywriting, public relations, lead generation", "CRM (Salesforce Marketing Cloud, Mailchimp)",
        "marketing automation, analytics, ROI tracking", "PPC, display advertising, conversion optimization"
    ],
    'Project Manager': [
        "PMP, Agile, Scrum, Waterfall", "project planning, risk management, budget control",
        "team leadership, stakeholder communication, conflict resolution", "Jira, Asana, Microsoft Project",
        "software development lifecycle (SDLC), product delivery", "resource allocation, vendor management",
        "change management, quality assurance, project charter", "budget forecasting, timeline management"
    ],
    'Customer Service': [
        "customer support, client relations, technical assistance", "issue resolution, conflict de-escalation",
        "CRM software (Salesforce, Zendesk), ticketing systems", "communication skills, empathy, problem-solving",
        "call center experience, live chat support", "customer satisfaction, service level agreements (SLAs)",
        "multilingual support, product knowledge", "feedback collection, service improvement"
    ],
    'Financial Analyst': [
        "financial modeling, valuation, forecasting", "Excel, Bloomberg Terminal, financial statements",
        "investment analysis, portfolio management, risk assessment", "corporate finance, budgeting",
        "GAAP, IFRS, regulatory compliance", "data analysis, financial reporting, variance analysis",
        "capital markets, mergers & acquisitions (M&A)", "due diligence, economic research"
    ],
    'HR Specialist': [
        "recruitment, talent acquisition, onboarding", "employee relations, performance management",
        "HRIS (Workday, SAP SuccessFactors), payroll processing", "benefits administration, compliance (EEO, OSHA)",
        "training and development, HR policies", "compensation, diversity & inclusion (D&I)",
        "employee engagement, workforce planning", "talent management, organizational development"
    ],
    'Graphic Designer': [
        "Adobe Creative Suite (Photoshop, Illustrator, InDesign)", "UI/UX design, wireframing, prototyping",
        "branding, logo design, print design", "web design, motion graphics",
        "typography, color theory, visual communication", "Figma, Sketch, Adobe XD",
        "illustration, animation, video editing", "brand guidelines, creative direction"
    ],
    'Sales Professional': [
        "B2B sales, B2C sales, account management", "lead generation, cold calling, prospecting",
        "CRM (Salesforce), sales pipeline management", "negotiation, closing deals, quota attainment",
        "product demonstration, client presentations", "sales strategy, market analysis",
        "customer retention, revenue growth", "cross-selling, upselling, sales forecasting"
    ],
    'Operations Manager': [
        "supply chain management, logistics, inventory control", "process optimization, efficiency improvement",
        "lean manufacturing, Six Sigma", "team leadership, resource allocation",
        "quality assurance, operational planning, vendor management", "budget management, cost reduction",
        "performance metrics, continuous improvement", "workflow automation, resource optimization"
    ],
    'Product Manager': [
        "product roadmap, product strategy, market analysis", "user stories, agile development, Scrum",
        "feature prioritization, competitive analysis", "go-to-market strategy, product launch",
        "user research, A/B testing, data-driven decisions", "Jira, Confluence, Aha!",
        "cross-functional team leadership, stakeholder management", "product lifecycle, user feedback"
    ],
    'Business Analyst': [
        "requirements gathering, process mapping, data analysis", "SQL, Excel, Tableau, Power BI",
        "stakeholder interviews, business process improvement", "system analysis, functional specifications",
        "UML, flowcharts, use cases", "problem-solving, analytical thinking",
        "data modeling, reporting, business intelligence"
    ],
    'UX Researcher': [
        "user interviews, usability testing, surveys", "qualitative research, quantitative research",
        "persona development, journey mapping, empathy maps", "A/B testing, eye tracking, card sorting",
        "Figma, Sketch, Adobe XD", "report writing, presentation skills",
        "design thinking, user-centered design, heuristic evaluation"
    ],
    'DevOps Engineer': [
        "CI/CD, Jenkins, GitLab CI, Azure DevOps", "Docker, Kubernetes, containerization",
        "cloud platforms (AWS, Azure, GCP), infrastructure as code (Terraform, CloudFormation)",
        "scripting (Bash, Python), automation", "monitoring (Prometheus, Grafana), logging (ELK stack)",
        "system administration, network configuration, security best practices",
        "Ansible, Chef, Puppet, configuration management"
    ],
    'Network Administrator': [
        "network configuration, troubleshooting, security protocols", "Cisco, Juniper, Palo Alto",
        "LAN/WAN, VPN, firewall management", "network monitoring, performance optimization",
        "TCP/IP, DNS, DHCP", "hardware installation, software upgrades",
        "technical support, documentation", "routing, switching, network design"
    ],
    'Cybersecurity Analyst': [
        "threat detection, vulnerability assessment, penetration testing", "SIEM, IDS/IPS, firewalls",
        "incident response, security audits, compliance (GDPR, HIPAA)", "network security, endpoint protection",
        "cryptography, ethical hacking, security frameworks (NIST, ISO 27001)"
    ],
    'Cloud Architect': [
        "cloud solution design, migration strategies, cost optimization", "AWS, Azure, Google Cloud Platform",
        "serverless architectures, microservices, containerization", "hybrid cloud, multi-cloud environments",
        "security architecture, scalability, reliability"
    ],
    'Content Writer': [
        "blog posts, articles, website content, SEO writing", "copywriting, editing, proofreading",
        "content strategy, digital storytelling, social media content", "grammar, style guides (AP, Chicago)",
        "keyword research, content management systems (CMS)"
    ],
    'Legal Counsel': [
        "contract negotiation, legal research, compliance", "corporate law, intellectual property",
        "litigation, regulatory affairs, risk assessment", "legal advice, dispute resolution",
        "drafting agreements, policy development"
    ],
    'Research Scientist': [
        "experimental design, data collection, statistical analysis", "academic research, scientific writing",
        "literature review, hypothesis testing, peer-reviewed publications", "laboratory techniques, modeling",
        "problem-solving, critical thinking, grant writing"
    ]
}

general_phrases = [
    "Proven track record of success.",
    "Strong problem-solver and analytical thinker.",
    "Excellent communication and interpersonal skills.",
    "Highly motivated and results-driven.",
    "Adept at working in fast-paced environments.",
    "Committed to continuous learning and professional development.",
    "Collaborative team player.",
    "Detail-oriented and organized.",
    "Ability to manage multiple projects simultaneously.",
    "Passionate about innovation and technology."
]

resume_texts = []
job_categories_list = []

num_entries = 10000 # Number of dummy entries to generate

print(f"Generating {num_entries} dummy resume entries...")
for _ in range(num_entries):
    category = random.choice(job_categories)
    
    num_keywords_to_sample = random.randint(2, min(6, len(category_keywords[category])))
    keywords = random.sample(category_keywords[category], k=num_keywords_to_sample)
    
    experience_level = random.choice([
        "Experienced", "Senior", "Junior", "Recent graduate", "Mid-level", "Lead", "Seasoned", "Expert", "Entry-level"
    ])
    years_exp = random.randint(0, 25)
    
    base_text = ""
    if experience_level == "Recent graduate":
        base_text = f"{experience_level} with a strong academic background and relevant coursework. Eager to apply skills in {category}."
    elif experience_level == "Entry-level":
        base_text = f"{experience_level} professional seeking opportunities in {category}. Developed foundational skills through internships and projects."
    else:
        base_text = f"{experience_level} {category} with {years_exp} years of experience."

    skill_phrases = []
    for i, kw_set in enumerate(keywords):
        if i == 0:
            skill_phrases.append(f"Proficient in {kw_set}")
        elif i == 1:
            skill_phrases.append(f"Expertise in {kw_set}")
        elif i == 2:
            skill_phrases.append(f"Strong background in {kw_set}")
        elif i == 3:
            skill_phrases.append(f"Key skills include {kw_set}")
        elif i == 4:
            skill_phrases.append(f"Highly skilled in {kw_set}")
        else:
            skill_phrases.append(f"Additional skills: {kw_set}")

    num_general_phrases = random.randint(1, 3)
    selected_general_phrases = random.sample(general_phrases, k=num_general_phrases)
    
    combined_phrases = skill_phrases + selected_general_phrases
    random.shuffle(combined_phrases)
    
    resume_text = base_text + " " + ". ".join(combined_phrases) + "."
    
    resume_texts.append(resume_text)
    job_categories_list.append(category)

df = pd.DataFrame({'resume_text': resume_texts, 'job_category': job_categories_list})

print("Dummy data generation complete.")

# --- Save to CSV ---
csv_file_path = 'dummy_resume_data.csv'
df.to_csv(csv_file_path, index=False)
print(f"Data saved to {csv_file_path}")

# --- Save to JSON ---
json_file_path = 'dummy_resume_data.json'
df.to_json(json_file_path, orient='records', indent=4) # 'records' for list of objects, 'indent' for readability
print(f"Data saved to {json_file_path}")

print("\nTo use this data in your model training script:")
print(f"1. Open your 'model_training_code' script (e.g., train_model.py).")
print(f"2. Locate the '--- 1. Data Loading (for Resume Screening) ---' section.")
print(f"3. Replace the entire dummy data generation block with one of the following lines:")
print(f"   For CSV: df = pd.read_csv('{csv_file_path}')")
print(f"   For JSON: df = pd.read_json('{json_file_path}')")
print(f"4. Ensure your X_raw and y assignments match the column names in the file (e.g., X_raw = df['resume_text'], y = df['job_category']).")
