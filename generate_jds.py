import os

# Define a much more diverse list of job roles
job_roles = [
    "Software Engineer",
    "Data Scientist",
    "Product Manager",
    "Marketing Specialist",
    "Financial Analyst",
    "Human Resources Generalist",
    "UX/UI Designer",
    "Cloud Architect",
    "Cybersecurity Engineer",
    "DevOps Specialist",
    "Mobile App Developer (iOS)",
    "Business Development Manager",
    "Project Coordinator",
    "Technical Writer",
    "Sales Representative",
    "Customer Support Specialist",
    "Operations Coordinator",
    "Supply Chain Manager",
    "Mechanical Design Engineer",
    "Electrical Systems Engineer",
    "Civil Structural Engineer",
    "Research Chemist",
    "Clinical Biologist",
    "Registered Nurse (ER)",
    "High School Math Teacher"
]

# Create a 'data' directory if it doesn't exist
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Generate a placeholder job description for each role
for role in job_roles:
    file_name = role.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "") + ".txt"
    file_path = os.path.join(output_dir, file_name)

    jd_content = f"""
Job Title: {role}

About Us:
We are a leading company in [Industry/Field placeholder] looking for passionate and skilled individuals to join our team. We value innovation, collaboration, and continuous learning.

Job Summary:
As a {role}, you will be responsible for [key responsibility 1 placeholder], [key responsibility 2 placeholder], and [key responsibility 3 placeholder]. You will work closely with [team/department placeholder] to [achieve specific goal placeholder].

Key Responsibilities:
- [Specific duty 1 related to role placeholder]
- [Specific duty 2 related to role placeholder]
- [Specific duty 3 related to role placeholder]
- Collaborate with cross-functional teams.
- Ensure high quality and timely delivery of projects.
- Stay updated with industry best practices and emerging technologies.

Required Skills and Qualifications:
- Bachelor's or Master's degree in [Relevant Field placeholder].
- X+ years of proven experience in {role.replace(" (iOS)", "")} or a similar role.
- Strong proficiency in [Core skill 1 placeholder].
- Experience with [Core skill 2 placeholder].
- Excellent problem-solving and analytical skills.
- Strong communication and interpersonal abilities.

Preferred Qualifications:
- Experience with [Additional desirable skill/tool placeholder].
- Certifications relevant to the role.
- Ability to work in a fast-paced and dynamic environment.

Benefits:
- Competitive salary and benefits package.
- Opportunities for professional growth and development.
- Flexible work arrangements.
- Inclusive and supportive work environment.
"""
    # Add specific content based on role for better differentiation
    if "Software Engineer" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "technology solutions")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "designing scalable software systems")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "developing high-quality code")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "participating in code reviews")
        jd_content = jd_content.replace("[team/department placeholder]", "development team")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "deliver innovative software products")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Develop and maintain robust software applications using Python and Java.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Implement RESTful APIs and microservices architectures.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Write unit and integration tests to ensure code quality.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Computer Science or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Python, Java, Spring Boot")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "REST APIs, Microservices, SQL databases")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Docker, Kubernetes, AWS, Git, Agile Scrum")
    elif "Data Scientist" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "data analytics")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "building predictive models")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "analyzing large datasets")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "communicating insights to stakeholders")
        jd_content = jd_content.replace("[team/department placeholder]", "data science team")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "drive data-driven decision-making")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Develop and implement machine learning models using Python and R.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Perform exploratory data analysis and feature engineering.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Create compelling data visualizations and dashboards.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Data Science, Statistics, or Computer Science")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Python (Pandas, NumPy, Scikit-learn), R, SQL")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Machine Learning, Deep Learning, Statistical Modeling")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "TensorFlow, PyTorch, Spark, Tableau, Power BI, NLP")
    elif "Product Manager" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "software product development")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "defining product vision and strategy")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "managing product roadmaps")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "gathering user requirements")
        jd_content = jd_content.replace("[team/department placeholder]", "product and engineering teams")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "deliver successful products that meet market needs")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Define and prioritize product features based on market research and user feedback.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Translate business requirements into detailed user stories and specifications.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Collaborate with engineering, design, and marketing teams throughout the product lifecycle.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Business, Computer Science, or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Product Strategy, Roadmap Development, Agile Methodologies")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "User Stories, A/B Testing, Market Analysis")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Jira, Confluence, Figma, SQL, Google Analytics")
    elif "Marketing Specialist" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "digital marketing")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "developing marketing campaigns")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "managing social media presence")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "analyzing campaign performance")
        jd_content = jd_content.replace("[team/department placeholder]", "marketing department")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "increase brand awareness and lead generation")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Execute digital marketing campaigns across various channels (SEO, SEM, social media).")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Create engaging content for websites, blogs, and social media platforms.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Monitor and report on campaign performance using analytics tools.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Marketing, Communications, or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Digital Marketing, SEO, SEM, Social Media Marketing")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Content Creation, Email Marketing, Google Analytics")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "CRM platforms (e.g., HubSpot), Marketing Automation, A/B Testing")
    elif "Financial Analyst" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "financial services")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "conducting financial forecasting")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "performing variance analysis")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "preparing financial reports")
        jd_content = jd_content.replace("[team/department placeholder]", "finance team")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "support strategic financial planning")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Develop and maintain complex financial models for budgeting and forecasting.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Analyze financial data to identify trends, risks, and opportunities.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Prepare detailed financial reports and presentations for management.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Finance, Accounting, or Economics")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Financial Modeling, Valuation, Microsoft Excel (advanced)")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Financial Reporting, Data Analysis, SQL")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Power BI, Tableau, SAP, Python for financial analysis")
    elif "Human Resources Generalist" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "human resources")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "managing employee relations")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "supporting talent acquisition")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "administering HR policies")
        jd_content = jd_content.replace("[team/department placeholder]", "HR department")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "foster a positive and productive work environment")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Provide guidance and support to employees and managers on HR policies and procedures.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Assist with recruitment efforts, including sourcing, interviewing, and onboarding.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Manage HR administrative tasks, including record-keeping and benefits administration.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Human Resources, Business Administration, or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Employee Relations, Talent Acquisition, HR Policies & Compliance")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Performance Management, HRIS (Human Resources Information Systems)")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Workday, SAP SuccessFactors, Payroll Processing, Microsoft Office Suite")
    elif "UX/UI Designer" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "digital product design")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "conducting user research")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "creating wireframes and prototypes")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "designing intuitive user interfaces")
        jd_content = jd_content.replace("[team/department placeholder]", "design and product teams")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "enhance user experience and product usability")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Conduct user research, usability testing, and analyze user feedback.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Create wireframes, storyboards, user flows, and prototypes using design tools.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Design visually appealing and intuitive user interfaces (UI) for web and mobile applications.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Interaction Design, Graphic Design, or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "User Research, Wireframing, Prototyping, UI Design")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Figma, Sketch, Adobe XD, Usability Testing")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "HTML, CSS, JavaScript, Design Systems, Accessibility Standards")
    elif "Cloud Architect" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "cloud computing solutions")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "designing cloud infrastructure")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "implementing cloud migration strategies")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "optimizing cloud resources")
        jd_content = jd_content.replace("[team/department placeholder]", "architecture and engineering teams")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "build robust, scalable, and cost-effective cloud environments")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Design and implement scalable, secure, and highly available cloud architectures (AWS, Azure, GCP).")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Lead cloud migration projects, ensuring minimal disruption and data integrity.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Optimize cloud resource utilization and costs through effective governance.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Computer Science, Information Technology, or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "AWS, Azure, Google Cloud Platform (GCP), Cloud Architecture Design")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Terraform, Ansible, Kubernetes, Docker")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "CI/CD, Serverless Computing, Network Security, Solution Architecture")
    elif "Cybersecurity Engineer" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "information security")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "implementing security measures")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "monitoring for threats")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "responding to security incidents")
        jd_content = jd_content.replace("[team/department placeholder]", "security operations center (SOC)")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "protect organizational assets from cyber threats")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Design, implement, and maintain security systems and controls (firewalls, IDS/IPS).")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Conduct vulnerability assessments and penetration testing.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Respond to security incidents, analyze root causes, and implement corrective actions.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Cybersecurity, Computer Science, or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Network Security, Incident Response, Vulnerability Management")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "SIEM (Security Information and Event Management), Firewalls, Endpoint Protection")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Python for scripting, Security Audits, Compliance (e.g., ISO 27001, NIST)")
    elif "DevOps Specialist" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "software delivery automation")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "automating CI/CD pipelines")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "managing infrastructure as code")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "optimizing deployment processes")
        jd_content = jd_content.replace("[team/department placeholder]", "development and operations teams")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "accelerate software delivery and improve system reliability")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Design, implement, and maintain CI/CD pipelines using Jenkins, GitLab CI, or similar tools.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Manage cloud infrastructure using Infrastructure as Code (IaC) tools like Terraform and Ansible.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Implement monitoring and logging solutions (Prometheus, Grafana, ELK Stack) for system health.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Computer Science, DevOps, or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "CI/CD, Docker, Kubernetes, Cloud Platforms (AWS/Azure/GCP)")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Terraform, Ansible, Scripting (Python/Bash)")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Git, Linux, Microservices, System Administration")
    elif "Mobile App Developer (iOS)" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "mobile application development")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "developing iOS applications")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "implementing new features")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "optimizing app performance")
        jd_content = jd_content.replace("[team/department placeholder]", "mobile development team")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "create engaging and high-performing iOS applications")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Design and build advanced applications for the iOS platform using Swift and SwiftUI.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Collaborate with cross-functional teams to define, design, and ship new features.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Unit-test code for robustness, including edge cases, usability, and general reliability.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Computer Science or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Swift, SwiftUI, iOS SDK, Xcode")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "UI/UX principles, RESTful APIs, Git")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Objective-C, Firebase, Core Data, Agile development")
    elif "Business Development Manager" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "sales and growth")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "identifying new business opportunities")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "building client relationships")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "driving revenue growth")
        jd_content = jd_content.replace("[team/department placeholder]", "sales and marketing teams")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "expand market reach and achieve sales targets")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Identify and pursue new business opportunities and partnerships.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Build and maintain strong relationships with key clients and stakeholders.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Develop and execute strategic sales plans to achieve revenue goals.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Business Administration, Marketing, or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Sales Strategy, Client Relationship Management, Negotiation")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Lead Generation, Market Analysis, CRM Software (e.g., Salesforce)")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Presentation Skills, Financial Acumen, Contract Management")
    elif "Project Coordinator" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "project management")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "assisting with project planning")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "tracking project progress")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "facilitating team communication")
        jd_content = jd_content.replace("[team/department placeholder]", "project management office (PMO)")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "ensure projects are completed on time and within budget")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Support Project Managers in developing project plans, schedules, and budgets.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Monitor project progress, identify potential issues, and assist in resolution.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Organize and facilitate project meetings, prepare agendas, and document minutes.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Business, Project Management, or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Project Planning, Scheduling, Communication Skills")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Microsoft Office Suite (Excel, Word, PowerPoint), Project Management Software")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Jira, Asana, Smartsheet, Agile Methodologies")
    elif "Technical Writer" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "technical documentation")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "creating user manuals")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "developing API documentation")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "maintaining knowledge bases")
        jd_content = jd_content.replace("[team/department placeholder]", "product and engineering teams")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "produce clear, concise, and accurate technical content")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Write, edit, and maintain high-quality technical documentation, including user guides and release notes.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Collaborate with subject matter experts to gather information and ensure accuracy.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Organize and structure content for optimal readability and user experience.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Technical Communication, English, or Computer Science")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Technical Writing, Documentation, Content Management Systems (CMS)")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "API Documentation, Markdown, XML/HTML")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "MadCap Flare, Confluence, Git, Adobe FrameMaker")
    elif "Sales Representative" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "B2B sales")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "generating new leads")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "presenting product solutions")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "closing deals")
        jd_content = jd_content.replace("[team/department placeholder]", "sales team")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "exceed sales quotas and expand customer base")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Identify and qualify new sales opportunities through prospecting and outreach.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Conduct compelling product demonstrations and presentations to potential clients.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Negotiate contracts and close sales to achieve revenue targets.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Business, Sales, or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Sales Acumen, Lead Generation, CRM Software (e.g., Salesforce)")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Negotiation, Presentation Skills, Client Relationship Management")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Cold Calling, Sales Forecasting, Microsoft Office Suite")
    elif "Customer Support Specialist" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "customer service")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "resolving customer inquiries")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "providing product assistance")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "maintaining customer satisfaction")
        jd_content = jd_content.replace("[team/department placeholder]", "customer support team")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "deliver exceptional customer service and build loyalty")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Respond to customer inquiries via phone, email, and chat in a timely and professional manner.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Troubleshoot product issues and provide effective solutions.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Document customer interactions and feedback accurately in the CRM system.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Customer Service, Communications, or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Customer Service, Problem Solving, Communication (written & verbal)")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "CRM Software (e.g., Zendesk, Salesforce Service Cloud), Troubleshooting")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Active Listening, Empathy, Multitasking, Product Knowledge")
    elif "Operations Coordinator" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "business operations")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "streamlining operational processes")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "managing logistics activities")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "ensuring efficient workflow")
        jd_content = jd_content.replace("[team/department placeholder]", "operations department")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "optimize operational efficiency and support business growth")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Assist in the planning and execution of daily operational activities.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Coordinate logistics, inventory, and supply chain processes.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Identify areas for process improvement and implement solutions.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Business Administration, Supply Chain, or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Process Improvement, Logistics Coordination, Data Entry")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Microsoft Office Suite (Excel), ERP Systems")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Project Management Software, Communication Skills, Problem Solving")
    elif "Supply Chain Manager" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "supply chain and logistics")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "overseeing end-to-end supply chain operations")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "optimizing inventory levels")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "managing supplier relationships")
        jd_content = jd_content.replace("[team/department placeholder]", "supply chain team")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "ensure efficient and cost-effective flow of goods and services")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Develop and implement supply chain strategies to improve efficiency and reduce costs.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Manage inventory levels, demand forecasting, and logistics operations.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Negotiate contracts with suppliers and manage vendor performance.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Supply Chain Management, Logistics, or Business Administration")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Supply Chain Management, Logistics, Inventory Optimization")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Demand Planning, Supplier Management, ERP Systems (e.g., SAP, Oracle SCM)")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Data Analysis (Excel, SQL), Lean Six Sigma, Project Management")
    elif "Mechanical Design Engineer" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "product design and manufacturing")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "designing mechanical components")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "performing engineering analysis")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "creating technical drawings")
        jd_content = jd_content.replace("[team/department placeholder]", "engineering design team")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "develop innovative and reliable mechanical products")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Design mechanical components and assemblies using CAD software (SolidWorks, AutoCAD).")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Conduct engineering analysis, including FEA (Finite Element Analysis) and thermodynamics.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Prepare detailed technical drawings, specifications, and BOMs (Bills of Materials).")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Mechanical Engineering or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "CAD Software (SolidWorks, AutoCAD), FEA, Product Design")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Thermodynamics, Materials Science, GD&T (Geometric Dimensioning & Tolerancing)")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "MATLAB, ANSYS, Prototyping, Manufacturing Processes")
    elif "Electrical Systems Engineer" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "electrical engineering")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "designing electrical circuits")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "developing embedded systems")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "testing electrical components")
        jd_content = jd_content.replace("[team/department placeholder]", "electrical engineering team")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "develop robust and efficient electrical systems")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Design and analyze electrical circuits, including analog and digital components.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Develop and debug firmware for embedded systems using C/C++.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Perform testing and validation of electrical systems and components.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Electrical Engineering or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Circuit Design, Embedded Systems, PCB Layout")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Microcontrollers, Signal Processing, Power Electronics")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Altium Designer, Eagle, SPICE, LabVIEW")
    elif "Civil Structural Engineer" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "civil engineering and construction")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "designing structural elements")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "performing structural analysis")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "preparing construction documents")
        jd_content = jd_content.replace("[team/department placeholder]", "civil engineering department")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "ensure the safety and stability of civil structures")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Design structural elements for buildings, bridges, and other civil infrastructure.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Perform structural analysis using industry-standard software (e.g., SAP2000, ETABS).")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Prepare detailed construction drawings and specifications (AutoCAD, Civil 3D, Revit).")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Civil Engineering with a focus on Structures")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Structural Analysis, AutoCAD, Civil 3D, Revit")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Concrete Design, Steel Design, Geotechnical Engineering")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Project Management, Construction Methods, Building Codes (e.g., IBC)")
    elif "Research Chemist" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "chemical research and development")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "designing chemical experiments")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "conducting laboratory analysis")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "interpreting scientific data")
        jd_content = jd_content.replace("[team/department placeholder]", "R&D laboratory")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "innovate and develop new chemical compounds and processes")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Design and execute chemical experiments following established protocols and safety guidelines.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Perform analytical testing using techniques like HPLC, GC-MS, NMR, and FTIR.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Analyze and interpret complex scientific data, preparing reports and presentations.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Chemistry, Chemical Engineering, or a related scientific field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Organic Synthesis, Analytical Chemistry, Spectroscopy (NMR, FTIR)")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "HPLC, GC-MS, Laboratory Safety, Data Interpretation")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "LIMS (Laboratory Information Management System), ChemDraw, Statistical Analysis Software")
    elif "Clinical Biologist" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "biomedical research and diagnostics")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "conducting biological experiments")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "analyzing biological samples")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "interpreting clinical data")
        jd_content = jd_content.replace("[team/department placeholder]", "clinical research laboratory")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "advance understanding of biological processes and disease mechanisms")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Design and conduct biological experiments using techniques such as cell culture, PCR, and Western Blot.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Analyze biological samples and interpret results for research or diagnostic purposes.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Maintain accurate laboratory records and contribute to scientific publications.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Biology, Biochemistry, Molecular Biology, or a related life science")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Molecular Biology, Cell Culture, PCR, Western Blot")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Microscopy, Bioinformatics, Data Analysis, Laboratory Techniques")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Flow Cytometry, ELISA, R or Python for data analysis, GLP/GCP regulations")
    elif "Registered Nurse (ER)" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "emergency healthcare")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "providing emergency patient care")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "administering medications")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "documenting patient information")
        jd_content = jd_content.replace("[team/department placeholder]", "Emergency Room (ER)")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "deliver high-quality, compassionate care to emergency patients")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Assess, plan, implement, and evaluate patient care in a fast-paced emergency setting.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Administer medications and treatments as prescribed, monitoring patient responses.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Document all patient care activities accurately and timely in electronic health records (EHR).")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Nursing (ADN or BSN)")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Emergency Patient Care, Triage, Medication Administration")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "BLS/ACLS Certification, Critical Thinking, Electronic Health Records (EHR)")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "PALS Certification, Trauma Nursing, Crisis Intervention, Communication Skills")
    elif "High School Math Teacher" in role:
        jd_content = jd_content.replace("[Industry/Field placeholder]", "secondary education")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "developing engaging math lessons")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "instructing students in various math subjects")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "assessing student progress")
        jd_content = jd_content.replace("[team/department placeholder]", "Mathematics Department")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "foster a love for mathematics and prepare students for future success")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", "Develop and deliver engaging math lessons for high school students (Algebra, Geometry, Calculus).")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Assess student understanding through various methods and provide constructive feedback.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Collaborate with colleagues, parents, and administrators to support student learning.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "Mathematics Education or a related field")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "Lesson Planning, Classroom Management, Differentiated Instruction")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "Algebra, Geometry, Calculus, Student Assessment")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "Google Classroom, Interactive Whiteboards, Special Education Needs (SEN) strategies")
    # Add more specific replacements for other roles as needed (e.g., for any roles without explicit conditions above)
    else:
        # Fallback for any roles not explicitly handled, using more generic but still descriptive placeholders
        jd_content = jd_content.replace("[Industry/Field placeholder]", "diverse industries")
        jd_content = jd_content.replace("[key responsibility 1 placeholder]", "contributing to key initiatives")
        jd_content = jd_content.replace("[key responsibility 2 placeholder]", "collaborating with various teams")
        jd_content = jd_content.replace("[key responsibility 3 placeholder]", "driving successful outcomes")
        jd_content = jd_content.replace("[team/department placeholder]", "dynamic teams")
        jd_content = jd_content.replace("[achieve specific goal placeholder]", "achieve organizational objectives")
        jd_content = jd_content.replace("[Specific duty 1 related to role placeholder]", f"Perform core duties related to {role}.")
        jd_content = jd_content.replace("[Specific duty 2 related to role placeholder]", "Contribute to strategic planning and execution.")
        jd_content = jd_content.replace("[Specific duty 3 related to role placeholder]", "Analyze data and provide actionable recommendations.")
        jd_content = jd_content.replace("[Relevant Field placeholder]", "relevant field of study")
        jd_content = jd_content.replace("[Core skill 1 placeholder]", "strong analytical skills")
        jd_content = jd_content.replace("[Core skill 2 placeholder]", "excellent communication abilities")
        jd_content = jd_content.replace("[Additional desirable skill/tool placeholder]", "relevant industry tools and software")


    with open(file_path, "w", encoding="utf-8") as f:
        f.write(jd_content.strip())
    print(f"Generated: {file_path}")

print("\nAll job descriptions generated successfully in the 'data' folder.")
