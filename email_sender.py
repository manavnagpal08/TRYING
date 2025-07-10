import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Replace these with your actual Gmail and App Password
SENDER_EMAIL = "manav.nagpal2005@gmail.com"
APP_PASSWORD = "fuzgqatrvugzxbnb"

def send_email_to_candidate(name, score, feedback, recipient, subject, message):
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient
        msg["Subject"] = subject

        body = f"""
        Hello,

        {message}

        --- 
        📄 Candidate: {name}  
        ✅ Score: {score}%  
        💬 Feedback: {feedback}  

        Best regards,  
        HR Team
        """
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)

        print(f"✅ Email sent to {recipient}")
    except Exception as e:
        print(f"❌ Email failed: {e}")
