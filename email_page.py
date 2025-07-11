import streamlit as st
import pandas as pd
import sqlite3 # Import sqlite3 for database interaction
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Database Configuration (MUST match screener.py) ---
DATABASE_FILE = "screening_data.db"

# --- Email Sending Function (Self-contained for this page) ---
def send_email(recipient_email, subject, body):
    # Use Streamlit Secrets for email credentials
    try:
        sender_email = st.secrets["email"]["username"]
        sender_password = st.secrets["email"]["password"]
    except KeyError:
        st.error("Email credentials not found in Streamlit Secrets. Please configure them in .streamlit/secrets.toml.")
        return False
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email. Error: {e}. Check sender email/password and ensure 'Less secure app access' is enabled or use App Passwords for Gmail.")
        return False

# --- Style Enhancements ---
st.markdown("""
<style>
.email-box {
    padding: 2rem;
    background: #f9f9fb;
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.07);
    animation: fadeInUp 0.7s ease;
    margin-top: 1.5rem;
}
.email-entry {
    margin-bottom: 1.2rem;
    padding: 1rem;
    background: white;
    border-radius: 12px;
    border-left: 4px solid #00cec9;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

st.subheader("📧 Send Email to Shortlisted Candidates")

# --- Load Shortlisted Candidates from Database ---
conn = None # Initialize conn outside try block
try:
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    # Fetch candidates explicitly marked as shortlisted
    c.execute('SELECT candidate_name, predicted_score, years_experience, ai_suggestion, email FROM results WHERE shortlisted = TRUE')
    
    columns = [description[0] for description in c.description]
    shortlisted_data = c.fetchall()
    
    shortlisted_df = pd.DataFrame(shortlisted_data, columns=columns)
    
    # Rename columns for display and consistency with template
    shortlisted_df = shortlisted_df.rename(columns={
        'predicted_score': 'Score (%)',
        'years_experience': 'Years Experience',
        'ai_suggestion': 'AI Suggestion'
    })

except sqlite3.OperationalError as e:
    st.error(f"Database error: {e}. Ensure the database file is accessible and not locked.")
    shortlisted_df = pd.DataFrame()
except Exception as e:
    st.error(f"Error loading shortlisted candidates from database: {e}")
    shortlisted_df = pd.DataFrame()
finally:
    if conn:
        conn.close()

if not shortlisted_df.empty:
    st.success(f"✅ {len(shortlisted_df)} candidates currently shortlisted in the database.")
    
    # Display the shortlisted candidates (using 'AI Suggestion' instead of 'Feedback')
    st.dataframe(shortlisted_df[["candidate_name", "Score (%)", "Years Experience", "AI Suggestion", "email"]].rename(columns={'candidate_name': 'Candidate Name', 'email': 'Email'}), use_container_width=True)

    st.markdown("<div class='email-box'>", unsafe_allow_html=True)

    st.markdown("### ✉️ Assign Emails")
    # Pre-fill emails if available from the database
    email_map = {row['candidate_name']: row['email'] if row['email'] != 'Not Found' else '' for _, row in shortlisted_df.iterrows()}

    for i, row in shortlisted_df.iterrows():
        candidate_name = row["candidate_name"]
        default_email = email_map.get(candidate_name, '')
        with st.container():
            st.markdown(f"<div class='email-entry'>", unsafe_allow_html=True)
            email_input = st.text_input(f"📧 Email for {candidate_name}", value=default_email, key=f"email_{i}")
            email_map[candidate_name] = email_input
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📝 Customize Email Template")
    subject = st.text_input("Subject", value="🎉 You're Shortlisted for [Job Title]!")
    body_template = st.text_area("Body", height=180, value="""
Hi {{candidate_name}},

Congratulations! 🎉

You've been shortlisted for the next phase based on your resume review for the [Job Title] position.

We were particularly impressed by your profile, which aligns well with the role's requirements.

We'll be in touch with next steps shortly.

Warm regards,  
HR Team
""")

    if st.button("📤 Send All Emails"):
        sent_count = 0
        for _, row in shortlisted_df.iterrows():
            candidate_name = row["candidate_name"]
            score = row["Score (%)"]
            ai_suggestion = row["AI Suggestion"] # Use AI Suggestion
            recipient = email_map.get(candidate_name)

            if recipient and "@" in recipient:
                # Replace placeholders in the body template
                message = body_template.replace("{{candidate_name}}", candidate_name)
                message = message.replace("{{score}}", f"{score:.2f}") # Include score if desired, but less explicit
                message = message.replace("{{ai_suggestion}}", ai_suggestion) # Include AI suggestion if desired

                if send_email(recipient=recipient, subject=subject, body=message):
                    st.success(f"Email sent successfully to {recipient} ({candidate_name})!")
                    sent_count += 1
                else:
                    st.error(f"Failed to send email to {recipient} ({candidate_name}).")
            else:
                st.warning(f"Skipping email for {candidate_name}: Invalid or missing email address.")
        
        if sent_count > 0:
            st.success(f"✅ Successfully sent {sent_count} email(s).")
        else:
            st.info("No emails were sent. Check recipient addresses.")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("⚠️ No candidates are currently marked as 'shortlisted' in the database. Please screen resumes and mark candidates as shortlisted in the main app to see them here.")
