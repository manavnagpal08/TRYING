import streamlit as st
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

# --- Load Shortlisted Candidates from Session State ---
shortlisted_df = pd.DataFrame() # Initialize as empty DataFrame

# Check if 'screening_results_df' exists in session state
if 'screening_results_df' in st.session_state and not st.session_state.screening_results_df.empty:
    # Filter the DataFrame for shortlisted candidates
    # Ensure 'shortlisted' column exists and is boolean
    if 'shortlisted' in st.session_state.screening_results_df.columns:
        shortlisted_df = st.session_state.screening_results_df[st.session_state.screening_results_df['shortlisted'] == True].copy()
        
        # Rename columns for display and consistency with template
        # Ensure these columns exist in your session_state DataFrame
        shortlisted_df = shortlisted_df.rename(columns={
            'predicted_score': 'Score (%)',
            'years_experience': 'Years Experience',
            'ai_suggestion': 'AI Suggestion',
            'candidate_name': 'Candidate Name', # Ensure this is consistent
            'email': 'Email' # Ensure this is consistent
        })
    else:
        st.warning("The 'shortlisted' column was not found in the screening results. Please ensure it's set in the main app.")
else:
    st.info("No screening results found in session state. Please screen resumes in the main app first.")


if not shortlisted_df.empty:
    st.success(f"✅ {len(shortlisted_df)} candidates currently shortlisted.")
    
    # Display the shortlisted candidates
    # Ensure these columns exist in your shortlisted_df after renaming
    display_columns = ["Candidate Name", "Score (%)", "Years Experience", "AI Suggestion", "Email"]
    # Filter display_columns to only include those actually in the DataFrame
    display_columns = [col for col in display_columns if col in shortlisted_df.columns]
    
    st.dataframe(shortlisted_df[display_columns], use_container_width=True)

    st.markdown("<div class='email-box'>", unsafe_allow_html=True)

    st.markdown("### ✉️ Assign Emails")
    # Pre-fill emails if available from the DataFrame
    email_map = {row['Candidate Name']: row['Email'] if pd.notna(row['Email']) and row['Email'] != 'Not Found' else '' for _, row in shortlisted_df.iterrows()}

    for i, row in shortlisted_df.iterrows():
        candidate_name = row["Candidate Name"]
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
            candidate_name = row["Candidate Name"]
            score = row.get("Score (%)", "N/A") # Use .get() for robustness
            ai_suggestion = row.get("AI Suggestion", "No specific AI suggestion.") # Use AI Suggestion
            recipient = email_map.get(candidate_name)

            if recipient and "@" in recipient:
                # Replace placeholders in the body template
                message = body_template.replace("{{candidate_name}}", candidate_name)
                message = message.replace("{{score}}", str(score))
                message = message.replace("{{ai_suggestion}}", ai_suggestion)

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
    st.warning("⚠️ No candidates are currently marked as 'shortlisted' in the session state. Please screen resumes and mark candidates as shortlisted in the main app to see them here.")

