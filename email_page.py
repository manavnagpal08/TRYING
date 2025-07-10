# 📤 Email Candidates Page UI Enhancer
import streamlit as st
import pandas as pd
from email_sender import send_email_to_candidate

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

try:
    df = pd.read_csv("results.csv")
except FileNotFoundError:
    st.warning("⚠️ `results.csv` not found. Run Resume Screener first.")
    st.stop()

shortlisted = df[(df["Score (%)"] >= 80) & (df["Years Experience"] >= 2)]

if not shortlisted.empty:
    st.success(f"✅ {len(shortlisted)} candidates shortlisted.")
    st.dataframe(shortlisted[["File Name", "Score (%)", "Years Experience", "Feedback"]])

    st.markdown("<div class='email-box'>", unsafe_allow_html=True)

    st.markdown("### ✉️ Assign Emails")
    email_map = {}
    for i, row in shortlisted.iterrows():
        file_name = row["File Name"]
        with st.container():
            st.markdown(f"<div class='email-entry'>", unsafe_allow_html=True)
            email_input = st.text_input(f"📧 Email for {file_name}", key=f"email_{i}")
            email_map[file_name] = email_input
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📝 Customize Email Template")
    subject = st.text_input("Subject", value="🎉 You're Shortlisted!")
    body_template = st.text_area("Body", height=180, value="""
Hi,

Congratulations! 🎉

You've been shortlisted for the next phase based on your resume review.

✅ Score: {{score}}%
💬 Feedback: {{feedback}}

We'll be in touch with next steps shortly.

Warm regards,  
HR Team
""")

    if st.button("📤 Send All Emails"):
        sent = 0
        for _, row in shortlisted.iterrows():
            name = row["File Name"]
            score = row["Score (%)"]
            feedback = row["Feedback"]
            recipient = email_map.get(name)

            if recipient and "@" in recipient:
                message = body_template.replace("{{score}}", str(score)).replace("{{feedback}}", feedback)
                send_email_to_candidate(
                    name=name,
                    score=score,
                    feedback=feedback,
                    recipient=recipient,
                    subject=subject,
                    message=message
                )
                sent += 1
        st.success(f"✅ {sent} email(s) sent successfully.")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("⚠️ No shortlisted candidates with score ≥ 80 and experience ≥ 2 yrs.")
