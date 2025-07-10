from email_sender import send_email_to_candidate

send_email_to_candidate(
    name="TestUser",
    score=95,
    feedback="Excellent candidate!",
    recipient="motimahalreception@gmail.com",
    subject="Test Email from Resume Screener",
    message="Hi, this is a test email to check SMTP works!"
)
