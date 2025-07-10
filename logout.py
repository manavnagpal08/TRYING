import streamlit as st

# --- Logout Style ---
st.markdown("""
<style>
.logout-box {
    margin-top: 4rem;
    padding: 2rem;
    background: linear-gradient(135deg, #fdfbfb, #ebedee);
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    animation: fadeSlideOut 0.8s ease;
}
@keyframes fadeSlideOut {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
.logout-box h2 {
    color: #00cec9;
    font-weight: 700;
}
.logout-box p {
    font-size: 1.1rem;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# --- Logout Logic ---
st.session_state.authenticated = False

# --- Message UI ---
st.markdown("""
<div class="logout-box">
    <h2>🚪 You’ve been logged out!</h2>
    <p>Thank you for using the HR Admin Panel.<br>
    You can close this tab or <a href="/">log in again</a> anytime.</p>
</div>
""", unsafe_allow_html=True)
