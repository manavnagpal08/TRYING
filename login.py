import streamlit as st

# Hardcoded username and password
USERNAME = "admin"
PASSWORD = "1234"  # Change to something secure

def login_section():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form("login_form", clear_on_submit=False):
            st.subheader("🔐 HR Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                if username == USERNAME and password == PASSWORD:
                    st.session_state.authenticated = True
                    st.success("✅ Login successful!")
                else:
                    st.error("❌ Invalid credentials")
    return st.session_state.authenticated
