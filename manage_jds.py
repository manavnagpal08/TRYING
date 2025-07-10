import streamlit as st
import os

# --- JD Folder ---
jd_folder = "data"
os.makedirs(jd_folder, exist_ok=True)

# --- UI Styling ---
st.markdown("""
<style>
.manage-jd-container {
    padding: 2rem;
    background: rgba(255, 255, 255, 0.96);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    animation: fadeSlideUp 0.7s ease-in-out;
    margin-bottom: 2rem;
}
@keyframes fadeSlideUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
h3 {
    color: #00cec9;
    font-weight: 700;
}
.upload-box {
    background: #f9f9f9;
    padding: 1rem;
    border-radius: 10px;
    border: 1px dashed #ccc;
}
.select-box, .text-box {
    background: #fff;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="manage-jd-container">', unsafe_allow_html=True)
st.markdown("### 📁 Job Description Manager")

# --- JD Upload ---
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.markdown("#### 📤 Upload New JD (.txt)")
    uploaded_jd = st.file_uploader("Select file", type="txt", key="upload_jd")
    if uploaded_jd:
        jd_path = os.path.join(jd_folder, uploaded_jd.name)
        with open(jd_path, "wb") as f:
            f.write(uploaded_jd.read())
        st.success(f"✅ Uploaded: `{uploaded_jd.name}`")
    st.markdown('</div>', unsafe_allow_html=True)

# --- JD Listing & Viewer ---
jd_files = [f for f in os.listdir(jd_folder) if f.endswith(".txt")]

if jd_files:
    st.markdown('<div class="select-box">', unsafe_allow_html=True)
    selected_jd = st.selectbox("📄 Select JD to view or delete", jd_files)
    st.markdown('</div>', unsafe_allow_html=True)

    if selected_jd:
        with open(os.path.join(jd_folder, selected_jd), "r", encoding="utf-8") as f:
            jd_content = f.read()

        st.markdown('<div class="text-box">', unsafe_allow_html=True)
        st.markdown("#### 📜 Job Description Content")
        st.text_area("View or Copy", jd_content, height=300, key="jd_content", disabled=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"🗑️ Delete `{selected_jd}`"):
                os.remove(os.path.join(jd_folder, selected_jd))
                st.success(f"🗑️ Deleted: `{selected_jd}`")
                st.experimental_rerun()
        with col2:
            st.download_button("⬇️ Download JD", data=jd_content, file_name=selected_jd, mime="text/plain")

        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("📂 No JD files uploaded yet.")

st.markdown('</div>', unsafe_allow_html=True)
