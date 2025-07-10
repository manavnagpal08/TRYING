import streamlit as st
import pdfplumber
import re
import pandas as pd
import io

# --- Styling ---
st.markdown("""
<style>
.search-box {
    padding: 2rem;
    margin-top: 1rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.95);
    box-shadow: 0 8px 30px rgba(0,0,0,0.07);
    animation: slideFade 0.6s ease-in-out;
}
.result-box {
    background: #f7faff;
    padding: 1.2rem;
    margin-bottom: 1.2rem;
    border-radius: 14px;
    border-left: 4px solid #00cec9;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    animation: fadeInResult 0.6s ease;
}
.highlight {
    background-color: #ffeaa7;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
}
@keyframes slideFade {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInResult {
    0% { opacity: 0; transform: scale(0.98); }
    100% { opacity: 1; transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# --- UI Header ---
st.markdown('<div class="search-box">', unsafe_allow_html=True)
st.subheader("🔍 Resume Search Engine")
st.caption("Upload resumes and search for single or multiple keywords (e.g., `python, sql`).")

# --- File Upload ---
resumes = st.file_uploader("📤 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True, key="resume_search_upload")
resume_texts = {}

if resumes:
    st.success(f"✅ {len(resumes)} resume(s) uploaded.")
    for resume in resumes:
        try:
            with pdfplumber.open(resume) as pdf:
                text = ''.join(page.extract_text() or '' for page in pdf.pages)
                resume_texts[resume.name] = text
        except Exception as e:
            st.warning(f"⚠️ Error reading {resume.name}")

    query = st.text_input("🔎 Enter keywords (comma-separated)").strip().lower()
    download_rows = []

    if query:
        keywords = [q.strip() for q in query.split(',') if q.strip()]
        st.markdown("### 📄 Search Results")
        found = False

        for name, content in resume_texts.items():
            content_lower = content.lower()
            matched_snippets = []
            for keyword in keywords:
                if keyword in content_lower:
                    found = True
                    idx = content_lower.find(keyword)
                    snippet = content[max(0, idx - 40): idx + 160]
                    highlighted = re.sub(
                        f"({re.escape(keyword)})",
                        r"<span class='highlight'>\1</span>",
                        snippet,
                        flags=re.IGNORECASE
                    )
                    matched_snippets.append(highlighted)

            if matched_snippets:
                combined_snippet = " ... ".join(matched_snippets)
                st.markdown(f"""<div class="result-box">
                <b>📄 {name}</b><br>{combined_snippet}...
                </div>""", unsafe_allow_html=True)

                download_rows.append({
                    "File Name": name,
                    "Matched Keywords": ", ".join(keywords),
                    "Snippet": ' '.join(snippet for snippet in matched_snippets)
                })

        if not found:
            st.error("❌ No matching resumes found.")

        # --- Export Button ---
        if download_rows:
            df_download = pd.DataFrame(download_rows)
            csv_buffer = io.StringIO()
            df_download.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Matched Results (CSV)", data=csv_buffer.getvalue(), file_name="matched_resumes.csv", mime="text/csv")

else:
    st.info("📁 Please upload resume PDFs to begin searching.")

st.markdown("</div>", unsafe_allow_html=True)
