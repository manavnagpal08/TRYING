import streamlit as st
import json
import os

# --- Styling ---
st.markdown("""
<style>
.notes-container {
    padding: 2rem;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    animation: fadeInSlide 0.6s ease-in-out;
}
.note-box {
    background: #f0f9ff;
    border-left: 4px solid #00cec9;
    padding: 1rem;
    margin-bottom: 1rem;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.03);
}
@keyframes fadeInSlide {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="notes-container">', unsafe_allow_html=True)
st.subheader("📝 Candidate Notes")

notes_file = "notes.json"
if os.path.exists(notes_file):
    with open(notes_file, "r", encoding="utf-8") as f:
        notes = json.load(f)
else:
    notes = {}

candidates = sorted(notes.keys())
selected = st.selectbox("📄 Select Candidate", candidates)

if selected:
    st.markdown(f"#### 🗒️ Notes for {selected}")
    st.markdown('<div class="note-box">', unsafe_allow_html=True)
    text = st.text_area("Edit Note", value=notes[selected], height=150)
    col1, col2 = st.columns(2)
    if col1.button("💾 Save Note"):
        notes[selected] = text
        with open(notes_file, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2)
        st.success("✅ Note updated.")

    if col2.button("🗑️ Delete Note"):
        notes.pop(selected, None)
        with open(notes_file, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2)
        st.warning("🗑️ Note deleted.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.markdown("### ➕ Add New Note")
new_name = st.text_input("👤 Candidate Name")
new_note = st.text_area("📝 Note", height=100)
if st.button("➕ Save New Note"):
    if new_name.strip():
        notes[new_name.strip()] = new_note.strip()
        with open(notes_file, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2)
        st.success(f"✅ Note added for {new_name.strip()}")
        st.rerun()
    else:
        st.error("❌ Candidate name cannot be empty.")

st.markdown('</div>', unsafe_allow_html=True)
