import streamlit as st

st.title('Welcome to SkillSensei! 🤖')
st.write("Interact with SkillSensei, the advanced chatbot capable of analyzing, comparing, and providing answers to any questions related to the uploaded resumes!")

file = st.file_uploader("Upload Resume")

with st.sidebar:
    st.title('SkillSensei')
    st.subheader("by Krishna Advait Sripada & Tarak Ram")
    st.caption("© 2023 by SkillSensei")
        
input_msg = st.text_input("Ask a question!")