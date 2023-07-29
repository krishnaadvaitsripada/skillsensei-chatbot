import streamlit as st

st.title('Welcome to SkillSensei!')
st.write("Interact with SkillSensei, the advanced chatbot capable of analyzing, comparing, and providing answers to any questions related to the uploaded resumes!")

file = st.file_uploader("Upload Resume")

