import streamlit as st

st.title('Welcome to SkillSensei')
st.write("Upload your resume here")

file = st.file_uploader("Upload Resume")

