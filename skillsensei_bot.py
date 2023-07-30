import streamlit as st
import openai
import chardet
from PyPDF2 import PdfReader
from secret_keys import openapi_key

# Set your OpenAI API key
openai.api_key = openapi_key

def generate_answer(question, resume_text):
    max_prompt_length = 4096 - len(f"Question: {question}\nResume: ")
    truncated_resume = resume_text[:max_prompt_length].rsplit("\n", 1)[0]
    prompt = f"Question: {question}\nResume: {truncated_resume}\nAnswer:"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=150,
        stop=["\n"]
    )
    answer = response.choices[0].text.strip()
    return answer

def extract_text_from_pdf(file):
    pdf_document = PdfReader(file)
    text = ""
    for page in pdf_document.pages:
        text += page.extract_text()
    return text

def main():
    st.title('Welcome to SkillSensei! 🤖')
    st.write("Interact with SkillSensei, the advanced chatbot capable of analyzing, comparing, and providing answers to any questions related to the uploaded resumes!")

    file = st.file_uploader("Upload Resume", type=["pdf"])

    with st.sidebar:
        st.title('SkillSensei')
        st.subheader("by Krishna Advait Sripada & Tarak Ram")
        st.caption("© 2023 by SkillSensei")
        
    if file is not None:
        resume_text = extract_text_from_pdf(file)

        # Ask questions
        question = st.text_input("Ask a question related to the resume:")
        if question:
            answer = generate_answer(question, resume_text)
            st.text(f"Answer: {answer}")

if __name__ == "__main__":
    main()
