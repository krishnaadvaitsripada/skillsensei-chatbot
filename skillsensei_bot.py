import streamlit as st
import openai
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 

# Set the OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory import ConversationTokenBufferMemory, ConversationSummaryBufferMemory

llm = ChatOpenAI(temperature=0.0)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=200)
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=False
)

def extract_text_from_pdf(file):
    pdf_document = PdfReader(file)
    text = ""
    for page in pdf_document.pages:
        text += page.extract_text()
    return text

def generate_answer(question, resume_texts):
    max_prompt_length = 4096 - len(f"Question: {question}\nResumes: ")
    truncated_resumes = " ".join(resume[:max_prompt_length] for resume in resume_texts)
    history_answer = conversation.predict(input=question)
    messages = [
        {
            'role': 'system',
            'content': f"You are an assistant by the name of SkillSensei who \
                responds in a formal, professional, and polite manner. The user has \
                provided you with {len(resume_texts)} PDF documents that are resumes of candidates. \
                Here are the first few words of each uploaded resume: {truncated_resumes} \
                Analyze the resume texts provided and answer any queries the user may have pertaining to the content \
                in the documents. If the user uses any inappropriate language, please warn them. \
                All your responses must be appropriately long and no more than the token limit of {max_prompt_length}. \
                If any of the user's questions appear to reference a prior conversation, the appropriate \
                response should be obtained using {history_answer}"
        },
        {
            'role': 'user',
            'content': question,
        },
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    return response.choices[0].message["content"]

def main():
    st.title('Welcome to SkillSensei! 🤖')
    st.write("Interact with SkillSensei, the advanced chatbot capable of analyzing, comparing, and providing answers to any questions related to the uploaded resumes!")

    files = st.file_uploader("Upload Resumes (up to 20)", type=["pdf"], accept_multiple_files=True)
    resume_texts = []

    with st.sidebar:
        st.title('SkillSensei')
        st.subheader("by Krishna Advait Sripada & Tarak Ram")
        st.caption("© 2023 by SkillSensei")

    if files is not None:
        for file in files:
            resume_text = extract_text_from_pdf(file)
            resume_texts.append(resume_text)

        # Ask questions
        question = st.text_input("Enter your Question:", key="first_question")
        i = 0
        while True:
            if question:
                answer = generate_answer(question, resume_texts)
                memory.save_context({"input": question}, {"output": answer})
                st.text_area("Answer:", value=answer, height=200)
                question = st.text_input("Enter your next question:", key="next_question"+str(i))
                i += 1 # to avoid duplicate keys on the previous line

if __name__ == "_main_":
    main()