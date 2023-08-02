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

def generate_answer(question, resume_text):
    max_prompt_length = 4096 - len(f"Question: {question}\nResume: ")
    truncated_resume = resume_text[:max_prompt_length].rsplit("\n", 1)[0]
    history_answer = conversation.predict(input=question)
    messages = [
        {
            'role': 'system',
            'content': f"You are an assistant by the name of SkillSensei who \
                responds in a formal, professional, and polite manner. The user has \
                provided you with a list of PDF documents (either 1 or more) that are resumes of candidates. \
                Here is the text provided in the uploaded PDF document: {truncated_resume} \
                Analyze the resume text provided and answer any queries the user may have pertaining to the content \
                in the document. If the user uses any inappropriate language, please warn them. \
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

    file = st.file_uploader("Upload Resume", type=["pdf"])

    with st.sidebar:
        st.title('SkillSensei')
        st.subheader("by Krishna Advait Sripada & Tarak Ram")
        st.caption("© 2023 by SkillSensei")

    if file is not None:
        resume_text = extract_text_from_pdf(file)
        
        # Ask questions
        question = st.text_input("Enter your Question:", key="first_question")
        i = 0
        while True:
            if question:
                answer = generate_answer(question, resume_text)
                memory.save_context({"input": question}, {"output": answer})
                st.text_area("Answer:", value=answer, height=200)
                question = st.text_input("Enter your next question:", key="next_question"+str(i))
                i+=1 # to avoid duplicate keys on the previous line

if __name__ == "__main__":
    main()

