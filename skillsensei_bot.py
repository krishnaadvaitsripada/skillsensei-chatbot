import streamlit as st
import openai
import chardet
import tiktoken
from PyPDF2 import PdfReader
#from secret_keys import openapi_key
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# Set your OpenAI API key
openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings

def generate_answer(question, resume_text):
    max_prompt_length = 4096 - len(f"Question: {question}\nResume: ")
    truncated_resume = resume_text[:max_prompt_length].rsplit("\n", 1)[0]
    prompt = f"Question: {question}\nResume: {truncated_resume}\nAnswer:"

    messages =  [  
        {'role':'system',
        'content':"""You are an assistant by the name of SkillSensei who \
        responds in a formal, professional and polite manner. The user has \
        provided you with a list of PDF documents (either 1 or more) that are resumes of candidates. \
        Here is the text provided in the uploaded PDF document: {truncated_resume} \
        Analyze the resume text provided and answer any queries the user may have pertaining to the contetnt \
        in the document. If the user uses any inappropriate language, please warn them.
        All your responses must be appropriately long and no more than the token limit of {max_prompt_length}."""},    
        {'role':'user',
        'content':"""{question}"""},
    ] 
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        #prompt=prompt
        temperature=0,
        max_tokens=max_prompt_length,
        #stop=["\n"]
    )
    #answer = response.choices[0].text.strip()
    return response.choices[0].message["content"]


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
            #answer = generate_answer(question, resume_text)
            #st.text(f"Answer: {answer}")
            embeddings = OpenAIEmbeddings()
            loader = PyPDFLoader(file)
            docs = loader.load()
            db = DocArrayInMemorySearch.from_documents(
                    docs, 
                    embeddings
            )
            
            index = VectorstoreIndexCreator(
                    vectorstore_cls=DocArrayInMemorySearch,
                    embedding=embeddings
                    ).from_documents([file])
            query ="{question}"
            docs = db.similarity_search(query)
            retriever = db.as_retriever()
            
            llm = ChatOpenAI(temperature = 0.0)
            qdocs = "".join([docs[i].page_content for i in range(len(docs))])
            
            response = llm.call_as_llm(f"{qdocs} Question: {question}") 

            qa_stuff = RetrievalQA.from_chain_type(
                    llm=llm, 
                    chain_type="stuff", 
                    retriever=retriever, 
                    verbose=True
            )
            #response = qa_stuff.run(query)
            response = index.query(query, llm=llm)
            st.text(f"Answer: {response}")

if __name__ == "__main__":
    main()
