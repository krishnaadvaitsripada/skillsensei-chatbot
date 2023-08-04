import streamlit as st
import openai
import os
from pgml import Database
from PyPDF2 import PdfReader
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 
import asyncio

# Set the OpenAI API Key
openai.api_key = os.environ['OPENAI_API_KEY']

# Set up Zapier API Key
zapier_api_key = os.environ['ZAPIER_API_KEY']

# database
conninfo = os.environ["DATABASE_URL"]
db = Database(conninfo)

collection_name = "candidate_resumes"

# Enable memory for the chatbot
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory import ConversationTokenBufferMemory, ConversationSummaryBufferMemory

llm = ChatOpenAI(temperature=0)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=200)
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=False
)

# Extract and initialize Zapier Agent
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper

zapier = ZapierNLAWrapper(zapier_nla_api_key=zapier_api_key)
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(
    toolkit.get_tools(), OpenAI(temperature=0), 
    agent="zero-shot-react-description", verbose=False
)

def extract_text_from_pdf(file):
    pdf_document = PdfReader(file)
    text = ""
    for page in pdf_document.pages:
        text += page.extract_text()
    return text

def generate_answer(question, resume_texts, database_context):
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
                response should be obtained using {history_answer}. \
                \
                \
                You also have access to all the resumes stored in the database. Here's the information stored in the \
                database: {database_context}. The question asked by the user can refer to the content of the resumes \
                stored in the database as well, in which case use the information provided to you."
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

async def add_to_database_async(collection_to_add, content):
    collection = await db.create_or_get_collection(collection_to_add)
    
    await collection.upsert_documents([content])
    await collection.generate_chunks()
    await collection.generate_embeddings()

def add_to_database(collection_to_add, content):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(add_to_database_async(collection_to_add, content))

async def do_vector_search_async(collection_name, question):
    # Get the collection for vector search
    collection = await db.get_collection(collection_name)
    
    search_results = await collection.vector_search(question, top_k=2)
    
    context = ""
    for result in search_results:
        context += result[1] + "\n"
        
    return context

def do_vector_search(collection_name, question):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(do_vector_search_async(collection_name, question))

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
            
            # add to database
            add_to_database(collection_name, {"filename": file.name, "content": resume_text})
            
            resume_texts.append(resume_text)
        
        # Ask questions
        question = st.text_input("Enter your Question:", key="first_question")
        i = 0
        while True:
            if question:
                answer = generate_answer(question, resume_texts, do_vector_search(collection_name, question))
                memory.save_context({"input": question}, {"output": answer})
                st.text_area("Answer:", value=answer, height=200)
                question = st.text_input("Enter your next question:", key="next_question"+str(i))
                i += 1 # to avoid duplicate keys on the previous line

if __name__ == "__main__":
    main()