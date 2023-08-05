import streamlit as st
import openai
import os
import uuid
import asyncio
from pgml import Database
from PyPDF2 import PdfReader
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())

DATABASE_URL="postgres://u_mzkul3arivqxfbr:dkkspr1yjiyavnn@02f7e6f1-1adb-4347-835a-02c74fcccb0e.db.cloud.postgresml.org:6432/pgml_dsu3bl0afodnrvg"
db = Database(DATABASE_URL)

# Set the OpenAI API Key
openai_api_key = os.environ['OPENAI_API_KEY']
openai.api_key = openai_api_key

# Set up Zapier API Key
zapier_api_key = os.environ['ZAPIER_API_KEY']

# database
#conninfo = os.environ["DATABASE_URL"]
#db = Database(conninfo)

collection_name = "candidate_resumes"

# Enable memory for the chatbot
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
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
    toolkit.get_tools(), OpenAI(temperature=0, openai_api_key=openai_api_key), 
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
    
    await collection.upsert_documents(content)
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

    # Initialize SessionState to store conversation history for each user
    #session_state = st.session_state.get(conversation_history={})
    
    # Initialize conversation history using a dictionary
    #if 'conversation_history' not in st.session_state:
        #st.session_state.conversation_history = {}
        #session_state = st.session_state.conversation_history

    files = st.file_uploader("Upload Resumes (up to 20)", type=["pdf"], accept_multiple_files=True)
    resume_texts = []
    add_to_database_button = st.button("Add to database")
    

    with st.sidebar:
        st.title('SkillSensei')
        st.subheader("by Krishna Advait Sripada & Tarak Ram")
        st.caption("© 2023 by SkillSensei")
        
        st.subheader("Send Interview Invitation")
        email_input = st.text_input("Provide details about the candidate to whom the email needs to be directed, \
            including interview date and time information, along with a concise outline of the job role.", key="email input")
        sender_name = st.text_input("Sender name")
        send_email_button = st.button("Send email")
        
        if send_email_button:
            with st.spinner('Sending email...'):
                agent.run(f"Send a professional email to a candidate via gmail inviting them for an online interview for a job, \
                    along with a google calendar invite for the interview. Details of the candidate, the job, and the interview \
                    are provided by the user. Here's the information provided by the user: {email_input}. Use this input \
                    text to extract the necessary details needed for the email and google calendar invite. \
                    \
                    Make sure that email content contains various paragraphs of clear and elaborated text. The signature should \
                    have the sender's name as {sender_name}. It follow the standard guidelines and format of that of a letter. \
                    Greet the candidate first and then talk about the inviting them for an interview. Have the details \
                    of the interview in a separate paragraph in the email. \
                    \
                    Use the date and time of the interview for creating and sending the Google Calendar event.")
                
                st.success('Email successfully sent!', icon="✅")
            
    if files is not None:
        for file in files:
            resume_text = extract_text_from_pdf(file)
            
            # Generate a unique ID for each file and use it as the key in the database
            file_id = str(uuid.uuid4())  # Use UUID as the unique identifie
            
            # Save the resume text along with its ID for future reference
            resume_texts.append({"file_id": file_id, "resume_text": resume_text, "filename": file.name})

        if add_to_database_button:
            with st.spinner('Adding files to database...'):
                
                add_to_database(collection_name, resume_texts)
                st.success('Files Added!', icon="✅")
                    
        # Ask questions
        questions = st.text_input("Enter your questions (comma-separated):", key="questions")
        if questions:
            questions = [q.strip() for q in questions.split(",")]

            for i, question in enumerate(questions):
                answer = generate_answer(question, [resume_text["resume_text"] for resume_text in resume_texts], 
                                         do_vector_search(collection_name, question))
                # Store the conversation history for each question
                #session_state.conversation_history[question] = answer
                memory.save_context({"input": question}, {"output": answer})
                st.text_area(f"Answer {i+1}:", value=answer, height=200)

if __name__ == "__main__":
    main()