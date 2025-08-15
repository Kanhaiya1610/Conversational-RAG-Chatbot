# app.py - Version for Google Gemini

import streamlit as st
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI # MODIFIED
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. SET UP THE LLM AND EMBEDDING MODEL ---

# Get your Google Gemini API key from the sidebar
# Note: The model name is gemini-1.5-flash, not 2.0
google_api_key = st.sidebar.text_input("Google Gemini API Key", type="password") # MODIFIED

if not google_api_key:
    st.info("Please add your Google Gemini API key to continue.")
    st.stop()

# Initialize the language model (LLM) from Google
# We are using the "gemini-1.5-flash" model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key) # MODIFIED

# Initialize the embedding model (this stays the same)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 2. LOAD DATA AND CREATE THE VECTOR DATABASE (Same as before) ---
@st.cache_resource
def load_data():
    df = pd.read_csv('data/mental_health_faq.csv')
    df['text'] = df['question'] + " " + df['answer']
    return df['text'].tolist()

texts = load_data()

@st.cache_resource
def create_vector_db():
    return Chroma.from_texts(texts=texts, embedding=embedding_function)

vector_db = create_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# --- 3. CREATE THE CONVERSATIONAL RAG CHAIN (Same as before) ---

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """You are a helpful mental health assistant.
Answer the user's question based only on the following context:
{context}
Keep your answers concise and easy to understand. If the context doesn't contain the answer,
say that you don't have enough information to answer."""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

# --- 4. BUILD THE STREAMLIT UI (Same as before) ---

st.title("🧠 Mental Health Support Chatbot (with Gemini)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about stress, anxiety, or sleep..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(
                {"input": prompt, "chat_history": st.session_state.messages}
            )
            st.write(response["answer"])

    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})