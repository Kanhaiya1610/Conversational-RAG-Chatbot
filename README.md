# 🧠 Mental Health Support Chatbot
- This project is a mental health support chatbot built with Streamlit, LangChain, and Google's Gemini 1.5 Flash model. It uses a Retrieval-Augmented Generation (RAG) architecture to provide helpful and informative responses to user queries related to mental health.

## Overview
- The chatbot is designed to answer questions about stress, anxiety, sleep, and other mental health topics. It uses a vector database to store and retrieve relevant information from a knowledge base of frequently asked questions (FAQs). The Gemini 1.5 Flash model is then used to generate a conversational and contextually appropriate response based on the retrieved information.

## Technology Stack
- Framework: Streamlit

- LLM Orchestration: LangChain

- Language Model: Google Gemini 1.5 Flash

- Vector Database: ChromaDB

- Embeddings: Hugging Face Sentence Transformers

## Setup and Run Locally
- Follow these steps to set up and run the chatbot on your local machine.

- 1. Clone the Repository
```
git clone <your-repository-url>
cd <repository-folder>
2. Create and Activate a Virtual Environment
Create the environment:
python -m venv venv
```
Activate it:

On Windows:

```
.\venv\Scripts\activate
```
On macOS/Linux:

```
source venv/bin/activate
```
- 3. Prepare the requirements.txt File
-- Ensure your requirements.txt file has the following content:

- streamlit
- langchain
- langchain-community
- langchain-google-genai
- chromadb
- sentence-transformers
- pandas
- 
- 4. Install Dependencies
Run the following command in your terminal (with the virtual environment active) to install all the necessary libraries:

```
pip install -r requirements.txt
```
- 5. Get Your Google Gemini API Key
You will need an API key to use the Gemini model.

Obtain your key from Google AI Studio.

You will paste this key into the chatbot's sidebar when you run the application.

## 6. Run the App
- Launch the Streamlit application by running the following command from your project's root directory:
```
streamlit run app.py
```
## Your chatbot will now be running locally in your web browser! Use my Gemini API key to test the App. ( AIzaSyC0Y5PsYXVqK4liAgMlI82BQ6-RzkvCcjQ )

# Data Source
- The chatbot uses a CSV file named mental_health_faq.csv located in the data/ directory. This file contains the questions and answers that form the chatbot's knowledge base. You can easily modify or extend this file to customize the chatbot's knowledge.

# Contributing
- Contributions to this project are welcome! Feel free to submit pull requests with bug fixes, new features, or improvements to the documentation.
