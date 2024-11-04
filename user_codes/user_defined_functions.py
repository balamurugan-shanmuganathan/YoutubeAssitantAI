from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import streamlit as st
import os
from dotenv import load_dotenv

from IPython.display import display
from IPython.display import Markdown
import textwrap

def to_markdown(text):
  text = text.replace(".", "*")
  return Markdown(textwrap.indent(text,'> ', predicate = lambda _:True))


def llm_model():  
    # LLM Model
    os.environ["GOOGLE_API_KEY"] =GOOGLE_API_KEY
    google_llm = ChatGoogleGenerativeAI(model='gemini-1.0-pro')
    return google_llm


def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, hf_embeddings)
    return db 


def get_response_from_query(db, query, k=4):

    docs = db.similarity_search(query, k = k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = llm_model()

    prompt = ChatPromptTemplate.from_template("""
      You are a helpful youtube assistant that can answer questions about videos based on the video's transcript:

      Answer the following question: {query}
      By searching the following video transcript: {docs}

      Only use the factual information from the transcript to answer the question.

      If you feel like you don't have enough information to answer the question, say "Hmm, I'm not sure".

      Your answers should be verbose and detailed.

    """)
    chain = (
        {"query": RunnablePassthrough(), "docs": RunnableLambda(lambda x: docs_page_content)}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = chain.invoke(query)
    return response

