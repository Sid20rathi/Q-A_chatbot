import streamlit as st
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import time
import requests

st.title("RAG Document Q&A")


llm = ChatOllama(model="deepseek-r1:7b")


prompt = ChatPromptTemplate.from_template("""
   Answer the following question based on the provided context only.
   Provide the most accurate response.

   <context>
   {context}
   </context>

   Question: {input}
""")

pdf_file = st.file_uploader("Upload a PDF file", type=['pdf'])

def create_vector_embedding():
  
    if "vectors" not in st.session_state and pdf_file is not None:
      

        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())

   
        st.session_state.embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
        st.session_state.loader = PyPDFLoader("temp.pdf")  
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, embedding=st.session_state.embeddings)
        st.success("Embedding completed successfully!")

user_prompt = st.text_input("Enter your query from the PDF")

if st.button("Document Embedding"):
    create_vector_embedding()

if user_prompt and "vectors" in st.session_state:
    retriever = st.session_state.vectors.as_retriever()
    
    document_chain = create_stuff_documents_chain(llm, prompt=prompt)
    
    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    st.write(f"Response Time: {time.process_time() - start:.2f} seconds")

    st.write(response['answer'])

    with st.expander("More Details"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---------------")
