# Import necessary libraries and packages
import os
import config
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from flask import Flask, render_template, request

# Define the directory where PDF files are stored
pdf_directory = config.pdf_directory

# Get a list of all PDF files in the directory
pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Load PDF files using the UnstructuredPDFLoader from langchain
data = []
for pdf_file in pdf_files:
    loader = UnstructuredPDFLoader(pdf_file)
    data.extend(loader.load())

# Print some information about the loaded data
print(f'You have {len(data)} document(s) in your data')
print(f'There are {len(data[0].page_content)} characters in your document(s)')

# Split the loaded data into smaller chunks using the RecursiveCharacterTextSplitter from langchain
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# Print some information about the split data
print(f"Now you have {len(texts)} document(s) after splitting into chunks")

# Set up OpenAI API key and create OpenAIEmbeddings instance
OPENAI_API_KEY = config.OPENAI_API_KEY
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create a vector store using Chroma from langchain
vectorstore = Chroma("langchain_store", embeddings)

# Set up index name and create Chroma instance using from_texts method
index_name = "langchain_store"
docsearch = Chroma.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

# # Create OpenAI instance and load question answering chain from langchain
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

