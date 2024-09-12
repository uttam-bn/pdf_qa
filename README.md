# PDF Document Q&A with Sentence Transformers and FAISS

This project provides a simple web application using Streamlit that allows users to upload PDF documents and perform Q&A on the document's content using NLP techniques. The application uses Sentence Transformers to create embeddings and FAISS for similarity search.

## Features

- Upload PDF files and extract text.
- Split text into manageable chunks.
- Generate embeddings using `all-MiniLM-L6-v2` model from Sentence Transformers.
- Perform similarity searches using FAISS to find the most relevant text chunks.
- Log questions and answers for review.

## Installation

To run this project, you need to have Python installed along with the following Python libraries:

- Streamlit
- PyPDF2
- Sentence Transformers
- FAISS
- NumPy
- langchain

## Clone this repository:

- https://github.com/uttam-bn/pdf_qa.git
- cd pdf_qa_streamlit_webapp

## Run the Streamlit application:

- streamlit run makethiswork.py

- Open your web browser and navigate to http://localhost:8501 to use the application.

## Usage

- Upload a PDF file through the provided interface.
- Ask questions related to the content of the PDF.
- The application will use sentence embeddings and similarity search to find the most relevant answers based on the document.
