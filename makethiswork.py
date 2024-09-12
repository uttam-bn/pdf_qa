import streamlit as st
from PyPDF2 import PdfReader
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
#import cohere
import textwrap
import faiss
import os
import numpy as np  


from sentence_transformers import SentenceTransformer

ai_output = ""
total_chunks = 0


st.write("PDF document Q&A")


# Use SentenceTransformer for embeddings 
embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')  

# Upload PDF File
uploaded_file = st.file_uploader("Select a PDF file")

if uploaded_file is not None:
    st.subheader("Upload your PDF file, please ensure it is not confidential content")
    reader = PdfReader(uploaded_file)
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    st.write("The file is ready to be processed.")

    # Split the text into manageable chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    total_chunks += 1
    st.write("File is split into chunks: ", len(texts), total_chunks)

    # Create embeddings using sentence-transformers
    embeddings = [embeddings_model.encode(text) for text in texts]
    
    # Convert the list of embeddings to a NumPy array
    embeddings_array = np.array(embeddings)
    
    
    index = faiss.IndexFlatL2(embeddings_array.shape[1])  
    index.add(embeddings_array)  

    st.caption("FAISS index created with sentence-transformers embeddings.")

    # Function to perform a similarity search
    def similarity_search(query):
        query_vector = embeddings_model.encode([query])  
        query_vector = np.array(query_vector)  
        D, I = index.search(query_vector, k=5)  
        return [texts[i] for i in I[0]]  

    # Create or open the log file
    log_file = 'qa_log.txt'
    if not os.path.exists(log_file):
        open(log_file, 'w').close()  

    # Function to generate output from a user prompt
    def generate_output(user_prompt):
        relevant_docs = similarity_search(user_prompt)

        combined_text = "\n".join(relevant_docs)
        shortened_answer = textwrap.shorten(combined_text, width=300, placeholder="...")

    # Format the final answer
        answer = f"Answer to the question '{user_prompt}' based on the document:\n{shortened_answer}"

    # Log the question and answer
        with open(log_file, "a") as f:
            f.write(f"Question: {user_prompt}\nAnswer: {answer}\n\n")

        return answer

    # Display the form for user to ask a question
    st.subheader("Ask your question")

    form = st.form(key="user_settings")
    with form:
        user_input = st.text_input("Enter your question:")
        generate_button = form.form_submit_button("Submit Question")

        if generate_button:
            if user_input == "":
                st.error("Question cannot be blank.")
            else:
                st.write("Answer:")
                ai_output = generate_output(user_input)
                st.write(ai_output)

    # Option to display the log file
    st.sidebar.subheader("Log of Questions and Answers")
    if st.sidebar.button("Show Log"):
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                log_content = f.read()
                st.sidebar.text_area("Q&A Log", log_content, height=300)
        else:
            st.sidebar.write("No log file found.")

# Display end note
st.write('')
st.markdown("Document issues resolved.")
