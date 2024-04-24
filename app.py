import streamlit as st
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_community.vectorstores import FAISS

# Setup API Key (if using Langchain genai models)
with open(r'C:\Users\Dell\Downloads\RAG System\google_api.txt', 'r') as f:
    GOOGLE_API_KEY = f.read()

def process_pdf(pdf_path):
    """
    Processes the PDF, extracting text and performing basic pre-processing.
    """
    text = ""
    try:
        loader = PyPDFLoader(pdf_path)
        text = loader.load_and_split()
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

    
    return text

def split_and_embed_documents(text):
    """
    Splits the text into chunks.
    """
    # Tokenize and split into chunks
    text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(text)

    # Ensure that each chunk is a string
    chunks = [str(chunk) for chunk in chunks]

    return chunks

def main():
    """
    Main function for running the application.
    """
    st.title("Explore 'Leave No Context Behind' with RAG")
    st.markdown("<h2 style='color:Purple;'>Ask a Question</h2>", unsafe_allow_html=True)

    # Optional: Load Langchain GenAI model (replace with your model name)
    chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-pro-latest")

    # Load the "Leave No Context Behind" paper (replace with your PDF path)
    pdf_path = r"C:\Users\Dell\Downloads\RAG System\2404.07143.pdf"  # Replace with your PDF path
    
    # Add loading icon while processing PDF
    with st.spinner("Loading PDF..."):
        text = process_pdf(pdf_path)

    if text:
        # Split the text into chunks
        with st.spinner("Splitting text..."):
            chunks = split_and_embed_documents(text)

        # Define the chat prompt template
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="This system uses the 'Leave No Context Behind' paper to answer your questions."),
            HumanMessagePromptTemplate.from_template("""Answer the question based on the information in the 'Leave No Context Behind' paper and the context provided.
            Context:
            {context}
            
            Question: 
            {question}
            
            Answer: """)
        ])

        # Initialize output parser
        output_parser = StrOutputParser()

        # Define the core Langchain retrieval pipeline
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}  # Using RunnablePassthrough for now
            | chat_template
            | (chat_model if GOOGLE_API_KEY else SystemMessage(content="Chat model not loaded due to missing API key"))
            | output_parser
        )

        # User input and response generation
        user_question = st.text_input("Enter your question here:")
        if st.button("Submit", key="submit_button"):
            if user_question:
                with st.spinner("Generating response..."):
                    response = rag_chain.invoke({"context": chunks, "question": user_question})
                st.markdown(f"**Answer:**\n {response}")

if __name__ == "__main__":
    main()
