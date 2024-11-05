import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_pinecone import PineconeVectorStore as lang_pinecone
from dotenv import load_dotenv
import tempfile
import os
load_dotenv()

def main():
    st.title("Document Upload and Query with RAG")


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "GoogleAI"

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'csv'], accept_multiple_files=True)
        google_api_key = os.getenv("google_api_key")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_INDEX_NAME= os.getenv("PINECONE_INDEX_NAME")

        if not google_api_key or not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
            st.info("Please add your API keys to continue.")
            st.stop()
        
        process = st.button("Process")
        if process:
            pages = get_files_text(uploaded_files)
            st.write("File loaded...")
            if pages:
                st.write(f"Total pages loaded: {len(pages)}")
                text_chunks = get_text_chunks(pages)
                st.write(f"File chunks created: {len(text_chunks)} chunks")
                if text_chunks:
                    vectorstore = get_vectorstore(text_chunks)
                    st.write("Vector Store Created...")
                    st.session_state.conversation = vectorstore
                    st.session_state.processComplete = True
                    st.session_state.session_id = os.urandom(16).hex()  # Initialize a unique session ID
                else:
                    st.error("Failed to create text chunks.")
            else:
                st.error("No pages loaded from files.")

    if st.session_state.processComplete:
        input_query = st.chat_input("Ask Question about your files.")
        if input_query:
            response_text = rag(st.session_state.conversation, input_query, google_api_key)
            st.session_state.chat_history.append({"content": input_query, "is_user": True})
            st.session_state.chat_history.append({"content": response_text, "is_user": False})

            response_container = st.container()
            with response_container:
                for i, message_data in enumerate(st.session_state.chat_history):
                    message(message_data["content"], is_user=message_data["is_user"], key=str(i))


def get_files_text(uploaded_files):
    if not uploaded_files:
        return []
        
    documents = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        try:
            if file_extension == ".pdf":
                loader = PyMuPDFLoader(temp_file_path)
                pages = loader.load()
            elif file_extension == ".docx":
                loader = Docx2txtLoader(temp_file_path)
                pages = loader.load()
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
                pages = loader.load()
            else:
                st.error(f"Unsupported file format: {file_extension}")
                os.remove(temp_file_path)
                continue

            documents.extend(pages)

        except Exception as e:
            st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)
    
    return documents



def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
    vectorstore = lang_pinecone.from_documents(
        text_chunks,
        embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        namespace="Arabic"
    )
    return vectorstore

def get_text_chunks(pages):
    text_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
    for page in pages:
        pg_split = text_splitter.split_text(page.page_content)
        for pg_sub_split in pg_split:
            metadata = {"source": "Arabic", "page_no": page.metadata["page"] + 1}
            doc_string = Document(page_content=pg_sub_split, metadata=metadata)
            text_chunks.append(doc_string)
    return text_chunks


def pinecone_clint():
    embeddings= HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    retriver = PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embeddings ,namespace="Arabic")

    return retriver

vector_db = pinecone_clint()


def rag(vector_db, input_query, google_api_key):
    try:
        template = """You are a multilingual assistant. Analyze the following:
IMPORTANT RULES:
1. ONLY use information present in the context
2. If the information is not in the context, say "I don't find this information in the provided documents" in the same language as the question
3. Do not add any external knowledge or assumptions
4. Match the language of the question in your response:
   - English question → English answer
   - Arabic question → Arabic answer

Question: {question}

Context: {context}

Please provide a clear and accurate response in the same language as the question:"""

        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()})

        model = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=google_api_key)
   
        output_parser = StrOutputParser()
        rag_chain = (
            setup_and_retrieval
            | prompt
            | model
            | output_parser
        )
        response = rag_chain.invoke(input_query)
        return response
    except Exception as ex:
        return str(ex)

if __name__ == '__main__':
    main()
