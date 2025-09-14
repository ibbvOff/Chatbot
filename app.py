import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from htmlTemplates import css, bot_template, user_template
from io import BytesIO
from typing import List
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import os
import nest_asyncio

# nest_asyncio.apply() is necessary for environments where an event loop is already running.
nest_asyncio.apply()

# --- Configuration of local models ---
LOCAL_EMBEDDING_MODEL = "hkunlp/instructor-xl"
LOCAL_LLM_MODEL = "maziyarpanahi/Phi-3-mini-4k-instruct-gguf"
MIN_TEXT_LENGTH_FOR_OCR_FALLBACK = 100

# --- OCR: Extracting text from images ---
def get_text_from_image(image: Image) -> str:
    return pytesseract.image_to_string(image, lang="eng+ukr")

# --- File Processing: Extracting text from PDF (with fallback to OCR) ---
def get_text_from_pdf(file) -> str:
    text = ""
    try:
        file_bytes = file.read()
        file.seek(0)
        pdf_reader = PdfReader(BytesIO(file_bytes))
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
        if len(text.strip()) < MIN_TEXT_LENGTH_FOR_OCR_FALLBACK:
            filename = getattr(file, "name", "PDF-file")
            st.info(f"Little text found in '{filename}'. Starting OCR...")
            text = ""
            images = convert_from_bytes(file_bytes)
            for i, image in enumerate(images):
                st.write(f"Processing page {i+1} ('{filename}') with OCR...")
                text += get_text_from_image(image) + "\n"
    except Exception as e:
        filename = getattr(file, "name", "PDF-file")
        st.error(f"Error reading file '{filename}': {e}")
        return ""
    return text

# --- Text Chunking ---
def get_text_chunks(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# --- Vector Store Creation ---
def get_vectorstore(text_chunks: List[str], embeddings) -> FAISS:
    if not text_chunks:
        return None
    try:
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Unable to create vector storage: {e}")
        return None

# --- Model Loading ---

@st.cache_resource
def load_embedding_local():
    st.info(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)

@st.cache_resource
def load_llm_local():
    st.info(f"Loading local LLM: {LOCAL_LLM_MODEL}")
    return CTransformers(
        model=LOCAL_LLM_MODEL,
        model_type="llama",
        config={'context_length': 4096, 'max_new_tokens': 1024, 'temperature': 0.7}
    )

@st.cache_resource
def load_embedding_google():
    st.info("Loading embedding model from Google...")
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

@st.cache_resource
def load_llm_google():
    st.info("Loading LLM from Google (Gemini)...")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )

# --- Creating a conversation chain ---
def get_conversation_chain(vectorstore: FAISS, llm):
    # 1. Prompt for creating a standalone question based on history
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # 2. A chain that creates an independent question and passes it to the retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, vectorstore.as_retriever(search_kwargs={'k': 3}), contextualize_q_prompt
    )

    # 3. Prompt for providing an answer based on the retrieved documents
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 4. A chain that combines the documents into a single string
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 5. Final chain that brings everything together
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# --- User Input Handling ---
def handle_userinput(user_question: str):
    if not st.session_state.get("conversation_runnable"):
        st.warning("Please upload and process documents.")
        return
    if not user_question.strip():
        return

    with st.spinner("Thinking..."):
        # Call the new chain
        response = st.session_state.conversation_runnable.invoke(
            {'input': user_question},
            config={"configurable": {"session_id": "default_session"}}
        )
    
    # The history is now updated automatically, we just need to receive a response.
    st.session_state.chat_history = st.session_state.store.get("default_session")
    # The response is now located in the 'answer' key
    # bot_response_content = response.get('answer', 'Unfortunately, an error occurred.')

# --- Main application function ---
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Your Documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation_runnable" not in st.session_state:
        st.session_state.conversation_runnable = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "store" not in st.session_state:
        st.session_state.store = {} # Storage for chat history

    st.header("Chat with Your Documents :books:")

    # --- Sidebar ---
    with st.sidebar:
        st.subheader("Settings")
        model_provider = st.radio(
            "Select model provider:",
            ("Google (Gemini)", "Local (Phi-3)")
        )
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader(
            "Upload files and click 'Process'",
            accept_multiple_files=True,
            type=["pdf", "png", "jpg", "jpeg"]
        )

        if st.button("Process"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Processing documents..."):
                    # 1. Text extraction
                    raw_text = ""
                    for file in uploaded_files:
                        ext = file.name.split('.')[-1].lower()
                        if ext == "pdf":
                            raw_text += get_text_from_pdf(file)
                        elif ext in ["png", "jpg", "jpeg"]:
                            st.write(f"Processing image '{file.name}' with OCR...")
                            image = Image.open(file)
                            raw_text += get_text_from_image(image)
                    
                    if not raw_text.strip():
                        st.error("Failed to extract text from the provided files.")
                    else:
                        # 2. Model loading
                        try:
                            if model_provider == "Google (Gemini)":
                                if not os.getenv("GOOGLE_API_KEY"):
                                    st.error("Google API key not found.")
                                    return
                                embeddings = load_embedding_google()
                                llm = load_llm_google()
                            else:
                                embeddings = load_embedding_local()
                                llm = load_llm_local()
                        except Exception as e:
                            st.error(f"Error loading models: {e}")
                            return

                        # 3. Creating vector store and chain
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks, embeddings)
                        
                        if vectorstore:
                            rag_chain = get_conversation_chain(vectorstore, llm)

                            # Function to get message history from our store
                            def get_session_history(session_id: str):
                                if session_id not in st.session_state.store:
                                    st.session_state.store[session_id] = ChatMessageHistory()
                                return st.session_state.store[session_id]

                            # Wrap the chain for automatic history management
                            st.session_state.conversation_runnable = RunnableWithMessageHistory(
                                rag_chain,
                                get_session_history,
                                input_messages_key="input",
                                history_messages_key="chat_history",
                                output_messages_key="answer",
                            )
                            st.session_state.chat_history = []
                            st.session_state.store = {}
                            st.success("Processing complete!")
                        else:
                            st.error("Failed to create vector store.")

        if st.button("Clear chat"):
            st.session_state.conversation_runnable = None
            st.session_state.chat_history = []
            st.session_state.store = {}
            st.rerun()

    # --- Main chat area ---

    user_question = st.chat_input("Ask something about your documents...")
    if user_question:
        handle_userinput(user_question)

    if st.session_state.chat_history:
        for msg in st.session_state.chat_history.messages:
            template = user_template if msg.type == "human" else bot_template
            st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

if __name__ == '__main__':
    main()