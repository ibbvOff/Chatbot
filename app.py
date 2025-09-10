import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import CTransformers
from io import BytesIO
from typing import List
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

# --- If you want to use OpenAI, add the key to .env and uncomment: ---
# from langchain_openai import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings

# --- Model configuration ---
EMBEDDING_MODEL = "hkunlp/instructor-xl"
LLM_MODEL = "maziyarpanahi/Phi-3-mini-4k-instruct-gguf"

# --- OCR: text from image ---
def get_text_from_image(image: Image) -> str:
    return pytesseract.image_to_string(image, lang="eng+ukr")

# --- OCR: text from PDF (if needed) ---
def get_text_from_pdf(file) -> str:
    text = ""
    try:
        if hasattr(file, "read"):
            file_bytes = file.read()
            file.seek(0)
        else:
            file_bytes = file

        pdf_reader = PdfReader(BytesIO(file_bytes))
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

        if len(text.strip()) < 100:
            filename = getattr(file, "name", "PDF-file")
            st.info(f"Little text found in '{filename}'. Starting OCR...")
            text = ""
            images = convert_from_bytes(file_bytes)
            for i, image in enumerate(images):
                st.write(f"Processing page {i+1} ({filename}) via OCR...")
                text += get_text_from_image(image) + "\n"

    except Exception as e:
        filename = getattr(file, "name", "PDF-file")
        st.error(f"Error reading '{filename}': {e}")
        return ""

    return text

# --- Breaking into chunks ---
def get_text_chunks(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# --- Embeddings --- 
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# --- Vector store ---
def get_vectorstore(text_chunks: List[str], embeddings) -> FAISS:
    if not text_chunks:
        return None
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# --- LLM ---
@st.cache_resource
def load_llm():
    return CTransformers(
        model=LLM_MODEL,
        model_type="llama",
        config={'context_length': 4096, 'max_new_tokens': 1024, 'temperature': 0.7}
    )

# --- Functions for OpenAI (if needed)

# @st.cache_resource
# def load_embedding_openai():
#     return OpenAIEmbeddings(
#         model="text-embedding-3-small",
#         openai_api_key=os.getenv("OPENAI_API_KEY")
#     )

# @st.cache_resource
# def load_llm_openai():
#     return ChatOpenAI(
#         model="gpt-4o-mini",  # або інша модель з OpenAI
#         temperature=0.7,
#     )

# --- Chain ---
def get_conversation_chain(vectorstore: FAISS, llm) -> ConversationalRetrievalChain:
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        memory=memory
    )

# --- Processing user queries ---
def handle_userinput(user_question: str):
    if not st.session_state.conversation:
        st.warning("Please upload and process documents.")
        return
    if not user_question.strip():
        return

    with st.spinner("Thinking..."):
        response = st.session_state.conversation.invoke({'question': user_question})

    st.session_state.chat_history = response['chat_history']

    for msg in st.session_state.chat_history:
        template = user_template if msg.type == "human" else bot_template
        st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

# --- Main program ---
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Local HuggingFace embedding chunks
    embeddings = load_embedding_model()

    # --- If you have an OpenAI key and want to use OpenAI embeddings: ---
    # embeddings = load_embedding_openai()  # <-- uncomment for OpenAI

    # Local model via CTransformers
    llm = load_llm()

    # --- If you have an OpenAI key and want to work via OpenAI: ---
    # llm = load_llm_openai()  # <-- uncomment for OpenAI

    st.header("Chat with your documents :books:")

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, PNG, JPG and click 'Process'",
            accept_multiple_files=True,
            type=["pdf", "png", "jpg", "jpeg"]
        )

        if st.button("Process"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Processing documents..."):
                    raw_text = ""
                    for file in uploaded_files:
                        ext = file.name.split('.')[-1].lower()
                        if ext == "pdf":
                            raw_text += get_text_from_pdf(file)
                        elif ext in ["png", "jpg", "jpeg"]:
                            st.write(f"Processing image '{file.name}' through OCR...")
                            image = Image.open(file)
                            raw_text += get_text_from_image(image)
                    
                    if not raw_text.strip():
                        st.error("Failed to extract text from the provided files.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks, embeddings)
                        if vectorstore:
                            st.session_state.conversation = get_conversation_chain(vectorstore, llm)
                            st.session_state.chat_history = []
                            st.success("Processing complete! You can now ask questions.")
                        else:
                            st.error("Failed to create vector store.")

        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.session_state.conversation = None
            st.rerun()

    # chat
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            template = user_template if msg.type == "human" else bot_template
            st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

    user_question = st.chat_input("Ask something about your documents...")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
