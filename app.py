import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-small")

st.set_page_config(page_title="Personal PDF Chatbot", page_icon="📄", layout="centered")
st.title("Personal PDF Chatbot")
st.write(
    "Upload a PDF and ask questions. Answers are generated from the text in that PDF only."
)

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def load_generation_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_length=256,
        truncation=True,
    )

@st.cache_data(show_spinner=False)
def load_pdf_documents(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    return loader.load()


def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = load_embedding_model()
    return Chroma.from_documents(chunks, embeddings, collection_name="personal_pdf")


def generate_answer(question: str, docs, top_k: int = 3) -> str:
    context = "\n\n---\n\n".join([doc.page_content for doc in docs[:top_k]])
    prompt = (
        "You are an assistant that answers questions using only the provided context. "
        "If the answer is not present in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    llm = load_generation_pipeline()
    result = llm(prompt, max_length=256, do_sample=False)
    return result[0]["generated_text"].strip()

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
question = st.text_input("Ask a question about the document", "")

if uploaded_file:
    with st.spinner("Loading PDF and building the local vector store..."):
        documents = load_pdf_documents(uploaded_file.read())
        if not documents:
            st.warning("No text was extracted from the PDF.")
        else:
            vector_store = build_vector_store(documents)
            st.success("PDF loaded. Ask a question now.")
            if question:
                with st.spinner("Searching the PDF and generating an answer..."):
                    results = vector_store.similarity_search(question, k=3)
                    answer = generate_answer(question, results)
                    st.subheader("Answer")
                    st.write(answer)
                    st.subheader("Retrieved chunks")
                    for i, chunk in enumerate(results, start=1):
                        excerpt = chunk.page_content.strip().replace("\n", " ")
                        st.write(f"**Chunk {i}:** {excerpt[:500]}{'...' if len(excerpt) > 500 else ''}")
else:
    st.info("Upload a PDF file to start.")
