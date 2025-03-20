import os
import shutil
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from config import bedrock_client, EMBEDDING_MODEL_ID, LLM_MODEL_ID

prompt_template = """
Human: Using the provided context, craft a concise response to the question at the end. Ensure the summary is at least 250 words and includes detailed explanations. If the answer is unknown, simply state that you do not know and avoid creating a speculative response.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

bedrock_embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID, client=bedrock_client)

@st.cache_data
def process_uploaded_pdfs(uploaded_files):
    temp_dir = "uploaded_data"
    os.makedirs(temp_dir, exist_ok=True)

    for file in uploaded_files:
        with open(os.path.join(temp_dir, file.name), "wb") as f:
            f.write(file.read())

    docs = []
    for file in os.listdir(temp_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(temp_dir, file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = splitter.split_documents(docs)

    shutil.rmtree(temp_dir)
    return chunks

@st.cache_resource
def load_vector_store():
    return FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

def get_vector_store(docs):
    vectorstore = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore.save_local("faiss_index")

def get_llm():
    return Bedrock(model_id=LLM_MODEL_ID, client=bedrock_client, model_kwargs={"max_gen_len": 512})

def get_response_llm(llm, vectorstore, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa({"query": query})

def handle_sidebar():
    st.title("📂 Upload PDFs & Create Vector Store")
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("📥 Process & Store"):
        with st.spinner("Processing PDFs..."):
            docs = process_uploaded_pdfs(uploaded_files)
            get_vector_store(docs)
            st.success("✅ Vector store created!")

def handle_query():
    query = st.text_input("💬 Ask a question about the PDFs")

    if st.button("Send"):
        if not os.path.exists("faiss_index"):
            st.error("⚠️ Vector store not found. Please upload PDFs and click 'Process & Store' first.")
            return

        with st.spinner("Generating answer..."):
            vectorstore = load_vector_store()
            llm = get_llm()
            result = get_response_llm(llm, vectorstore, query)

            st.subheader("📢 Answer")
            st.write(result["result"])

            with st.expander("🔍 Retrieved Context"):
                for doc in result["source_documents"]:
                    st.markdown(doc.page_content)

def main():
    st.set_page_config(page_title="PDF QnA - RAG on Bedrock", layout="wide")
    st.header("📄 PDF-based Question Answering with Amazon Bedrock")
    handle_sidebar()
    handle_query()

if __name__ == "__main__":
    main()
