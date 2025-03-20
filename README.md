# 📄 PDF-QnA-E2E-RAG-System-on-Amazon-Bedrock

An end-to-end **Retrieval-Augmented Generation (RAG)** system built on **Amazon Bedrock** to answer natural language questions from PDF documents. This project extracts relevant content from PDFs and uses foundation models to generate accurate, context-aware responses.

---

## 🚀 Features

✅ **Intelligent PDF Q&A** – Extracts and retrieves relevant content dynamically  
✅ **Amazon Bedrock Integration** – Uses state-of-the-art foundation models  
✅ **Efficient Semantic Search** – FAISS-based vector storage for fast retrieval  
✅ **Streamlit UI** – Interactive demo for seamless user experience  

---

## 🏗️ System Architecture

Below is the high-level architecture of the system:

![RAG Architecture](images/rag_architecture.png)

### 🔹 Workflow:
1. **PDF Upload** → Extract text using `PyPDF`  
2. **Chunking** → Break text into meaningful sections using `LangChain`  
3. **Embedding** → Convert text chunks into vector representations (`Amazon Titan Embeddings`)  
4. **Vector Store** → Store embeddings using `FAISS`  
5. **Retriever** → Fetch the most relevant chunks  
6. **Bedrock LLM** → Generate accurate, context-aware responses  
7. **Streamlit UI** → Interactive interface for user queries  

---

## 🛠️ Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/mahdimirmojarabian/PDF-QnA-E2E-RAG-System-on-Amazon-Bedrock.git
cd PDF-QnA-E2E-RAG-System-on-Amazon-Bedrock
```

### 2️⃣ Set Up a Virtual Environment

```bash
python -m venv rag_bedrock
```

Activate the environment:

- **Linux/macOS**:

  ```bash
  source rag_bedrock/bin/activate
  ```

- **Windows**:

  ```bash
  rag_bedrock\Scripts\activate
  ```

### 3️⃣ Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

---

## 🔐 AWS Configuration

### 4️⃣ Install AWS CLI

Follow the official guide to install the AWS CLI:  
👉 [AWS CLI Installation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

### 5️⃣ Configure AWS Credentials

Run:

```bash
aws configure
```

Provide your **AWS Access Key ID, Secret Access Key, Region, and Output Format**. Make sure your IAM role has access to **Amazon Bedrock** and related services.

---

## ⚙️ Environment Configuration

This project uses a `.env` file to manage environment-specific configurations. These variables are loaded automatically using `python-dotenv` in the `config.py` file.

### 📦 What These Variables Do

| Variable             | Description                                                                                         |
|----------------------|-----------------------------------------------------------------------------------------------------|
| `AWS_REGION`         | Your default AWS region where Bedrock is available                                                  |
| `EMBEDDING_MODEL_ID` | The ID of the embedding model used to convert PDF chunks into vectors (Titan, etc.)                 |
| `LLM_MODEL_ID`       | The foundation model ID used to generate context-aware responses via Bedrock (LLaMA2, Claude, etc.) |

---

## 🎯 Running the Application

### Streamlit Demos

You can run two different Streamlit applications:

#### 🧪 Bedrock Connectivity Test (Simple Chatbot)

Run this to verify your Bedrock setup using a simple LLM chat interface:

```bash
streamlit run bedrock_test.py
```

#### 🚀 Main RAG Application (PDF-Based Q&A)

This runs the complete Retrieval-Augmented Generation system for PDF-based Q&A:

```bash
streamlit run rag_demo.py
```

---

## 📚 Use Cases

🔹 **Legal & Compliance** – Search contracts and policies for key insights  
🔹 **Customer Support** – Automate responses from FAQs and support docs  
🔹 **Research & Academics** – Extract relevant information from scientific papers  
🔹 **Enterprise Knowledge Base** – Access company documents with natural language queries  

---

## 🧠 Powered By

- **Amazon Bedrock (LLMs & Embeddings)**  
- **LangChain for Retrieval-Augmented Generation (RAG)**  
- **FAISS (Vector Database)**  
- **Streamlit (UI & Demos)**  
- **Python 3.9+**

---

## 🤝 Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📬 Contact

For questions, suggestions, or collaborations, reach out via:  
🔗 **[LinkedIn](https://www.linkedin.com/in/m-mahdi-mir)**
