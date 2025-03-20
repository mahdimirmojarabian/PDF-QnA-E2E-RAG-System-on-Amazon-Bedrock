import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config import bedrock_client, LLM_MODEL_ID

llm = Bedrock(
    model_id=LLM_MODEL_ID,
    client=bedrock_client,
    model_kwargs={"temperature": 0.9}
)

def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables=["language", "user_text"],
        template="You are a chatbot. You are in {language}.\n\n{user_text}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain({"language": language, "user_text": user_text})

st.set_page_config(page_title="Bedrock Chatbot Demo", layout="centered")
st.title("🤖 Bedrock Chatbot Demo")

language = st.sidebar.selectbox("Choose Language", [
    "english", "spanish", "hindi", "french", "german",
    "chinese", "japanese", "arabic", "farsi"
])

user_text = st.sidebar.text_area("💬 Ask your question:", max_chars=200)

if user_text:
    with st.spinner("Generating response..."):
        response = my_chatbot(language, user_text)
        st.subheader("📢 Response")
        st.write(response['text'])
