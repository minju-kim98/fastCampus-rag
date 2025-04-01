import os
import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote.prompts import load_prompt
from langchain_teddynote import logging
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from retriever import create_retriever

load_dotenv()

logging.langsmith("[Project] PDF RAG")

if not os.path.exists('.cache'):
    os.mkdir('.cache')

if not os.path.exists('.cache/embeddings'):
    os.mkdir('.cache/embeddings')

if not os.path.exists('.cache/files'):
    os.mkdir('.cache/files')

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])

@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return create_retriever(file_path)

def create_chain(retriever, model_name="ollama"):
    if model_name == "ollama":
        # 단계 6: 프롬프트 생성(Create Prompt)
        prompt = load_prompt("prompts/pdf-rag-ollama.yaml")

        # 단계 7: 언어모델(LLM) 생성
        llm = ChatOllama(model="EEVE-Korean-Instruct-10.8B:latest", temperature=0)
    elif model_name == "gpt-4o":
        # 단계 6: 프롬프트 생성(Create Prompt)
        prompt = load_prompt("prompts/pdf-rag.yaml")

        # 단계 7: 언어모델(LLM) 생성
        llm = ChatOpenAI(model_name=model_name)

    # 단계 8: 체인(Chain) 생성
    chain = (
            {"context": retriever | format_doc, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain

st.title("Local 모델 기반 RAG")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

with st.sidebar:
    clear_btn = st.button("대화 초기화")
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    selected_model = st.selectbox("LLM 선택", ["ollama", "gpt-4o-mini"], index=0)

if uploaded_file:
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

if clear_btn:
    st.session_state["messages"] = []

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")

warning_msg = st.empty()

if user_input:
    chain = st.session_state["chain"]

    if chain is not None:
        st.chat_message("user").write(user_input)

        response = chain.stream(user_input)
        ai_answer = ""
        with st.chat_message("assistant"):
            container = st.empty()

            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_msg.error("파일을 업로드 해주세요.")