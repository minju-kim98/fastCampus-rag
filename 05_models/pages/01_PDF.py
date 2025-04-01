import os
import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote.prompts import load_prompt
from langchain_teddynote import logging
from langchain_core.messages.chat import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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

@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()

    return retriever


def create_chain(retriever, model_name="gpt-4o-mini"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    prompt = load_prompt("prompts/pdf-rag.yaml")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name)

    # 단계 8: 체인(Chain) 생성
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain

st.title("PDF 기반 QA")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

with st.sidebar:
    clear_btn = st.button("대화 초기화")
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini", "o1-mini"], index=1)

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