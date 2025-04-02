import os
import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

logging.langsmith("[Project] 멀티턴 챗봇")

if not os.path.exists('.cache'):
    os.mkdir('.cache')

if not os.path.exists('.cache/embeddings'):
    os.mkdir('.cache/embeddings')

if not os.path.exists('.cache/files'):
    os.mkdir('.cache/files')

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])

def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

def create_chain(model_name="gpt-4o"):
    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요.",
            ),
            # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
        ]
    )

    # llm 생성
    llm = ChatOpenAI(model_name="gpt-4o")

    # 일반 Chain 생성
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )

    return chain_with_history


st.title("대화 내용을 기억하는 챗봇")

with st.sidebar:
    clear_btn = st.button("대화 초기화")
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)
    session_id = st.text_input("세션 ID를 입력하세요.", "abc123")

if clear_btn:
    st.session_state["messages"] = []

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")

warning_msg = st.empty()

if "chain" not in st.session_state:
    st.session_state["chain"] = create_chain(model_name=selected_model)

if user_input:
    chain = st.session_state["chain"]
    if chain:
        response = chain.stream(
            # 질문 입력
            {"question": user_input},
            # 세션 ID 기준으로 대화를 기록합니다.
            config={"configurable": {"session_id": session_id}},
        )
        st.chat_message("user").write(user_input)

        ai_answer = ""
        with st.chat_message("assistant"):
            container = st.empty()

            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

    add_message("user", user_input)
    add_message("assistant", ai_answer)
