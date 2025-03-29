import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_teddynote.prompts import load_prompt
from langchain import hub

load_dotenv()

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def create_chain(prompt_file_path, task=""):
    prompt = load_prompt(prompt_file_path)
    if task:
        prompt = prompt.partial(task=task)

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain

st.title("나만의 챗GPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    clear_btn = st.button("대화 초기화")

    import glob

    prompt_files = glob.glob("./prompts/*.yaml")
    prompt_names = []
    for prompt_file in prompt_files:
        prompt_names.append(prompt_file.split("\\")[1].replace(".yaml", ""))

    selected_prompt = st.selectbox(
        "프롬프트를 선택해 주세요.",
        prompt_names,
        index=0
    )

    task_input = st.text_input("TASK 입력", "")

if clear_btn:
    st.session_state["messages"] = []

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.chat_message("user").write(user_input)
    chain = create_chain(f"./prompts/{selected_prompt}.yaml", task=task_input)
    response = chain.stream({"question": user_input})
    ai_answer = ""
    with st.chat_message("assistant"):
        container = st.empty()

        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # ai_answer = chain.invoke({"question": user_input})
    # st.chat_message("assistant").write(ai_answer)

    add_message("user", user_input)
    add_message("assistant", ai_answer)