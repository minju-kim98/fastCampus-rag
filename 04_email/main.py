import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import BasePromptTemplate, loading
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper

load_dotenv()

class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    company: str = Field(description="메일을 보낸 사람의 회사 정보")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")

def load_prompt(file_path, encoding="utf8") -> BasePromptTemplate:
    """
    파일 경로를 기반으로 프롬프트 설정을 로드합니다.

    이 함수는 주어진 파일 경로에서 YAML 형식의 프롬프트 설정을 읽어들여,
    해당 설정에 따라 프롬프트를 로드하는 기능을 수행합니다.

    Parameters:
    file_path (str): 프롬프트 설정 파일의 경로입니다.

    Returns:
    object: 로드된 프롬프트 객체를 반환합니다.
    """
    with open(file_path, "r", encoding=encoding) as f:
        import yaml
        config = yaml.safe_load(f)

    return loading.load_prompt_from_config(config)

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def create_email_parsing_chain():
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    output_parser = PydanticOutputParser(pydantic_object=EmailSummary)
    prompt = load_prompt("./prompts/basic_template.yaml")
    prompt = prompt.partial(format=output_parser.get_format_instructions())
    return prompt | llm | output_parser

def create_report_chain():
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    prompt = load_prompt("./prompts/email_search_result.yaml" )

    return prompt | llm | StrOutputParser()

@st.cache_resource(show_spinner="이메일을 분석중입니다...")
def analyse_email(user_input):
    chain = create_email_parsing_chain()

    answer = chain.invoke({"email_conversation": user_input})

    report_chain = create_report_chain()

    params = {"engine": "google", "gl": "kr", "hl": "ko", "num": "3"}
    search = SerpAPIWrapper(params=params)
    query = f"{answer.person} {answer.company} {answer.email}"
    search_result = search.run(query)
    search_result = eval(search_result)
    search_result_string = "\n".join(search_result)

    response = report_chain.invoke(
        {
            "sender": answer.person,
            "additional_information": search_result_string,
            "company": answer.company,
            "email": answer.email,
            "subject": answer.subject,
            "summary": answer.summary,
            "date": answer.date,
        }
    )

    return response


st.title("Email 요약기")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    clear_btn = st.button("대화 초기화")


if clear_btn:
    st.session_state["messages"] = []

print_messages()

user_input = st.chat_input("수신한 이메일을 넣어주세요.")


if user_input:
    st.chat_message("user").write(user_input)

    ai_answer = analyse_email(user_input)

    with st.chat_message("assistant"):
        container = st.empty()
        container.markdown(ai_answer)

    add_message("user", user_input)
    add_message("assistant", ai_answer)