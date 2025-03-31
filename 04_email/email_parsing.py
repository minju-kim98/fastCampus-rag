import yaml
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import PromptTemplate, BasePromptTemplate, loading
from dotenv import load_dotenv

load_dotenv()

email_conversation = """From: 테디 (teddy@teddynote.com)
To: 이은채 대리님 (eunchae@teddyinternational.me)
Subject: RAG 솔루션 시연 관련 미팅 제안

안녕하세요, 이은채 대리님,

저는 테디노트의 테디입니다. 최근 귀사에서 AI를 활용한 혁신적인 솔루션을 모색 중이라는 소식을 들었습니다. 테디노트는 AI 및 RAG 솔루션 분야에서 다양한 경험과 노하우를 가진 기업으로, 귀사의 요구에 맞는 최적의 솔루션을 제공할 수 있다고 자부합니다.

저희 테디노트의 RAG 솔루션은 귀사의 데이터 활용을 극대화하고, 실시간으로 정확한 정보 제공을 통해 비즈니스 의사결정을 지원하는 데 탁월한 성능을 보입니다. 이 솔루션은 특히 다양한 산업에서의 성공적인 적용 사례를 통해 그 효과를 입증하였습니다.

귀사와의 협력 가능성을 논의하고, 저희 RAG 솔루션의 구체적인 기능과 적용 방안을 시연하기 위해 미팅을 제안드립니다. 다음 주 목요일(7월 18일) 오전 10시에 귀사 사무실에서 만나 뵐 수 있을까요?

미팅 시간을 조율하기 어려우시다면, 편하신 다른 일정을 알려주시면 감사하겠습니다. 이은채 대리님과의 소중한 만남을 통해 상호 발전적인 논의가 이루어지길 기대합니다.

감사합니다.

테디
테디노트 AI 솔루션팀"""

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
        config = yaml.safe_load(f)

    return loading.load_prompt_from_config(config)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
output_parser = PydanticOutputParser(pydantic_object=EmailSummary)
prompt = load_prompt("./prompts/basic_template.yaml")

# format 에 PydanticOutputParser의 부분 포맷팅(partial) 추가
prompt = prompt.partial(format=output_parser.get_format_instructions())

chain = prompt | llm | output_parser

answer = chain.invoke({"email_conversation": email_conversation})

print(answer.summary)
print('-------------------------------------')
print(answer)
print('-------------------------------------')

params = {"engine": "google", "gl": "kr", "hl": "ko", "num": "3"}

search = SerpAPIWrapper(params=params)
query = f"{answer.person} {answer.company} {answer.email}"
search_result = search.run(query)
search_result = eval(search_result)
search_result_string = "\n".join(search_result)

report_prompt = load_prompt("./prompts/email_search_result.yaml")

report_chain = (
    report_prompt | ChatOpenAI(model="gpt-4-turbo", temperature=0) | StrOutputParser()
)

report_response = report_chain.invoke(
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

print(report_response)