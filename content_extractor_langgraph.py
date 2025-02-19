from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState

from typing import Union, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


def content_extractor_langgraph(model,detector_prompt,extractor_prompt):
    
    ### prompt detector
    if detector_prompt == '':
        detector_prompt = """You are an expert in detecting whether a markdown document includes "table of contents".
table of contents normally have the title of the heading followed by the page number. There might be dots or spaces between them.
"""

    llm4omini = ChatOpenAI(model=model, temperature=0)
    
    class toc_TF(BaseModel):
        """Binary score for whether the document includes table of contents."""
        binary_score: str = Field(
            description="Documents include table of contents, 'yes' or 'no'"
        )
    llm4omini = ChatOpenAI(model=model, temperature=0)


    content_detector_LLM = llm4omini.with_structured_output(toc_TF)

    content_detector_prompt = ChatPromptTemplate.from_messages([
        ("system", detector_prompt),
        ("user", "[Document]\n{doc}"),
    ])
    content_detector = content_detector_prompt | content_detector_LLM
    
    def content_extractvEND(state):
        """
        Determines whether the document contains table of contents.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        doc = state['doc']
        response = content_detector.invoke({"doc": doc})
        grade = response.binary_score

        # Check hallucination
        if grade == "yes":
            return "extract"
        else:
            return "END"

    ###prompt extractor
    if extractor_prompt == '':
        extractor_prompt = """You are an expert in extracting "topic" and "page" from markdown documents that contain a "table of contents".
1. The pattern for extracting table of contents and page numbers is expressed using tabs or dots between the table of contents and pages, as shown in the example below.

2. If the page contains a "table of contents", then categorize as follows:
Example :
OOOOOOOOOOOO·························································13\n
Answer : [["OOOOOOOOOOOO",13]]

Example :
OOOOOOOO(OOOOOOOO) ·····································································································49\n
Answer : [["OOOOOOOO(OOOOOOOO)",49]]

Example :
OOOOOOOO(OOOOOOOO)·························································159\n
    OOOOOOOO(OOOOOOOO) ·····································································································164\n
Answer : [["OOOOOOOO(OOOOOOOO)",159], ["OOOOOOOO(OOOOOOOO)",164]]

Example :
OOO OOO OOOO OO OOOOO·························································49\n
    OOO OOO OOOO OO OOOOO ·····································································································67\n
    OOO OOO OOOO OO OOOOO ·······················································································126\n
    OOO OOO OOOO OO OOOOO····································································149\n
Answer : [["OOO OOO OOOO OO OOOOO",49],["OOO OOO OOOO OO OOOOO",67],["OOO OOO OOOO OO OOOOO",126],["OOO OOO OOOO OO OOOOO",149]]

Example :
Ⅱ. 질병 관련 특별약관·······························································98\n
     2-1-1. 갱신형 질병사망보장 특별약관································98\n
     2-1-2. 질병사망보장 특별약관·············································98\n
     2-2-1. 갱신형 질병80%이상후유장해보장 특별약관·······100\n
Answer : [["Ⅱ. 질병 관련 특별약관",98],["2-1-1. 갱신형 질병사망보장 특별약관",98],["2-1-2. 질병사망보장 특별약관",98],["2-2-1. 갱신형 질병80%이상후유장해보장 특별약관",100]]

Example :
비급여 자기공명영상진단(MRI/MRA) 운동위험 특별약관(II)(자동차경기 등)188\n\n장애인전용보험전환 특별약관························································189\n\n별 표·························································································· 195\n\n【별표 1】장해분류표···································································195\n\n【별표 2】해외여행자통지 ···························································231\n\n【별표 3】관련 법규·····································································232\n\n상법····························································································232\n\n전자서명법··················································································233\n\n신용정보의 이용 및 보호에 관한 법률·········································234\n\n
Answer : 
[["비급여 자기공명영상진단(MRI/MRA) 운동위험 특별약관(II)(자동차경기 등)",188],["장애인전용보험전환 특별약관",189],["별 표",195],["【별표 1】장해분류표",195],["【별표 2】해외여행자통지",231],["【별표 3】관련 법규",232],["상법",232],["신용정보의 이용 및 보호에 관한 법률",234]]

Example :
전자서명 특별약관········································································97 화재, 폭발, 파열위험부담보 특별약관···········································98 급배수설비 누출위험부담보 특별약관············································98 건물에 부착된 유리위험부담보 특별약관·······································99 건물위험부담보 특별약관······························································99 지정대리청구서비스 특별약관···································································112 【별표 2】관련 법규·····································································148 상법····························································································148 전자서명법··················································································149 신용정보의 이용 및 보호에 관한 법률·········································150 신용정보의 이용 및 보호에 관한 법률 시행령·····························154 개인정보 보호법···································································163 장애아동 복지지원법···································································165 국가유공자 등 예우 및 지원에 관한 법률····································166 \n목 차 \n유 의\n사\n항 \n가 입\n자 \n요 약\n서 \n주 요\n내\n용 \n해 설 \n보 험\n용\n어 \n주\n요 분 쟁\n사\n례 \n오\n인\n하\n기 쉬 운 \n보 통\n약\n관 \n특 별\n약\n관 \n별 표 \n----- 
Answer : [["전자서명 특별약관",97],["화재, 폭발, 파열위험부담보 특별약관",98],["급배수설비 누출위험부담보 특별약관",98],["건물에 부착된 유리위험부담보 특별약관",99],["건물위험부담보 특별약관",99],["지정대리청구서비스 특별약관",112],["【별표 2】관련 법규",148],["전자서명법",149],["신용정보의 이용 및 보호에 관한 법률",150],["신용정보의 이용 및 보호에 관한 법률 시행령",154],["개인정보 보호법",163],["장애아동 복지지원법",165],["국가유공자 등 예우 및 지원에 관한 법률",166]]


3. If the document does not appear to contain a "table of contents" as it lacks the pattern of topics followed by page numbers, then according to the criteria provided, the response should be: "check_topics_pages":[["None", "None"]]

4. Please exclude items that do not have any meaning in the table of contents.
Example :
유가\n\n의 입\n사 자\n항\n\n\n요주\n\n약 요\n서 내\n\n용\n\n\n해보\n\n설 험\n\n용\n어\n\n\n주\n요\n\n분\n\n쟁\n사\n례\n\n\n오\n인\n하\n기\n\n쉬\n\n운\n\n\n보\n\n통\n약\n관\n\n\n특\n\n별\n약\n관\n\n\n별\n\n표\n\n\n-----\n\n
Answer : [[]]
"""
    
    
    class TopicPageItem(BaseModel):
        """list of extracted topics and pages from table of contents in a markdown document."""
        check_topics_pages: list[list[Union[str, int]]] = Field(
            description="List of [topic, page_number] pairs extracted from table of contents, where topic is a string and page_number is an integer"
        )

    content_extractor_LLM = llm4omini.with_structured_output(TopicPageItem)

    content_extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", extractor_prompt),
        ("user", "[Document]\n{doc}"),
    ])

    content_extractor = content_extractor_prompt | content_extractor_LLM
    
    def content_extractor_node(state):
        """
        extracts table of contents from pages that include table of contents.

        Args:
            state (dict): The current graph state

        Returns:
            list[list[Union[str, int]]]: list of [topic, page_number] pairs extracted from table of contents, where topic is a string and page_number is an integer
        """
        
        doc = state['doc']
        response = content_extractor.invoke({"doc": doc})
        return response
    
    
    class State(AgentState):
        doc :str
        response : list[Any] = []
        
        
    graph = StateGraph(State)



    graph.add_node("content_extractor_chain", content_extractor)

    graph.add_conditional_edges(
        START,
        content_extractvEND,
        {
            "extract": "content_extractor_chain",
            "END": END,
        },
    )
    
    graph.add_edge("content_extractor_chain",END)

    content_extractor_langgraph = graph.compile()
    
    return content_extractor_langgraph
        

        
    
    