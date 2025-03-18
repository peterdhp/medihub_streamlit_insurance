from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START, MessagesState
from langgraph.prebuilt.chat_agent_executor import AgentState
import os 
import streamlit as st
from typing import TypedDict, Annotated, List, Union, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage,ToolCall, ToolMessage
from langchain_core.tools import tool
import operator
import json
from langgraph.prebuilt import InjectedState, ToolNode
from collections import defaultdict
from langgraph.graph.message import add_messages
from datetime import datetime
from difflib import get_close_matches
from langgraph.checkpoint.memory import MemorySaver
from utils import render_policy_as_table_actual, render_policy_as_table_flat, process_and_print_active_policies
import unicodedata
import glob


os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']


llm4o = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

llm4omini = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

def purpose_classifier(state):
    chat_history = state['chat_history']
    chat_history_text = "\n".join(f"ai: {msg['content']}" if msg["type"] == "ai" else f"User: {msg['content']}" for msg in chat_history)

    class Purpose_Type(BaseModel):
        purpose: str = Field(
            description="One of: 'Payout Estimate', 'Claim Dispute', 'Medical Support for Claims', 'General Inquiry about enrolled insurance', 'General Inquiry','Unrelated'"
        )
    
    # Define the LLM with structured output
    llm4omini = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    purpose_classifier = llm4omini.with_structured_output(Purpose_Type)
        
    purpose_classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a classifier designed to categorize chat history into one of five specific categories based on the user's intent. This chatbot is used exclusively for questions related to life and health insurance. The categories are:  
    1. **Payout Estimate**: The user is inquiring about an estimated insurance payout for a life or health insurance policy.
    2. **Claim Dispute**: The user has received a payout for a life or health insurance claim but is dissatisfied and requires assistance from a claim adjuster.
    3. **Medical Support for Claims**: The user is seeking medical advice or documentation to support a life or health insurance claim adjustment.
    4. **General Inquiry about enrolled insurance**: The user has a question related to their own life or health insurance that does not fit into the above categories.
    4. **General Inquiry**: The user has a question related to life or health insurance that does not need information about the insurances they are enrolled in.
    5. **Unrelated**: The conversation is not about life or health insurance.
Your task is to read the chat history and return the most appropriate category."""),
        ("user", 
         "[chat history]\n{chat_history}")
    ])
    purpose_classifier = purpose_classifier_prompt | purpose_classifier

    result = purpose_classifier.invoke({'chat_history':chat_history_text})
    purpose = result.purpose
    
    if purpose == "Unrelated":
        return {"non_related" : "F"}
    else :
        return {"purpose": purpose, "non_related" : "T"}
class InsuranceQuery(BaseModel):
    """Represents a pair consisting of an insurance name and a related query."""
    insurance_name: str= Field(
        description="The name of the insurance policy from which information will be retrieved. Without the company name."
    )
    query: str= Field(
        description="A detailed query describing the specific information to be fetched from the terms and conditions."
    )

def retrieve_documents_by_metadata(documents, source=None, page=None):
    """
    Retrieve documents by matching source and topic in metadata.

    :param documents: List of documents loaded from storage.
    :param source: Source file path to match.
    :param topic: Topic to match.
    :return: List of matching documents.
    """
    doclist= [
        doc for doc in documents
        if (source is None or doc['metadata']['source'] == source) and
           (page is None or doc['metadata']['page'] == int(page))
    ]
    return doclist[0]

@tool("fetch_insurance_term_con")
def fetch_insurance_term_con(query : InsuranceQuery, insurance_enrollment_info: Annotated[dict, InjectedState("insurance_enrollment_info")]):
    """Retrieves relevant information from insurance terms and conditions based on a list of queries. 
Each query specifies an 'insurance_name(보험명)' and a 'query' describing the details to be extracted.
The company of the insurance should not be included in the insurance_name.
This is useful for finding context or specific information related to insurance policies."""
    
    #insurance_enrollment_info_text = process_and_print_active_policies(insurance_enrollment_info)
    
    result = []
        
    insurance_name = query.insurance_name
    matching_contract = {}
    insurance_company = ""
    insurance_start_date = ""
    
    
    flatrate_contracts = insurance_enrollment_info.get('data', {}).get('resFlatRateContractList', [])
    actualloss_contracts = insurance_enrollment_info.get('data', {}).get('resActualLossContractList', [])
    all_contracts = flatrate_contracts + actualloss_contracts
    
    #print(all_contracts)
    for contract in flatrate_contracts:
        if insurance_name in contract["resInsuranceName"] :
            matching_contract = contract
            insurance_company = contract["resCompanyNm"]
            insurance_start_date = contract["commStartDate"]
            matching_insurance_text = render_policy_as_table_flat(matching_contract)
            break
        
    for contract in actualloss_contracts : 
        if insurance_name in contract["resInsuranceName"] :
            matching_contract = contract
            insurance_company = contract["resCompanyNm"]
            insurance_start_date = contract["resCoverageLists"][0]["commStartDate"]
            matching_insurance_text = render_policy_as_table_actual(matching_contract)
            break
    #print(insurance_company)
    insurance_company_code_dict = {"메리츠화재보험" : "0101" , "한화손해보험" : "0102", "삼성화재해상보험" : "0108","KB손해보험":"0110", "DB손해보험" : "0111", "캐롯손해보험" :"0000","NH농협손해보험" : "0171", "삼성생명보험" : "0203","현대해상화재보험" : "0109"}
    insurance_company_code = insurance_company_code_dict.get(insurance_company, "Unknown")
    
    
    #print("insurance_company_code is ", insurance_company_code)
    
    
    # insurance_name을 이용해서 insurance_enrollment_info에서 해당 보험에 대한 정보를 가져올 옴
    # 이정보를 통해서 보험사 코드와 보험가입일 가졍오기
    # 이를 이용해서 해당 보험에 대한 보험약관 목차와 본문 json 파일을 가져와서 loaded_toc, loaded_documents에 저장
    query = query.query
    if insurance_company_code != "Unknown":

        with open("documents/contents_json/"+ insurance_company_code +".json", "r", encoding="utf-8") as json_file:
            loaded_company_toc = json.load(json_file)


        matching_items = [item for item in loaded_company_toc if item['name'] == insurance_company]
        
        if matching_items == []:  # If no exact matches found, use difflib for the closest match
            names = [item['name'] for item in loaded_company_toc]
            insurance_name_normalized = unicodedata.normalize('NFC', insurance_name)
            names_normalized = [unicodedata.normalize('NFC', c) for c in names]
            closest_match = get_close_matches(insurance_name_normalized, names_normalized, n=3)
            #print(insurance_name_normalized)
            #print(names_normalized)
            #print(closest_match)

            if closest_match:
                
                matching_items = [item for item in loaded_company_toc if unicodedata.normalize('NFC', item['name']) == closest_match[0]]

        
        if matching_items == []:  # If no valid items found, return None
            return "해당 약관에 대한 정보가 조회 불가능합니다. 약관 정보가 없는 선에서 최대한 답변을 주고 서비스 업데이트를 기다려달라는 안내해줘"
        
        # Filter items with start_date before insurance_start_date

        if len(str(insurance_start_date)) == 6:
            year = int(insurance_start_date[:2])
            current_year = datetime.now().year % 100
            century = "20" if year <= current_year else "19"
            insurance_start_date = century + insurance_start_date
            #print(insurance_start_date)
        valid_items = [
            item for item in matching_items
            if datetime.strptime(item['start_date'], "%y%m%d") < datetime.strptime(str(insurance_start_date), "%Y%m%d")
        ]
        
        
        if not valid_items:  # If no valid items found, return None
            return "해당 약관에 대한 정보가 조회 불가능합니다. 약관 정보가 없는 선에서 최대한 답변을 주고 서비스 업데이트를 기다려달라는 안내해줘"
        
        # Sort by start_date to find the latest one
        matching_item = max(valid_items, key=lambda x: datetime.strptime(x['start_date'], "%y%m%d"))
        #print(matching_item)
        toc_list = matching_item.get("sections", [])
        formatted_toc = "\n".join([f"{item['title']} - Page {item['page']}" for index, item in enumerate(toc_list)])
        
        page_selector_system_prompt = """Given a query and insurance enrollment info, select up to ONLY 10 relevant pages from the terms and conditions.

Key Considerations:
	•	Some policies prohibit duplicate payments
	•	Prioritize sections on : 지급사유, 보험금을 지급하지 않는 사유 etc.
[Insurance enrollment information]
{enroll_info}

[Table of contents] : The Table of Content below only lists the starting page numbers for each section. If you think a section should be selected, please output all the pages.
{table_of_contents}"""

        page_selector_prompt = ChatPromptTemplate.from_messages([
        ("system", page_selector_system_prompt),
        ("user", "query : {query}"),])
        
        class Pagelist(BaseModel):
            """list of page numbers to retrieve information from."""

            page_numbers: list[str] = Field(
            description="list of page numbers to retrieve information from."
        )
            
        structured_pagenum_llm = llm4o.with_structured_output(Pagelist)


        # Combine the prompt and classifier
        page_selector = page_selector_prompt | structured_pagenum_llm

        response = page_selector.invoke({"query" : query , "table_of_contents" : formatted_toc,"enroll_info" :matching_insurance_text})
        pages_to_include = response.page_numbers
        #print(pages_to_include)
            
        
        base_path = f"documents/vector_db/{insurance_company_code}/"

        # Use a wildcard to match files that start with the expected format
        pattern = f"{base_path}{matching_item['start_date']}_{matching_item['name']}*.json"

        # Find matching files
        matching_files = glob.glob(pattern)
        
        #with open("documents/vector_db/"+insurance_company_code+'/'+matching_item["start_date"]+'_'+matching_item['name']+".json", 'r', encoding='utf-8') as f:
        #    loaded_documents = json.load(f)
        if matching_files:
            with open(matching_files[0], 'r', encoding='utf-8') as f:
                loaded_documents = json.load(f)
        
        page_results =[]
        # Fetch and add sections once for each insurance
        for page in pages_to_include:
            matching_doc = retrieve_documents_by_metadata(
                loaded_documents,
                page=page
            )
            page_content = matching_doc.get("page_content", "")
            metadata = matching_doc.get("metadata", "")
            section = metadata.get("topic", "")
            insurance_name = metadata.get("name", "")
            formatted_content = {
                "insurance_name" : insurance_name,
                "query" : query,
                "section" : section,
                "page" : page,
                "content" : page_content
            }
            page_results.append(formatted_content)
        grouped = {}
        for d in page_results:
            key = (d["insurance_name"], d["page"])
            if key not in grouped:
                grouped[key] = {
                    "insurance_name": d["insurance_name"],
                    "page": d["page"],
                    "content": d["content"],  # Assuming content is identical for the same key
                    "sections": set(),
                    "queries": set()
                }
            grouped[key]["sections"].add(d["section"])
            grouped[key]["queries"].add(d["query"])
    
        for key, value in grouped.items():
            value["sections"] = list(value["sections"])  # Convert to list (or ','.join(value["topics"]) for a string)
            value["queries"] = list(value["queries"])  # Convert to list (or ','.join(value["queries"]) for a string)
            result.append(value)

    if result is not [] :
        result = sorted(result, key=lambda x: (x["insurance_name"], x["page"]))

        #f"Insurance Name: {insurance_name}\n"
        # # Combine all results for this insurance name
        # query_result = (f"보험명: {insurance_name}\n"f"Query: {query}")
        # query_result = "\n".join(page_results)
        # insurance_context += '\n\n---\n\n' + query_result #insurance_context를 dict로  
       
    return result


def human_retrieval_node(state):
   """Prompts the user for information. Useful for gathering details directly from the user,
   Especially when clarifying or collecting information related to their health condition.
   This tool should not be used to retrieve contents of the insurance term and conditions."""
   
   for tool_call in state["messages"][-1].tool_calls :
        if tool_call['name'] == "human_retrieval":
            question = tool_call['args']['question']
   
   return {
        "response": question, "end_of_session" : "continue"
    }


@tool("human_retrieval")
def human_retrieval(question : str):
   """Prompts the user for information. Useful for gathering details directly from the user,
   Especially when clarifying or collecting information related to their health condition.
   This tool should not be used to retrieve contents of the insurance term and conditions.
   
   args : 
       question : A question to ask the user.
    """
   
   return ""

def is_about_policy_terms(question :str):
    """
    Classifies whether a given question is about the 보험 약관 (insurance policy terms).

    Parameters:
        question (str): The question to classify.

    Returns:
        dict: A structured response indicating whether the question is about 보험 약관.
    """

    # Define the structured output model
    class PolicyTermsClassification(BaseModel):
        """Binary score for whether the question is about insurance policy terms and conditions."""

        is_about_policy_terms: str = Field(
            description="Answer 'yes' if the question is about insurance policy terms and conditions, otherwise 'no'."
        )


    # Define the LLM with structured output
    
    structured_llm_classifier = llm4omini.with_structured_output(PolicyTermsClassification)

    # Define the system prompt
    policy_terms_system_prompt = """You are a classifier that determines whether a question is asking about insurance policy terms and conditions.

Answer 'yes' if the question pertains to:
1. General policy terms applicable to all insurers.
2. Special terms and conditions, including payout criteria or claim amounts.

Answer 'no' if the question is unrelated to insurance policy terms and conditions."""

    # Define the prompt template
    policy_terms_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", policy_terms_system_prompt),
            ("human", "Question: {question}")
        ]
    )

    # Combine the prompt and classifier
    policy_terms_classifier = policy_terms_prompt | structured_llm_classifier

    # Classify the question
    result = policy_terms_classifier.invoke({"question": question})

    return result





def question_v_retrieval(state):
    """
    Determines whether to retrieve information from user or to use other tool

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    question = state["response"]
    response = is_about_policy_terms(question)
    grade = response.is_about_policy_terms

    if grade == "yes":
        return "END" ### 이후에 다시 수정해야함
    else:
        return "END"
    
def continue_v_error(state):
    """
    Determines whether to continue or return error

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    grade = state["non_related"]
    
    if grade == "T" :
        return "oracle"
    else :
        return "END"
    
@tool("final_answer_general")
def final_answer_general(
    title : str,
    chat_summary : str,
    source : str,
    answer : str
):
    """When enough details are provided, returns a natural language response to the user in the form of a report. 
The tone should be as polite as possible and attentive. Use korean and markdown format for readability.
The arguments recieved are the sections to this report.

    Args :
        title : The title of the report
        chat_summary : A summary of the chat conversation that includes the main question of the user.
        source: A list of referenced insurance clauses, including the policy name, explanation, and specific article and section numbers. Empty string if there are no sources.
        answer : The answer to the user's question
    """
    
    return ""

@tool("final_answer_payoutEstimate")
def final_answer_payoutEstimate(
    title : str,
    chat_summary : str,
    estimate_details : str,
    estimate : str,
    source : str,
    answer : str
):
    """When enough details are provided, returns a natural language response to the user in the form of a report. 
The tone should be as polite as possible and attentive. Use korean and markdown format for readability.
The arguments recieved are the sections to this report.

    Args :
        title : The title of the report
        chat_summary : A summary of the chat conversation that includes the main question of the user.
        estimate_details : A detailed explanation of how the estimate should be calculated. Also information of how it could vary. Be specific, not ambiguous
        estimate : The estimated payout amount of the user's case
        source: A list of referenced insurance clauses, including the policy name, explanation, and specific article and section numbers.
        answer : The answer to the user's question
    """
    
    return ""

@tool("final_answer_dispute")
def final_answer_dispute(
    title : str,
    chat_summary : str,
    dispute_reason : str,
    wanted_outcome : str,
    case_details : str,
    source : str,
    answer : str,
):
    """When enough details are provided, returns a natural language response to the user in the form of a report. 
The tone should be as polite as possible and attentive. Use korean and markdown format for readability.
The arguments recieved are the sections to this report.

    Args :
        title : The title of the report
        chat_summary : A summary of the chat conversation that includes the main question of the user.
        dispute_reason : What the user is dissatisfied with and the reason for the dispute
        wanted_outcome : The desired outcome of the claim of the user
        case_details : case details that are relevant to the dispute. Information that would help the claim adjuster understand the situation.
        source: A list of referenced insurance clauses, including the policy name, explanation, and specific article and section numbers.
        answer : The answer to the user's question
    """
    
    return ""

@tool("final_answer_medicalSupport")
def final_answer_medicalSupport(
    title : str,
    chat_summary : str,
    medical_details: str,
    medical_history : str,
    source : str,
    answer : str,
):
    """When enough details are provided, returns a natural language response to the user in the form of a report. 
The tone should be as polite as possible and attentive. Use korean and markdown format for readability.
The arguments recieved are the sections to this report.

    Args :
        title : The title of the report
        chat_summary : A summary of the chat conversation that includes the main question of the user.
        medical_details : medical details of the case related to the insurance claim adjustment. (diagnostic code, test results, etc.)
        medical_history : medical history of the user that is relevant to the insurance claim adjustment.
        source: A list of referenced insurance clauses, including the policy name, explanation, and specific article and section numbers.
        answer : The answer to the user's question
    """
    
    return ""

def final_answer_node(state):
    """When enough details are provided, returns a natural language response to the user in the form of a report. 
The tone should be as polite as possible and attentive. Use korean and markdown format for readability.
The arguments recieved are the sections to this report.
    """
    args = state["messages"][-1].tool_calls[0]['args']
    response = args["answer"]
    end_of_session_map = {"Payout Estimate" : "estimated_insurance_payout", "Claim Dispute" : "claims_adjuster", "Medical Support for Claims" : "medical_consulation", "General Inquiry" : "general","General Inquiry about enrolled insurance":"general"}
    end_of_session_str = end_of_session_map[state["purpose"]]
    
    
    return {'response' : response, "report" : args, "end_of_session" : end_of_session_str}


def run_oracle(state) :
    purpose = state["purpose"]
    
    purpose_final_answer_map = {"Payout Estimate" : "final_answer_payoutEstimate", "Claim Dispute" : "final_answer_dispute", "Medical Support for Claims" : "final_answer_medicalSupport", "General Inquiry" : "final_answer_general","General Inquiry about enrolled insurance":"final_answer_general"}
    final_answer_str = purpose_final_answer_map[purpose]

    oracle_system_prompt = """You are an insurance consultant. 
    When given insurance enrollment information, answer the user query using your tools. 

    If you need more info, ask the user via the human_retrieval tool. 
    You have full access to insurance policy, terms and conditions(보험약관) yourself through fetch_insurance_term_con to get coverage details from the documents.
    Never tell the user to contact the insurance company or read the insurance policy themselves. 

    Once you have collected plenty of information to answer the user's question use the {final_answer_str} tool. 
    """.format(final_answer_str=final_answer_str)

    oracle_prompt = ChatPromptTemplate.from_messages([
        ("system", oracle_system_prompt),
        ("ai", "먼저 오늘 날짜와 보험 가입 정보를 알려주세요."),
        ("user", "오늘 날짜: {today}\n 보험 가입 정보:\n{insurance_enrollment_info}"),
        ("ai", "알려주신 보험과 관련하여 어떤 것이 궁금하신가요?"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="messages"),
        ])

    if purpose == "Payout Estimate":
        tools=[fetch_insurance_term_con,human_retrieval,final_answer_payoutEstimate]
    if purpose == "Claim Dispute":
        tools=[fetch_insurance_term_con,human_retrieval,final_answer_dispute]
    if purpose == "Medical Support for Claims":
        tools=[fetch_insurance_term_con,human_retrieval,final_answer_medicalSupport]
    if purpose == "General Inquiry":
        tools=[human_retrieval,final_answer_general]
    if purpose == "General Inquiry about enrolled insurance":
        tools=[fetch_insurance_term_con,human_retrieval,final_answer_general]


    oracle = (
        {
            "user_input": lambda x: x["user_input"],
            "chat_history": lambda x: x["chat_history"],
            "today" : lambda x : datetime.today().strftime("%Y.%m.%d"),
            "insurance_enrollment_info": lambda x: process_and_print_active_policies(x["insurance_enrollment_info"]),
            "messages": lambda x: x["messages"],
        }
        | oracle_prompt
        | llm4o.bind_tools(tools, tool_choice="any")
    )
    
    out = oracle.invoke(state)
    
    return {
        "messages": [out]
    }


class State(AgentState):
    user_input: str
    insurance_enrollment_info : dict
    chat_history: list[BaseMessage]
    non_related : str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    response : str
    purpose : str
    report : dict
    end_of_session : str = ""
    

def router(state: list):
    # return the tool name to use
    if isinstance(state["messages"], list):
        for tool_call in state["messages"][-1].tool_calls :
            if tool_call['name'] == "human_retrieval":
                return "human_retrieval"
        if len(state["messages"][-1].tool_calls) == 1 and "final_answer" in state["messages"][-1].tool_calls[0]["name"] : 
            return "final_answer"
        else :
            return "tools"
    else:
        # if we output bad format go to final answer
        #print("Router invalid format")
        return "final_answer"


tools = tools=[fetch_insurance_term_con]
tool_node = ToolNode(tools)

from langgraph.graph import StateGraph, END

graph = StateGraph(State)

graph.add_node("purpose_classifier",purpose_classifier)


graph.add_node("oracle", run_oracle)
graph.add_node("tools", tool_node)
graph.add_node("final_answer", final_answer_node)
graph.add_node("human_retrieval", human_retrieval_node)


graph.add_edge(START,"purpose_classifier")
graph.add_conditional_edges(
    "purpose_classifier",
    continue_v_error,
    {
        "oracle": "oracle",
        "END": END,
    },
)
graph.add_conditional_edges(
    source="oracle",  # where in graph to start
    path=router,  # function to determine which node is called
)
graph.add_conditional_edges(
    "human_retrieval",
    question_v_retrieval,
    {
        "oracle": "oracle",
        "END": END,
    },
)

graph.add_edge("tools","oracle")
# create edges from each tool back to the oracle
#for tool_obj in tools:
#    if tool_obj.name != "final_answer" and tool_obj.name != "human_retrieval":
#        graph.add_edge(tool_obj.name, "oracle")

# if anything goes to final answer, it must then move to END
graph.add_edge("final_answer", END)

insurance_engine = graph.compile()