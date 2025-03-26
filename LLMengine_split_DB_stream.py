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
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from collections import defaultdict
from langgraph.graph.message import add_messages
from datetime import datetime, date
from difflib import get_close_matches
from langgraph.checkpoint.memory import MemorySaver
from utils import render_policy_as_table_actual, render_policy_as_table_flat, process_and_print_active_policies
import unicodedata
import glob
import logging
import socket
import pymysql
import pandas as pd


os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']


conn = st.connection("sql")
    

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
        return {"non_related" : "F", "specified_date" : datetime.today().strftime("%Y%m%d")}
    else :
        return {"purpose": purpose, "non_related" : "T", "specified_date" : datetime.today().strftime("%Y%m%d")}
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

@tool("fetch_insurance_enrollment_info")
def fetch_insurance_enrollment_info(
    specified_date: str,
    insurance_enrollment_info: Annotated[dict, InjectedState("insurance_enrollment_info")],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """Fetches the user's insurance enrollment information for a specified date and updates the session state with that date.

    Usage:
      - If no specific date is provided, defaults to today's date for general insurance inquiries.
      - For insurance claims, set `specified_date` based on the type of claim:
        • 실손 의료비 (Actual Medical Expenses): 영수증 수납일 (Receipt Payment Date)
        • 암 진단비 (Cancer Diagnosis Benefit): 진단서 발급일 (Diagnosis Certificate Issuance Date)
        • 질병 일당 (Daily Disease Allowance): 진단서 발급일 (Diagnosis Certificate Issuance Date)
        • 후유장해 (Permanent Disability Benefit): 후유장해진단서 발급일 (Permanent Disability Diagnosis Certificate Issuance Date)
        • 사망보험금 (Death Benefit): 사망진단서 발급일 (Death Certificate Issuance Date)

    Args:
        specified_date (str): Date string in YYYYMMDD format. If omitted, today's date is used.
        insurance_enrollment_info (dict): A dictionary containing the user's full insurance
                                          enrollment details (e.g., coverage start/end dates,
                                          policy types, etc.).
        tool_call_id (str): Unique identifier for this tool call.

    Returns:
        Command: Updates the session state with the new `specified_date` and returns
                 a message containing the relevant insurance enrollment information.
"""
    
    result = process_and_print_active_policies(insurance_enrollment_info,specified_date)
    return Command(
        update={
            # update the state keys
            "specified_date": specified_date,
            # update the message history
            "messages": [ToolMessage("insurance enrollment information of the user at {} : \n".format(datetime.strptime(specified_date, "%Y%m%d").strftime("%Y년 %m월 %d일"))+ result, tool_call_id=tool_call_id)]
        }
    )

# Setup logging just for insurance terms
logger = logging.getLogger('insurance_terms')
# Only configure the logger if it hasn't been configured yet
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('insurance_terms.log')
    console_handler = logging.StreamHandler()
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

@tool("fetch_insurance_term_con")
def fetch_insurance_term_con(query : InsuranceQuery, insurance_enrollment_info: Annotated[dict, InjectedState("insurance_enrollment_info")],specified_date : Annotated[str, InjectedState("specified_date")] ):
    """Retrieves relevant information from insurance terms and conditions based on a list of queries. 
Each query specifies an exact 'insurance_name(보험명)' from the insurance_enrollment_info and a 'query' describing the details to be extracted.
The company of the insurance should not be included in the insurance_name.
This is useful for finding context or specific information related to insurance policies."""
    
    try:
        logger.info(f"Starting fetch_insurance_term_con with insurance name: {query.insurance_name}")
        result = []
            
        insurance_name = query.insurance_name
        matching_contract = {}
        insurance_company = ""
        insurance_start_date = ""
        
        flatrate_contracts = insurance_enrollment_info.get('data', {}).get('resFlatRateContractList', [])
        actualloss_contracts = insurance_enrollment_info.get('data', {}).get('resActualLossContractList', [])
        logger.info(f"Found {len(flatrate_contracts)} flatrate and {len(actualloss_contracts)} actualloss contracts")
        
        # Contract matching logs
        for contract in flatrate_contracts:
            if insurance_name in contract["resInsuranceName"]:
                matching_contract = contract
                insurance_company = contract["resCompanyNm"]
                insurance_start_date = contract["commStartDate"]
                logger.info(f"Matched flatrate contract: {insurance_company} - {insurance_name}")
                matching_insurance_text = render_policy_as_table_flat(matching_contract,specified_date)
                break
            
        for contract in actualloss_contracts:
            if insurance_name in contract["resInsuranceName"]:
                matching_contract = contract
                insurance_company = contract["resCompanyNm"]
                insurance_start_date = contract["resCoverageLists"][0]["commStartDate"]
                logger.info(f"Matched actualloss contract: {insurance_company} - {insurance_name}")
                matching_insurance_text = render_policy_as_table_actual(matching_contract,specified_date)
                break
        
        if not matching_contract:
            logger.warning(f"No matching contract found for insurance name: {insurance_name}")
                
        df_1 = conn.query("SELECT * FROM insurance_company")
        insurance_company_id_dict = dict(zip(df_1['name'], df_1['id']))
        insurance_company_id = insurance_company_id_dict.get(insurance_company, "Unknown")
        logger.info(f"Insurance company ID lookup: {insurance_company} -> {insurance_company_id}")
        
        query = query.query
        if insurance_company_id != "Unknown":
            # Date processing
            if len(str(insurance_start_date)) == 6:
                year = int(insurance_start_date[:2])
                current_year = datetime.now().year % 100
                century = "20" if year <= current_year else "19"
                insurance_start_date = century + insurance_start_date
            
            
             
            # Convert insurance_start_date string to datetime.date object
            try:
                insurance_start_date = datetime.strptime(insurance_start_date, "%Y%m%d").date()
                logger.info(f"Converted insurance_start_date to date object: {insurance_start_date}")
            except ValueError as e:
                logger.error(f"Error converting insurance_start_date: {e}")
                return "해당 약관에 대한 정보가 조회 불가능합니다. 약관 정보가 없는 선에서 최대한 답변을 주고 서비스 업데이트를 기다려달라는 안내해줘"
            
            # Term lookup
            df_2 = conn.query(f"SELECT * FROM insurance_term WHERE COMPANY_ID = '{insurance_company_id}'")
            logger.info(f"Found {len(df_2)} terms for company ID {insurance_company_id}")
            logger.info(f"DataFrame columns: {df_2.columns.tolist()}")
            
            # Ensure start_date column is datetime.date type
            if not isinstance(df_2['start_date'].iloc[0], date):
                df_2['start_date'] = pd.to_datetime(df_2['start_date']).dt.date
                logger.info("Converted df_2 start_date column to date objects")
            
            
            valid_dates = df_2[df_2['start_date'] < insurance_start_date]
            logger.info(f"Found {len(valid_dates)} valid terms before {insurance_start_date}")
            
            if valid_dates.empty:
                logger.warning(f"No valid terms found for company ID {insurance_company_id} before {insurance_start_date}")
                return "해당 약관에 대한 정보가 조회 불가능합니다. 약관 정보가 없는 선에서 최대한 답변을 주고 서비스 업데이트를 기다려달라는 안내해줘"
            
            exact_matches = valid_dates[valid_dates['name'] == insurance_name]
            
            if exact_matches.empty:
                logger.info("No exact matches found, attempting fuzzy matching")
                names = valid_dates['name'].tolist()
                insurance_name_normalized = unicodedata.normalize('NFC', insurance_name)
                names_normalized = [unicodedata.normalize('NFC', name) for name in names]
                closest_matches = get_close_matches(insurance_name_normalized, names_normalized, n=1, cutoff=0.8)
                
                if closest_matches:
                    matching_name = names[names_normalized.index(closest_matches[0])]
                    matching_rows = valid_dates[valid_dates['name'] == matching_name]
                    logger.info(f"Found fuzzy match: {matching_name}")
                else:
                    logger.warning(f"No fuzzy matches found for insurance name: {insurance_name}")
                    return "해당 약관에 대한 정보가 조회 불가능합니다. 약관 정보가 없는 선에서 최대한 답변을 주고 서비스 업데이트를 기다려달라는 안내해줘"
            else:
                matching_rows = exact_matches
                logger.info(f"Found {len(matching_rows)} exact matches for insurance name: {insurance_name}")
            
            latest_term = matching_rows.loc[matching_rows['start_date'].idxmax()]
            term_id = latest_term['id']
            logger.info(f"Selected term ID: {term_id} with start date: {latest_term['start_date']}")
            
            # Contents processing
            df_3 = conn.query(f"SELECT * FROM insurance_contents WHERE TERM_ID = '{term_id}'")
            logger.info(f"Found {len(df_3)} content entries for term ID {term_id}")
            logger.info(f"DataFrame columns in df_3: {df_3.columns.tolist()}")
            
            # Modify this part to use the correct column names from your database
            toc_list = df_3[['title', 'page']].to_dict('records')
            formatted_toc = "\n".join([f"{item['title']} - {item['page']}" for item in toc_list])
            
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
                
            # Query the database for rows matching the term_id and pages
            page_conditions = " OR ".join([f"PAGE = '{page}'" for page in pages_to_include])
            df_4 = conn.query(f"SELECT * FROM insurance_term_detail WHERE TERM_ID = '{term_id}' AND ({page_conditions})")
            
            page_results = []
            # Fetch and add sections from the database
            for _, row in df_4.iterrows():
                formatted_content = {
                    "insurance_name": latest_term['name'],  # Using name from latest_term
                    "query": query,
                    "section": row['topic'].split(','),  # Split TOPIC on commas into a list
                    "page": row['page'],
                    "content": row['page_content']
                }
                page_results.append(formatted_content)

            grouped = {}
            for d in page_results:
                key = (d["insurance_name"], d["page"])
                if key not in grouped:
                    grouped[key] = {
                        "insurance_name": d["insurance_name"],
                        "page": d["page"],
                        "content": d["content"],
                        "sections": set(),
                        "queries": set()
                    }
                # Convert the list to a tuple before adding to set
                grouped[key]["sections"].add(tuple(d["section"]))
                grouped[key]["queries"].add(d["query"])
        
            for key, value in grouped.items():
                # Convert the set of tuples back to a list of lists
                value["sections"] = [list(section) for section in value["sections"]]
                value["queries"] = list(value["queries"])
                result.append(value)

        if result != []:
            result = sorted(result, key=lambda x: (x["insurance_name"], x["page"]))
            logger.info(f"Successfully processed {len(result)} results")

    except Exception as e:
        logger.error(f"Error in fetch_insurance_term_con: {str(e)}", exc_info=True)
        return "해당 약관에 대한 정보가 조회 불가능합니다. 약관 정보가 없는 선에서 최대한 답변을 주고 서비스 업데이트를 기다려달라는 안내해줘"
    return result


def human_retrieval_node(state):
   """Prompts the user for information. Useful for gathering details directly from the user,
   Especially when clarifying or collecting information related to their health condition.
   It is IMPORTANT that the tool is not used to retrieve contents of the insurance term and conditions."""
   
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
   It is IMPORTANT that the tool is not used to retrieve contents of the insurance term and conditions.
   
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
        source: A list of referenced insurance clauses, including the policy name, explanation, specific article and section numbers, and exact quotes from the relevant sections.
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
        source: A list of referenced insurance clauses, including the policy name, explanation, specific article and section numbers, and exact quotes from the relevant sections.
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
        source: A list of referenced insurance clauses, including the policy name, explanation, specific article and section numbers, and exact quotes from the relevant sections.
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
        source: A list of referenced insurance clauses, including the policy name, explanation, specific article and section numbers, and exact quotes from the relevant sections.
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
Never use the human retrieval to find contents in the insurance policy themselves. 
Never tell the user to contact the insurance company.

Once you have collected plenty of information to answer the user's question use the {final_answer_str} tool. 
    """.format(final_answer_str=final_answer_str)

    oracle_prompt = ChatPromptTemplate.from_messages([
        ("system", oracle_system_prompt),
        ("ai", "먼저 오늘 날짜와 고객님의 성함을 알려주세요."),
        ("user", "오늘 날짜: {today}\n 이름:\n{user_name}"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="messages"),
        ])

    if purpose == "Payout Estimate":
        tools=[fetch_insurance_term_con,human_retrieval,final_answer_payoutEstimate,fetch_insurance_enrollment_info]
    if purpose == "Claim Dispute":
        tools=[fetch_insurance_term_con,human_retrieval,final_answer_dispute,fetch_insurance_enrollment_info]
    if purpose == "Medical Support for Claims":
        tools=[fetch_insurance_term_con,human_retrieval,final_answer_medicalSupport,fetch_insurance_enrollment_info]
    if purpose == "General Inquiry":
        tools=[human_retrieval,final_answer_general,fetch_insurance_enrollment_info]
    if purpose == "General Inquiry about enrolled insurance":
        tools=[fetch_insurance_term_con,human_retrieval,final_answer_general,fetch_insurance_enrollment_info]


    oracle = (
        {
            "user_input": lambda x: x["user_input"],
            "user_name": lambda x: x["user_name"],
            "chat_history": lambda x: x["chat_history"],
            "specified_date_str" : lambda x: datetime.strptime(x["specified_date"], "%Y%m%d").strftime("%Y년 %m월 %d일") ,
            "today" : lambda x : datetime.today().strftime("%Y.%m.%d"),
            "insurance_enrollment_info": lambda x: process_and_print_active_policies(x["insurance_enrollment_info"],x["specified_date"]),
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
    user_name : str
    insurance_enrollment_info : dict
    chat_history: list[BaseMessage]
    specified_date : str 
    non_related : str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    response : str
    purpose : str
    report : dict
    end_of_session : str = ""
    

def router(state: list):
    try:
        if isinstance(state["messages"], list):
            for tool_call in state["messages"][-1].tool_calls:
                if tool_call['name'] == "human_retrieval":
                    return "human_retrieval"
            if len(state["messages"][-1].tool_calls) == 1 and "final_answer" in state["messages"][-1].tool_calls[0]["name"]:
                return "final_answer"
            return "tools"
        return "final_answer"
    except Exception as e:
        logger.error(f"Error in router: {str(e)}", exc_info=True)
        return "final_answer"  # Default to final_answer in case of error


tools = [fetch_insurance_term_con, fetch_insurance_enrollment_info]
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
