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
from langgraph.prebuilt import create_react_agent
import datetime
from difflib import get_close_matches


os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']



def is_active_policy(policy_dict):
    """
    Example logic: 
      - We consider a policy active if 'resContractStatus' == '정상'
      - Optionally also check date range (commEndDate in the future).
        But your data uses strings like '20200214'. You can parse them as needed.
    """
    if policy_dict.get('resContractStatus') != '정상':
        return False
    
    # Example: parse commEndDate and check if still in the future
    # (You can skip this if you just want to filter by '정상')
    end_date_str = policy_dict.get('commEndDate', '')  # e.g. "20200214"
    if not end_date_str:
        return False
    
    # Try to parse year-month-day
    try:
        end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d").date()
        today = datetime.date.today()  # or any reference date you want
        return end_date >= today
    except ValueError:
        # If date format is wrong or missing, skip
        return False

def extract_active_flat_rate_contracts(data: dict):
    """
    Return a list of active (정상) flat-rate contracts.
    """
    contracts = data.get('data', {}).get('resFlatRateContractList', [])
    active = []
    for c in contracts:
        if is_active_policy(c):
            active.append(c)
    return active

def render_policy_as_table(policy_dict):
    """
    Returns a multiline string for a single policy in your desired format.
    """
    # Basic policy fields
    company_name = policy_dict.get('resCompanyNm', 'Unknown')
    insurance_name = policy_dict.get('resInsuranceName', 'Unknown')
    policy_number = policy_dict.get('resPolicyNumber', 'Unknown')
    policyholder = policy_dict.get('resContractor', 'Unknown')
    start_date = policy_dict.get('commStartDate', 'Unknown')
    end_date   = policy_dict.get('commEndDate', 'Unknown')
    payment_cycle = policy_dict.get('resPaymentCycle', 'Unknown')
    payment_period = policy_dict.get('resPaymentPeriod', 'Unknown')
    premium = policy_dict.get('resPremium', 'Unknown')

    # Convert date format YYYYMMDD -> YYYY.MM.DD for nicer display
    def pretty_date(yyyymmdd):
        if len(yyyymmdd) == 8:
            return f"{yyyymmdd[0:4]}.{yyyymmdd[4:6]}.{yyyymmdd[6:8]}"
        return yyyymmdd

    start_date_str = pretty_date(start_date)
    end_date_str   = pretty_date(end_date)

    # Gather coverage rows
    coverage_list = policy_dict.get('resCoverageLists', [])

    # Build table rows
    coverage_rows = []
    for cov in coverage_list:
        coverage_type  = cov.get('resAgreementType', '')
        coverage_name  = cov.get('resCoverageName', '')
        coverage_stat  = cov.get('resCoverageStatus', '')
        coverage_amt   = cov.get('resCoverageAmount', '0')
        # Format coverage amount with commas
        try:
            coverage_amt = f"{int(coverage_amt):,}"
        except:
            pass

        # Example row: coverage_type, coverage_name, coverage_stat, coverage_amt
        coverage_rows.append(
            f"| {coverage_type:<30} "
            f"| {coverage_name:<60} "
            f"| {coverage_stat:<6} "
            f"| {coverage_amt:>10} |"
        )

    # Construct final output
    result_lines = []

    result_lines.append(f"Insurance Company: {company_name}")
    result_lines.append(f"Insurance Name: {insurance_name}")
    result_lines.append(f"Policy Number: {policy_number}")
    result_lines.append(f"Policyholder: {policyholder}")
    result_lines.append(f"Coverage Start Date: {start_date_str}")
    result_lines.append(f"Coverage End Date:   {end_date_str}")
    result_lines.append(f"Payment Frequency:   {payment_cycle}")
    result_lines.append(f"Payment Term:        {payment_period} years")
    result_lines.append(f"Premium per Payment: {premium} KRW")
    result_lines.append("Coverage Details:")
    result_lines.append("| Coverage Type                 | Coverage Name                                               | Status | Coverage Amount |")
    result_lines.append("|-------------------------------|------------------------------------------------------------|--------|----------------|")

    # Append coverage rows
    result_lines.extend(coverage_rows)

    # Join them all with newlines
    return "\n".join(result_lines) + "\n"

def process_and_print_active_policies(demo_data) -> str:
    """
    Filters for active policies, then builds and returns a
    single multiline string containing all those policies.
    """
    active_policies = extract_active_flat_rate_contracts(demo_data)
    
    if not active_policies:
        return "No active policies found."
    
    results = []
    for i, policy in enumerate(active_policies, start=1):
        table_str = render_policy_as_table(policy)
        # Add a section header + the table + a separator line
        block = f"[Insurance #{i}]\n{table_str}\n" + ("-" * 10)
        results.append(block)
    
    # Combine everything into one big string
    final_output = "\n\n".join(results)
    #print(final_output)
    return final_output


class InsuranceQuery(TypedDict):
    """Represents a pair consisting of an insurance name and a related query."""
    insurance_name: str= Field(
        description="The name of the insurance policy from which information will be retrieved."
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

def get_insurance_details(data, insurance_name):
    result = []
    contracts = data.get("data", {}).get("resFlatRateContractList", [])
    for contract in contracts:
        if contract["resInsuranceName"] == insurance_name:
            result.append({
                "company_name": contract["resCompanyNm"],
                "start_date": contract["commStartDate"]
            })
    return result

@tool("fetch_insurance_term_con")
def fetch_insurance_term_con(query_list : list[InsuranceQuery], insurance_enrollment_info: Annotated[dict, InjectedState("insurance_enrollment_info")]):
    """Retrieves relevant information from insurance terms and conditions based on a list of queries. 
Each query specifies an 'insurance_name' and a 'query' describing the details to be extracted. 
This is useful for finding context or specific information related to insurance policies."""
    
    insurance_context = ''
    #insurance_enrollment_info_text = process_and_print_active_policies(insurance_enrollment_info)
    
    
    
    for query in query_list:
        
        insurance_name = query['insurance_name']
        
        all_contracts = insurance_enrollment_info.get('data', {}).get('resFlatRateContractList', [])
        for contract in all_contracts:
            if contract["resInsuranceName"] == insurance_name:
                matching_contract = contract
                insurance_company = contract["resCompanyNm"]
                insurance_start_date = contract["commStartDate"]
                break
            
        insurance_company_code_dict = {'메리츠화재' : "0101" , "한화손보" : "0102", "삼성화재" : "0108", "DB손보" : "0111", "NH농협손해보험" : "0171", "삼성생명" : "0203"}
        insurance_company_code = insurance_company_code_dict.get(insurance_company, "Unknown")
        matching_insurance_text = render_policy_as_table(matching_contract)
        
        # insurance_name을 이용해서 insurance_enrollment_info에서 해당 보험에 대한 정보를 가져올 옴
        # 이정보를 통해서 보험사 코드와 보험가입일 가졍오기
        # 이를 이용해서 해당 보험에 대한 보험약관 목차와 본문 json 파일을 가져와서 loaded_toc, loaded_documents에 저장
        query = query["query"]
        query_result = ''
        
        with open("documents/contents_json/"+ insurance_company_code +".json", "r", encoding="utf-8") as json_file:
            loaded_company_toc = json.load(json_file)
                
        matching_items = [item for item in loaded_company_toc if item['name'] == insurance_company]
        
        if not matching_items:  # If no exact matches found, use difflib for the closest match
            names = [item['name'] for item in loaded_company_toc]
            closest_match = get_close_matches(insurance_company, names, n=1)
            if closest_match:
                matching_items = [item for item in loaded_company_toc if item['name'] == closest_match[0]]
        
        # Filter items with start_date before insurance_start_date
        valid_items = [
            item for item in matching_items 
            if datetime.strptime(item['start_date'], "%y%m%d") < insurance_start_date
        ]
        if not valid_items:  # If no valid items found, return None
            return None
        
        # Sort by start_date to find the latest one
        matching_item = max(valid_items, key=lambda x: datetime.strptime(x['start_date'], "%y%m%d"))
        toc_list = matching_item.get("sections", [])
        formatted_toc = "\n".join([f"{item['title']} - Page {item['page']}" for index, item in enumerate(toc_list)])
        
        page_selector_system_prompt = """Given a query and insurance enrollment info, decide which pages of the insurance terms and conditions you would like to retrieve information from.
Give up to 10 pages
[Insurance enrollment information]
{enroll_info}

[Table of contents] : The Table of Content below only lists the starting page numbers for each section.
{table_of_contents}"""

        page_selector_prompt = ChatPromptTemplate.from_messages([
        ("system", page_selector_system_prompt),
        ("user", "query : {query}"),])
        
        class Pagelist(BaseModel):
            """list of page numbers to retrieve information from."""

            page_numbers: list[str] = Field(
            description="list of page numbers to retrieve information from."
        )
            
        structured_pagenum_llm = llm.with_structured_output(Pagelist)


        # Combine the prompt and classifier
        page_selector = page_selector_prompt | structured_pagenum_llm

        response = page_selector.invoke({"query" : query , "table_of_contents" : formatted_toc,"enroll_info" :matching_insurance_text})
        pages_to_include = response.page_numbers
        #print(pages_to_include)
            
        
        with open("documents/vector_db/"+matching_item["start_date"]+'_'+matching_item['name']+"*.json", 'r', encoding='utf-8') as f:
            loaded_documents = json.load(f)
        
        
        page_results =[]
        # Fetch and add sections once for each insurance
        for page in pages_to_include:
            matching_doc = retrieve_documents_by_metadata(
                loaded_documents,
                source=f"{insurance_name}.pdf",
                page=page
            )
            page_content = matching_doc.get("page_content", "")
            section = matching_doc.get("topic", "")
            formatted_content = (
                f"Section: {section}\n\n"
                f"{page_content}"
            )
            page_results.append(formatted_content)


        #f"Insurance Name: {insurance_name}\n"
        # Combine all results for this insurance name
        query_result = (f"Insurance Name: {insurance_name}\n"f"Query: {query}")
        query_result = "\n".join(page_results)
        insurance_context += '\n\n---\n\n' + query_result

    return insurance_context


def human_retrieval_node(state):
   """Prompts the user for information. Useful for gathering details directly from the user, 
    especially when clarifying or collecting information related to their health condition."""
   
   query = state['user_input']
   chat_history = state['chat_history']
   chat_history_text = "\n".join(f"ai: {msg['content']}" if msg["type"] == "ai" else f"User: {msg['content']}" for msg in chat_history)
    
    
   human_retrieval_system_prompt = """Given a user query and chat history. Generate a question to ask the user in order to retrieve additional information needed to answer the query. Try to ask one question at a time.
Here are the some examples of required information for determining insurance claim eligibility and answering the user's question:

1. 실손 (Reimbursement Insurance):
   - Information to help determine if the treatment was for a legitimate medical purpose.
   - Details regarding hospitalization disputes (e.g., appropriateness of hospitalization, 6-hour standard)

2. 상해 (Injury):
   - Information about the initial accident (e.g. first examination records, emergency records, traffic accident or ambulance records).
   - The nature and severity of the injury to determine the injury grade.
   - Clarification whether the injury is related to degenerative disease or another condition.

3. 질병 (Illness):
   - Initial charts or records to determine the diagnosis and the nature of the illness.
   - Test results and clarify which tests were used to establish the diagnosis, as required for claim eligibility.
"""
   human_retrieval_prompt = ChatPromptTemplate.from_messages([
   ("system", human_retrieval_system_prompt),
   ("user", "chat history : \n {chat_history}\n\nquery : {input}"),
])
    
   llm4o = ChatOpenAI(
   model="gpt-4o",
   openai_api_key=os.environ["OPENAI_API_KEY"],
   temperature=0
)
   human_retrieval_chain = human_retrieval_prompt | llm4o | StrOutputParser()
   
   question = human_retrieval_chain.invoke({"input": query, "chat_history": chat_history_text})
   
   
   #tool_call_id = state["messages"][-1].tool_calls[0]["id"]
   #tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": question}]
   return {
        "response": question, "end_of_session" : "continue"
    }


@tool("human_retrieval")
def human_retrieval(user_input : Annotated[str, InjectedState("user_input")], chat_history : Annotated[list[BaseMessage], InjectedState("chat_history")]):
   """Prompts the user for information. Useful for gathering details directly from the user, 
    especially when clarifying or collecting information related to their health condition."""
   
   chat_history_text = "\n".join(f"ai: {msg['content']}" if msg["type"] == "ai" else f"User: {msg['content']}" for msg in chat_history)
    
    
   human_retrieval_system_prompt = """Given a user query and chat history. Generate a question to ask the user in order to retrieve additional information needed to answer the query. Try to ask one question at a time.
Here are the some examples of required information for determining insurance claim eligibility and answering the user's question:

1. 실손 (Reimbursement Insurance):
   - Information to help determine if the treatment was for a legitimate medical purpose.
   - Details regarding hospitalization disputes (e.g., appropriateness of hospitalization, 6-hour standard)

2. 상해 (Injury):
   - Information about the initial accident (e.g., first examination records, emergency records, traffic accident or ambulance records).
   - The nature and severity of the injury to determine the injury grade.
   - Clarification whether the injury is related to degenerative disease or another condition.

3. 질병 (Illness):
   - Initial charts or records to determine the diagnosis and the nature of the illness.
   - Test results and clarify which tests were used to establish the diagnosis, as required for claim eligibility.
   
"""
   human_retrieval_prompt = ChatPromptTemplate.from_messages([
   ("system", human_retrieval_system_prompt),
   ("user", "chat history : \n {chat_history}\n\nquery : {input}"),
])
    
    
   llm4o = ChatOpenAI(
   model="gpt-4o",
   openai_api_key=os.environ["OPENAI_API_KEY"],
   temperature=0
)
   human_retrieval_chain = human_retrieval_prompt | llm4o | StrOutputParser()
   
   question = human_retrieval_chain.invoke({"input": user_input, "chat_history": chat_history_text})
   
   return question

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
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    structured_llm_classifier = llm.with_structured_output(PolicyTermsClassification)

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


def verify(state):
    user_input = state['user_input']
    chat_history = state['chat_history']
    if len(chat_history) > 1 : 
        chat_history = chat_history[:-2]
        chat_history_text = "\n".join(f"ai: {msg['content']}" if msg["type"] == "ai" else f"User: {msg['content']}" for msg in chat_history)
    else :
        chat_history_text = 'none'
    class Relevance(BaseModel):
        """Binary score to assess user inputs's relevance to health or life related insurances. Return 'T' for user inputs relevant to health related medicine and 'F' for others."""
        binary_score: str = Field(
            description="Binary score: 'T' if the user input pertains to health or life related insurance, 'F' otherwise."
        )
        
    
    # Define the LLM with structured output
    llm4omini = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    relevance_classifier_llm = llm4omini.with_structured_output(Relevance)

        
    prompt_verify_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and user input , verify whether the user input pertains to health or life related insurances."),
        ("user", "[chat history]\n{chat_history}\n\n[user input]\n{user_input}")
    ])
    policy_terms_classifier = prompt_verify_prompt | relevance_classifier_llm

    response = policy_terms_classifier.invoke({'user_input':user_input,'chat_history':chat_history_text})
    relatedTF = response.binary_score
    
    return {'non_related' : relatedTF}



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
    
    

@tool("final_answer")
def final_answer(
    response : str
):
    """When enough details are provided, returns a natural language response to the user input in korean.
The tone should be as polite as possible and attentive.
When the response is referenced on a context from the term and condition, the name of the insurance and explanation should be provided. 
o not instruct the user to call the insurance company or read the policy themselves.

    Args :
        response : An answer to the user question
    """
    
    return response

def final_answer_node(state):
    """When enough details are provided, returns a natural language response to the user input in korean.
The tone should be as polite as possible and attentive.
When the response is referenced on a context from the term and condition, the name of the insurance and explanation should be provided. 
o not instruct the user to call the insurance company or read the policy themselves.

    Args :
        response : An answer to the user question
    """
    response = state["messages"][-1].tool_calls[0]['args']['response']
    
    
    return {'response' : response}

def answer_type_classifier(state):
    response = state["response"]
    chat_history = state['chat_history']
    if len(chat_history) > 1 : 
        chat_history = chat_history[:-2]
        chat_history_text = "\n".join(f"ai: {msg['content']}" if msg["type"] == "ai" else f"User: {msg['content']}" for msg in chat_history)
    else :
        chat_history_text = 'none'
    class Answer_Type(BaseModel):
        action: str = Field(
            description="One of: 'estimated_insurance_payout', 'claims_adjuster', 'medical_consultation', 'continue'"
        )
    
    # Define the LLM with structured output
    llm4omini = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    relevance_classifier_llm = llm4omini.with_structured_output(Answer_Type)
        
    response_classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are given a chat history and a final answer from that conversation. "
         "Decide which of the following actions is most appropriate: "
         "'estimated_insurance_payout' if the answer provides an insurance payout estimate, "
         "'claims_adjuster' if it suggests contacting a claims adjuster with enough info, "
         "'medical_consultation' if it suggests seeing a doctor with enough info, "
         "or 'continue' if more info is needed or no other category applies. "
         "Return only the category string."),
        ("user", 
         "[chat history]\n{chat_history}\n\n[final answer]\n{response}")
    ])
    response_classifier = response_classifier_prompt | relevance_classifier_llm

    result = response_classifier.invoke({'response':response,'chat_history':chat_history_text})
    EOS = result.action
    
    return {'end_of_session' : EOS}




# oracle_system_prompt = """You are the oracle, the great AI decision maker. Given the user's query you must decide what to do with it based on the list of tools provided to you.
# If you see that a tool has been used (in the scratchpad) with a particular query, do NOT use that same tool with the same query again. Also, do NOT use any tool more than twice (ie, if the tool appears in the scratchpad twice, do not use it again).
# You should aim to collect information from a diverse range (including the user) of sources before providing the answer to the user.
# Once you have collected plenty of information to answer the user's question (stored in the 'scratchpad') use the final_answer tool.
# If no additional information is needed, skip to the final answer tool
# Be cautious about the enrollment information."""

oracle_system_prompt = """You are an insurance consultant. 
You have full access to insurance enrollment information.
You have full access to insurance policy, terms and conditions(보험약관) through your tools. 
Never tell the user to contact the insurance company or read the insurance policy themselves. 
If you need more info, ask the user via the human_retrieval tool. 
Use fetch_insurance_term_con to get coverage details from the documents.

When asked about improving coverage, focus on explaining factors (e.g., policy tiers, coverage limits) rather than offering direct solutions. 
If specialized help (e.g., claims adjuster, doctor) is needed, gently request relevant details. When enough details have been provided, 
provide a summary of the information and reccomend contacting a claims adjuster or doctor.

Once you have collected plenty of information to answer the user's question use the final_answer tool. 

Insurance enrollment information:\n{insurance_enrollment_info}
"""

oracle_prompt = ChatPromptTemplate.from_messages([
    ("system", oracle_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="messages"),
    ])


llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0
)

tools=[
    fetch_insurance_term_con,
    human_retrieval,
    final_answer
]

tool_node = ToolNode(tools)

oracle = (
    {
        "user_input": lambda x: x["user_input"],
        "chat_history": lambda x: x["chat_history"],
        "insurance_enrollment_info": lambda x: process_and_print_active_policies(x["insurance_enrollment_info"]),
        "messages": lambda x: x["messages"],
    }
    | oracle_prompt
    | llm.bind_tools(tools, tool_choice="any")
)


class State(AgentState):
    user_input: str
    insurance_enrollment_info : dict
    chat_history: list[BaseMessage]
    non_related : str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    response : str
    end_of_session : str = ""
    

def run_oracle(state: list):
    print("run_oracle")
    print(f"messages: {state['messages']}")
    out = oracle.invoke(state)
    
    return {
        "messages": [out]
    }

def router(state: list):
    # return the tool name to use
    if isinstance(state["messages"], list):
        for tool_call in state["messages"][-1].tool_calls :
            if tool_call['name'] == "human_retrieval":
                return "human_retrieval"
        if state["messages"][-1].tool_calls[0]["name"] == "final_answer" :
            return "final_answer"
        else :
            return "tools"
    else:
        # if we output bad format go to final answer
        print("Router invalid format")
        return "final_answer"
    
    
tool_str_to_func = {
    "fetch_insurance_term_con": fetch_insurance_term_con,
    "human_retrieval": human_retrieval,
    "final_answer": final_answer
}



from langgraph.graph import StateGraph, END

graph = StateGraph(State)

graph.add_node("verify",verify)
graph.add_node("oracle", run_oracle)
graph.add_node("tools", tool_node)
graph.add_node("final_answer", final_answer_node)
graph.add_node("human_retrieval", human_retrieval_node)
graph.add_node("answer_type_classifier", answer_type_classifier)


graph.add_edge(START,"verify")
graph.add_conditional_edges(
    "verify",
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
graph.add_edge("final_answer", 'answer_type_classifier')
graph.add_edge("answer_type_classifier", END)

insurance_engine = graph.compile()