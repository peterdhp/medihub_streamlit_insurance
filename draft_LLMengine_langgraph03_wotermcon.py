
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt.chat_agent_executor import AgentState
import os 
import streamlit as st
from langchain_core.messages import BaseMessage


os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']



def verify(state):
    user_input = state['user_input']
    chat_history = state['chat_history']
    if len(chat_history) > 1 : 
        chat_history = chat_history[:-2]
        chat_history_text = "\n".join(f"ai: {msg['content']}" if msg["type"] == "ai" else f"User: {msg['content']}" for msg in chat_history)
    else :
        chat_history_text = 'none'
    
    class Question_Type(BaseModel):
        Qtype: str = Field(
            description="One of: 'not_related', 'enroll_info_required', 'enroll_info_not_required'"
        )    
    
    # Define the LLM with structured output
    llm4omini = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    question_classifier_llm = llm4omini.with_structured_output(Question_Type)

        
    prompt_verify_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given a chat history and user input , classify the user input into three categories.
not_related : if the user input is not related to health related insurances.
enroll_info_required : if the user input requires insurance enrollment information to answer the respond.
enroll_info_not_required : if the user input pertains to health related insurances and does not require insurance enrollment information to respond.
If the response contains instructions to """),
        ("user", "[chat history]\n{chat_history}\n\n[user input]\n{user_input}")
    ])
    policy_terms_classifier = prompt_verify_prompt | question_classifier_llm

    response = policy_terms_classifier.invoke({'user_input':user_input,'chat_history':chat_history_text})
    Q_type = response.Qtype
    
    return {'type' : Q_type}

    
def continue_v_error(state):
    """
    Determines whether to continue or return error

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    Q_class = state["type"]
    
    if Q_class == "enroll_info_not_required" :
        return "answer_node"
    else :
        return "END"
    

def answer_node(state):
    user_input = state['user_input']
    chat_history = state['chat_history']
    
    answer_system_prompt = """You are an insurance consultant. 
    Given a simple question about health insurance that doen't require enrollment information, please answer the question.
    """

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", answer_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_input}"),
        ])


    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=0
    )


    answer_generator = answer_prompt | llm | StrOutputParser()

    response = answer_generator.invoke({'user_input':user_input, 'chat_history':chat_history})
    
    return {"response": response}



class State(AgentState):
    user_input: str
    chat_history: list[BaseMessage]
    type : str
    response : str
    


from langgraph.graph import StateGraph, END

graph = StateGraph(State)

graph.add_node("verify",verify)
graph.add_node("answer_node",answer_node)


graph.add_edge(START,"verify")
graph.add_conditional_edges(
    "verify",
    continue_v_error,
    {
        "answer_node": "answer_node",
        "END": END,
    },
)

graph.add_edge("answer_node", END)

insurance_wotc_engine = graph.compile()