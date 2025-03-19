import streamlit as st
from langchain_core.tracers.context import collect_runs
from LLMengine_split_DB import insurance_engine
from langsmith import Client
from menu_streamlit import menu_with_redirect
import datetime
import pymongo
client = Client()



if "user" not in st.session_state:
    st.session_state.user = ''
if "birth" not in st.session_state:
    st.session_state.birth = ''
#if "thread_id" not in st.session_state:
#    st.session_state.thread_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

if "messages_w" not in st.session_state:
    st.session_state["messages_w"] = [{"type": "ai", "content": "보험과 관련해서 어떤게 궁금하신가요?"}]
    #st.session_state.thread_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
if 'log_str' not in st.session_state:
    st.session_state['log_str'] = 'ai: 보험과 관련해서 어떤게 궁금하신가요?'+ '\n\n'


def reset():
    st.session_state["messages_w"] = [{"type": "ai", "content": "보험과 관련해서 어떤게 궁금하신가요?"}]

    

@st.cache_resource
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo_connection_string"])


if 'user_data' not in st.session_state:
    client = init_connection()
    db = client['insurance_demo']
    collection = db['users']
    items = collection.find()
    st.session_state.user_data = list(items)
    
    
for item in st.session_state.user_data:
    if item.get('name') == st.session_state.user:
        insurance_enrollment_info = item.get('insurance_enrollment')

#insurance_enrollment_info = st.secrets['INSURANCE_ENROLLMENT'][st.session_state.user]

#config = {"configurable": {"thread_id": st.session_state.thread_id}}

def submit_feedback():
    client.create_feedback(
        run_id,
        "thumbs",
        score=st.session_state.feedback,
        comment=st.session_state.feedback_text
    )
    st.session_state.feedback = None
    st.session_state.feedback_text = None
    st.session_state.run_id = None

for msg in st.session_state.messages_w:
    st.chat_message(msg["type"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages_w.append({"type": "human", "content": prompt})
    st.chat_message("human").write(prompt)
    st.session_state.log_str += 'human: ' + prompt+ '\n\n'
    with collect_runs() as cb:
        with st.spinner('보장곰이 가입하신 보험들을 살펴보고 있습니다.'):
            response = insurance_engine.invoke({"user_name": st.session_state.user,"user_input": prompt, "insurance_enrollment_info" : insurance_enrollment_info, "chat_history": st.session_state.messages_w},config={"recursion_limit": 15})
        st.session_state.run_id = cb.traced_runs[-1].id
        #print(st.session_state.run_id)
    if response['non_related'] == 'F' :
        st.session_state.messages_w.append({"type": "ai", "content": "저는 건강보험 관련 질문에 대해서만 답변할 수 있어요."})
        st.chat_message("ai").write("저는 건강보험 관련 질문에 대해서만 답변할 수 있어요.")
        st.session_state.log_str += 'ai: ' + "저는 건강보험 관련 질문에 대해서만 답변할 수 있어요." + '\n\n'
    else :
        msg = response["response"]
        details = response.get("report")
        st.session_state.messages_w.append({"type": "ai", "content": msg})
        st.chat_message("ai").write(msg)
        st.session_state.log_str += 'ai: ' + msg + '\n\n'
        
        
        if details :
            with st.expander("상담 일지 펼쳐보기"):
                if response['end_of_session'] == 'general':
                    details_str = """# {title}
### 상담요약
{chat_summary}

### 답변
{answer}
                
### 참조
{source}""".format(title= details['title'], chat_summary=details["chat_summary"], answer=details["answer"], source=details["source"])
                elif response['end_of_session'] == 'estimated_insurance_payout':
                    details_str = """# {title}
### 상담요약
{chat_summary}
                    
### 예상 보장금액
{estimate}
                    
### 보장금액 산정 세부내용
{estimate_details}
                    
### 참조
{source}""".format(title= details["title"], chat_summary=details["chat_summary"], estimate=details["estimate"], estimate_details= details["estimate_details"], source=details["source"])
                elif response['end_of_session'] == 'claims_adjuster':
                    details_str = """# {title}
### 상담요약
{chat_summary}
                    
### 답변
{answer}
                    
### 보장금액 관련 불만 사항
{dispute_reason}
                    
### 보장금액 관련 희망사항
{wanted_outcome}
                    
### 사건 세부 내용
{case_details}
                    
### 참조
{source}""".format(title=details["title"], chat_summary=details["chat_summary"], answer= details["answer"], dispute_reason= details["dispute_reason"],wanted_outcome= details["wanted_outcome"],case_details= details["case_details"],source=details["source"])
                elif response['end_of_session'] == 'medical_consulation':
                    details_str = """# {title}
### 상담요약
{chat_summary}
                    
### 답변
{answer}
                    
### 의학적 세부 내역
{medical_details}
                    
### 사용자의 과거력
{medical_history}
                
### 참조
{source}""".format(title= details["title"], chat_summary=details["chat_summary"], answer=details["answer"], medical_details= details["medical_details"], medical_history= details["medical_history"], source=details["source"])
                else:
                    details_str = "\n\n".join(f"{key}: {value}" for key, value in details.items())
                st.write(details_str)
                st.session_state.log_str += '\n\nreport:\n' + details_str + '\n'
                
   
    
    if 'end_of_session' in response :    
        if response['end_of_session'] == 'general' :
            st.session_state.messages_w.append({"type": "ai", "content": "궁금한 점이 잘 해소되었나요?"})
            st.chat_message("ai").write("궁금한 점이 잘 해소되었나요?")
            st.session_state.log_str += 'ai: 궁금한 점이 잘 해소되었나요?'
            
        if response['end_of_session'] == 'estimated_insurance_payout' :
            st.session_state.messages_w.append({"type": "ai", "content": "*실제 보장 금액이 예상 금액과 다를 수 있습니다. 보장 금액을 충분히 받지 못했다고 생각될 때는 닥터플렉스의 도움을 받아보세요."})
            st.chat_message("ai").write("*실제 보장 금액이 예상 금액과 다를 수 있습니다. 보장 금액을 충분히 받지 못했다고 생각될 때는 닥터플렉스의 도움을 받아보세요.")
            st.session_state.log_str += 'ai: *실제 보장 금액이 예상 금액과 다를 수 있습니다. 보장 금액을 충분히 받지 못했다고 생각될 때는 닥터플렉스의 도움을 받아보세요.'
            
        if response['end_of_session'] == 'claims_adjuster' :
            st.session_state.messages_w.append({"type": "ai", "content": "보험사의 보장금액에 대해 문제를 겪고 계신 것 같군요. 닥터플렉스의 보험 의료 자문 서비스를 이용해보세요."})
            st.chat_message("ai").write("보험사의 보장금액에 대해 문제를 겪고 계신 것 같군요. 닥터플렉스의 보험 의료 자문 서비스를 이용해보세요.")
            st.session_state.log_str += 'ai: 보험사의 보장금액에 대해 문제를 겪고 계신 것 같군요. 닥터플렉스의 보험 의료 자문 서비스를 이용해보세요.'
            
        if response['end_of_session'] == 'medical_consulation' :
            st.session_state.messages_w.append({"type": "ai", "content": "보험과 관련해서 의학적인 도움이 필요하신 것 같군요. 닥터플렉스의 닥터나이트와 의료자문 서비스를 이용해보세요."})
            st.chat_message("ai").write("보험과 관련해서 의학적인 도움이 필요하신 것 같군요. 닥터플렉스의 닥터나이트와 의료자문 서비스를 이용해보세요.")
            st.session_state.log_str += 'ai: 보험과 관련해서 의학적인 도움이 필요하신 것 같군요. 닥터플렉스의 닥터나이트와 의료자문 서비스를 이용해보세요.'

    
            
if st.session_state.get("run_id"):
    run_id = st.session_state.run_id
    st.text_input('[선택]코멘트를 입력해주세요.',key="feedback_text")
    st.feedback(
        options="thumbs",
        on_change=submit_feedback,
        key="feedback",
    )
            

with st.sidebar :
    st.button("대화 내역 초기화 하기", on_click=reset)
    st.download_button(
                label="대화내역 저장하기",
                data=st.session_state.log_str,
                file_name=datetime.datetime.now().strftime("%Y%m%d_%H%M")+'_보장곰 내역.txt',
                mime='text/plain'
    )
menu_with_redirect()