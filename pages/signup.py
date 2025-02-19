import streamlit as st
import pymongo
from menu_streamlit import menu

@st.cache_resource
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo_connection_string"])


def extract_name_from_insurance_data(json_data):
    # Parse the JSON data
    data = json_data
    
    # Extract contractor name from the insurance data structure
    if "data" in data:
        # Check resFlatRateContractList
        if "resFlatRateContractList" in data["data"]:
            contracts = data["data"]["resFlatRateContractList"]
            if contracts:
                # Get first contract's contractor name
                contract = contracts[0]
                return contract.get("resContractor")
                
        # Check resActualLossContractList
        if "resActualLossContractList" in data["data"]:
            contracts = data["data"]["resActualLossContractList"]
            if contracts:
                # Get first contract's contractor name 
                contract = contracts[0]
                return contract.get("resContractor")
                
    return "미정"

st.title("보험정보 등록하기")
st.header("1. 신용정보원 회원 가입")
st.link_button("회원가입 하러 하기", "https://www.credit4u.or.kr:2443/memb/memberLoginView.do")
st.write("- 신용정보원 사이트에서 회원가입을 진행합니다.")
st.image("1.png")


st.header("2. 내 보험정보 조회하기")
st.link_button("내 보험 조회하기", "http://54.180.149.155:8091/hecto/insurance")
st.write("- 가입한 회원 정보로 내 보험을 조회합니다.")
st.image("2.png")



st.header("3. 내 보험 조회 파일 업로드하기")
uploaded_files = st.file_uploader("파일 업로드하기", type=['txt'],accept_multiple_files=False)
st.write("- 내 보험 조회 후 다운받은 insurance.txt 파일을 업로드 합니다.")
st.image("3.png")


st.header("4. 등록 완료")
st.write("- 파일 업로드까지 진행해주시면 등록이 완료됩니다. 접수 절차가 완료되면 따로 연락 드리겠습니다. 감사합니다")

if uploaded_files is not None:
    # Read the uploaded file content
    content = uploaded_files.read().decode('utf-8')
    #print(content)
    # Connect to MongoDB
    client = init_connection()
    db = client['insurance_demo']
    collection = db['users']
    
    # Parse the JSON content
    import json
    insurance_data = json.loads(content)
    # insurance_data is already a dict, no need to parse again
    contractor_name = extract_name_from_insurance_data(insurance_data)
    
    new_user = {
    "name": contractor_name,
    "birthdate": "medihub", 
    "insurance_enrollment": insurance_data
    }
    
    collection.insert_one(new_user)
    

menu()
