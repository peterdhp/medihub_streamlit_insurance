import streamlit as st
import pymongo
from menu_streamlit import menu

import json

@st.cache_resource
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo_connection_string"])


def extract_name_from_insurance_data(json_data):
    # Parse the JSON data
    data = json_data
    flat_rate_name = None
    actual_loss_name = None
    
    if "data" not in data:
        raise ValueError("Invalid insurance data format: 'data' field missing")
        
    # Check resFlatRateContractList
    if "resFlatRateContractList" in data["data"]:
        contracts = data["data"]["resFlatRateContractList"]
        if contracts:
            contract = contracts[0]
            contractor = contract.get("resContractor")
            insured = contract.get("resInsuredPerson")
            
            # Choose name without asterisk if they're different
            if contractor and insured:
                if '*' in contractor and '*' not in insured:
                    flat_rate_name = insured
                elif '*' in insured and '*' not in contractor:
                    flat_rate_name = contractor
                else:
                    flat_rate_name = contractor
            else:
                flat_rate_name = contractor or insured
                
    # Check resActualLossContractList
    if "resActualLossContractList" in data["data"]:
        contracts = data["data"]["resActualLossContractList"]
        if contracts:
            contract = contracts[0]
            actual_loss_name = contract.get("resContractor")
    
    # Raise error if both lists are missing
    if "resFlatRateContractList" not in data["data"] and "resActualLossContractList" not in data["data"]:
        raise ValueError("가입된 정액 보험 혹은 실손보험이 없습니다.")
        
    # Compare names if both exist
    if flat_rate_name and actual_loss_name and flat_rate_name != actual_loss_name:
        raise ValueError(f"오류가 발생하였습니다. 개발팀에게 문의해주시기 바랍니다.")
        
    # Return the first available name
    return flat_rate_name or actual_loss_name or "미정"

st.title("보험정보 등록하기")
st.header("1. 신용정보원 회원 가입")
st.link_button("회원가입 하러 하기", "https://www.credit4u.or.kr:2443/memb/memberLoginView.do")
st.write("- 신용정보원 사이트에서 회원가입을 진행합니다.")
st.image("1.png")


st.header("2. 내 보험정보 조회하기")
st.link_button("내 보험 조회하기", "http://3.34.108.141:8091/hecto/insurance")
st.write("- 가입한 회원 정보로 내 보험을 조회합니다.")
st.image("2.png")



st.header("3. 내 보험 조회 파일 업로드하기")
uploaded_files = st.file_uploader("파일 업로드하기", type=['txt'],accept_multiple_files=False)
st.write("- 내 보험 조회 후 다운받은 insurance.txt 파일을 업로드 합니다.")
st.image("3.png")


if uploaded_files is not None:
    # Read the uploaded file content
    content = uploaded_files.read().decode('utf-8')
    
    # Add birthdate input field
    birthdate = st.text_input('주민번호 앞 6자리(비밀번호)', type='password')
    submit_button = st.button("등록하기")
    
    if submit_button and birthdate:
        # Connect to MongoDB
        client = init_connection()
        db = client['insurance_demo']
        collection = db['users']
        
        # Parse the JSON content
        insurance_data = json.loads(content)
        contractor_name = extract_name_from_insurance_data(insurance_data)
        
        # Check if user already exists
        existing_user = collection.find_one({
            "name": contractor_name,
            "birthdate": birthdate
        })
        
        if existing_user:
            # Update existing user's insurance data
            collection.update_one(
                {"_id": existing_user["_id"]},
                {"$set": {"insurance_enrollment": insurance_data}}
            )
            st.success("기존에 등록된 정보가 있으십니다. 기존 사용자 보험 가입 정보가 업데이트되었습니다.")
        else:
            # Create new user
            new_user = {
                "name": contractor_name,
                "birthdate": birthdate,
                "insurance_enrollment": insurance_data
            }
            collection.insert_one(new_user)
            st.success("새로운 사용자로 등록되었습니다.")

menu()
