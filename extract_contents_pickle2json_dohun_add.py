# 80% CPU 코어 사용하기
# convert the document to markdown
import os
import time
import json
import random
import pickle
import datetime
from collections import Counter
from multiprocessing import Pool, cpu_count

import parmap

import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from typing import Annotated, List, TypedDict, Tuple, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from content_extractor_langgraph import content_extractor_langgraph

load_dotenv()

class SetConfig():
    def __init__(
        self, 
        model = "gpt-4o-mini",
        contents_extract_prompt_template = "",
        contents_detect_prompt_template = "",
        batch_size = 1,
        use_cpu_core_ratio = 0.1,
        max_page = 65,
        max_page_ratio = 0.15,
        short_text_threshold = 10,
        input_folder = None,
        trans_index = None,
        trans_name = None,
        from_extension = None,
        to_extension = None,
        check_file = "check_file.txt",
        error_file = "error_file.txt",
        ):
        
        self.model = model
        self.contents_extract_prompt_template = contents_extract_prompt_template
        self.contents_detect_prompt_template = contents_detect_prompt_template
        self.batch_size = batch_size
        self.use_cpu_core_ratio = use_cpu_core_ratio
        self.max_page = max_page
        self.max_page_ratio = max_page_ratio
        self.short_text_threshold = short_text_threshold
        self.input_folder = input_folder
        self.trans_index = trans_index
        self.trans_name = trans_name
        self.from_extension = from_extension
        self.to_extension = to_extension
        self.check_file = check_file
        self.error_file = error_file

    def __call__(self,):
        
        cb = f"""
        model \t: {self.model}
        batch_size \t: {self.batch_size}
        use_cpu_core_ratio \t: {self.use_cpu_core_ratio}
        max_page \t: {self.max_page}
        max_page_ratio \t: {self.max_page_ratio}
        short_text_threshold \t: {self.short_text_threshold}
        input_folder \t: {self.input_folder}
        trans_index \t: {self.trans_index}
        trans_name \t: {self.trans_name}
        from_extension \t: {self.from_extension}
        to_extension \t: {self.to_extension}
        """
        
        return cb


class TopicPageItem(BaseModel):
    """list of extracted topics and pages from table of contents in a markdown document."""
    check_topics_pages: list[list[Union[str, int]]] = Field(
        description="List of [topic, page_number] pairs extracted from table of contents, where topic is a string and page_number is an integer"
    )


def trans_extension(file_name, to_extension):
    
    file_name_list = file_name.split(".")
    
    new_file_name = ""
    for i in range(len(file_name_list) - 1):
        new_file_name += file_name_list[i] + "."
    new_file_name += to_extension
        
    return new_file_name


def make_trans_path(input_path, trans_index, trans_name, to_extension):

    split_path = input_path.split('/')    
    split_path[trans_index] = trans_name
    
    trans_path = '/'
    for i in range(1, len(split_path)):
        if (i == len(split_path) - 1):            
            trans_path = os.path.join(trans_path, trans_extension(split_path[i], to_extension))
        else:        
            trans_path = os.path.join(trans_path, split_path[i])
            os.makedirs(trans_path, exist_ok=True)

    #print("trans_path : ", trans_path)
    return trans_path


def text_preprocessing(text):
    # pymupdf4llm '�' --> ' ' 로 대체
    text = text.replace('�', ' ')

    # pymupdf4llm '\n\n' --> ' ' 로 대체
    text = text.replace('\n\n', ' ')

    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')

    return text
    

def filtering_page_split_contents(docs, config):

    # 본문의 내용이 많은 page를 포함하게 되면, max_length에 걸려 5로 설정
    #BATCH_SIZE = 5
    
    encoder = tiktoken.encoding_for_model(config.model)

    MAX_PAGE = max(config.max_page, int(len(docs) * config.max_page_ratio))
    if MAX_PAGE > len(docs):
        MAX_PAGE = len(docs)

    
    
    data_list = []
    ## 약관의 pages 가 많아 50 page 와 전체 page의 15% 중 큰 값까지 범위 설정
    
    for i in range(0, MAX_PAGE):
        
        ## 빈 페이지 또는 Short Text Filtering
        if len(docs[i]['text']) > config.short_text_threshold:
            
            # text_preprocessing
            data_list.append(text_preprocessing(docs[i]['text']))
    
    #print(f"MAX_PAGE : {MAX_PAGE}")
    #print(f"Filtering PAGE : {len(data_list)}")

    content_extractor = content_extractor_langgraph(config.model,config.contents_detect_prompt_template,config.contents_extract_prompt_template)

    contents_list = []

    tpm_count = 0
    
    earlystop_count = 0 ## 목차가 없는 경우 조기 종료
    
    for i in range((len(data_list)//config.batch_size)+1):
        
        if earlystop_count > 5: ## 목차가 뽑혔지만 목차가 없는 경우가 5번 생겼을 경우 조기 종료
            break

        sum_text = "\n".join(data_list[i*config.batch_size:(i+1)*config.batch_size])        

        token_count = encoder.encode(sum_text)        
        tpm_count += len(token_count)

        
        
        #print("tpm_count : ", tpm_count)

        ## TPM Limit 걸리는 부분 해결을 위해 TPM 확인
        #if tpm_count > 300000:
        #    time.sleep(30)
        #    tpm_count = 0

        
        #response = chain.batch(
        #    data_list[i*config.batch_size:(i+1)*config.batch_size]
        #)
                
        response = content_extractor.batch([{"doc": x} for x in data_list[i*config.batch_size:(i+1)*config.batch_size]])

        
        #for j in range(len(response)):
        for item in response:
            #print(item)
            if 'response' in item.keys():
                if len(item['response'].check_topics_pages) > 1:
                    
                    #for content in item.check_topics_pages:
                    for j in range(len(item['response'].check_topics_pages)):
                        
                        content = item['response'].check_topics_pages[j]
                        
                        #print("content : \n", content)
                        if len(content) > 1:
    
                            #if (content[1] != "None"):
                            if (content[1] != "None") and (type(content[1]) == int):                        
                                #print(content)
    
                                if j == 0:
                                    content_max_page = content[1]
    
                                # issue 
                                # 이전 page index로 접근하면 안됨
                                # ['관광진흥법 시행령', 159], ['【별표 2】해외체류자통지', 160], ['계약 전 알릴 의무', 2], ['계약 후 알릴 의무', 3], 
                                # ['상해보험', 4], ['배상책임보험', 5], ['화재보험', 6], ['보 통 약 관', 7], ['특 별 약 관', 8], ['별 표', 9]] ...
                                if content_max_page <= content[1]:
                                    #print("content_max_page : ", content_max_page)
                                    content_max_page = content[1]
    
                                    contents_list.append(content)
            elif len(contents_list) > 0:  
                earlystop_count += 1 ## 목차가 뽑혔지만 목차가 없는 경우 earlystop_count 증가
        
                

                                            
            #print(i, response[i].check_topics_pages, len(response[i].check_topics_pages))

    #print("contents_list : \n", contents_list)
    ## page 낮은 순으로 sort
    if len(contents_list) > 0:
        contents_list.sort(key=lambda x : x[1])
        
            
    return contents_list

def matching_contents_docs(contents_list, docs):

    documents = []
    for doc in docs:
        topic = []
        content_page = []
        
        new_doc = {
            "page_content":text_preprocessing(doc['text']),
            "metadata": {
                'source': doc['metadata']['file_path'],
                'insurance_name': doc['metadata']['file_path'].split('/')[-1][:-4],
                'page_count': doc['metadata']['page_count'],
                #'topic': contents_list[i][0],
                #'page': docs[PAGE_COUNT-1]['metadata']['page'] - docs[PAGE_COUNT-1]['metadata']['offset'],
                'page': doc['metadata']['page'],
                #'content_page': [],
                'offset': doc['metadata']['offset'],
                }
        }
        for i in range(len(contents_list)):
            
            curr_content_page = contents_list[i][1] + new_doc['metadata']['offset']

            if i < len(contents_list) - 1:
                next_content_page = contents_list[i+1][1] + new_doc['metadata']['offset']

            else:
                next_content_page = curr_content_page

            if (new_doc['metadata']['page'] >= curr_content_page) and (new_doc['metadata']['page'] <= next_content_page):
                #print("new_doc['metadata']['page'] : ", new_doc['metadata']['page'])
                #print("curr_content_page : ", curr_content_page)
                #print("next_content_page : ", next_content_page)
                topic.append(contents_list[i][0])
                content_page.append(contents_list[i][1])


        new_doc['metadata']['topic'] = topic
        new_doc['metadata']['content_page'] = content_page
        
        documents.append(new_doc)

    
    return documents


def contents_to_json(contents_list):
    
    contents_json = {"sections": []}
    
    for i in range(len(contents_list)):
        ## page 1부터 start, docs index 는 0부터 start
        ## 목차 맨 처음 page 이전은, 의미 없는 정보라고 가정하여 topic에 Remove 표시

        contents_json["sections"].append({"title": contents_list[i][0],"page": contents_list[i][1]})
        
    return contents_json


def pickle2json(data_chunk, config):

    contents_check_ratio = 0.6

    for dc in data_chunk:
        print("input path : ", dc)


        try:
            with open(dc, 'rb') as f:
                docs = pickle.load(f)
    
            #print(docs[0:5])
            if (docs[0]['metadata']['offset'] != None) and (docs[0]['metadata']['offset'] >= 0):
                contents_list = filtering_page_split_contents(docs, config)
    
                if len(contents_list) > 1:  ## [[]]
                    if (contents_list[-1][1] > docs[0]['metadata']['page_count'] * contents_check_ratio):
                        
                        #print("contents_list : \n", contents_list)
                        
                        with open(config.check_file, "r+") as f:
                            lines = f.read().splitlines()
                            if dc not in lines:
                                f.write(dc + "\n")
                        ################################################################################            
                        #matching_documents, contents_json = matching_contents_docs(contents_list, docs)
                        matching_documents = matching_contents_docs(contents_list, docs)
                        #print(matching_documents[:4])
        
                        
                        json_file_path = make_trans_path(
                            input_path = dc,
                            trans_index = config.trans_index, 
                            trans_name = config.trans_name, 
                            to_extension = config.to_extension, 
                        )
                        
                        with open(json_file_path, 'w') as f:
                            json.dump(matching_documents, f, indent=4, ensure_ascii=False)
                        print(f"json_file_path : {json_file_path}\n")
    
                        
                        contents_file_path = make_trans_path(
                            input_path = dc,
                            trans_index = config.trans_index, 
                            trans_name = "contents_extract", 
                            to_extension = config.to_extension, 
                        )
                        contents_json = contents_to_json(contents_list)
                        #print("contents_json : \n", contents_json)
                        
                        with open(contents_file_path, 'w') as f:
                            json.dump(contents_json, f, indent=4, ensure_ascii=False)
    
                        
    
                    
    
                    
        
                        ################################################################################
                    
        
        
                    else:
                        #print(f"docs[0]['metadata']['page_count'] * {contents_check_ratio} : {docs[0]['metadata']['page_count'] * contents_check_ratio}")
                        #print("contents_list : \n", contents_list)
                        #print(f"File Fail : {dc}")
                        with open(config.error_file, "r+") as f:
                            lines = f.read().splitlines()
                            if dc not in lines:
                                f.write(dc + "\n")


        except:
            pass


if __name__ == "__main__":
    
    data_path = []

    with open("./contents_extract_prompt_template", "r") as f:
        contents_extract_prompt_template = f.read()
    with open("./contents_detect_prompt_template", "r") as f:
        contents_detect_prompt_template = f.read()
    

    config = SetConfig(
        model="gpt-4o-mini",
        contents_extract_prompt_template = str(contents_extract_prompt_template),
        contents_detect_prompt_template = str(contents_detect_prompt_template),
        batch_size = 10,
        use_cpu_core_ratio = 0.4,
        input_folder = "/workspace/data1/insurance_pickle_2",
        trans_index = 3,
        trans_name = "insurace_json",
        from_extension = "pickle",
        to_extension = "json",
        #max_page = 65,
        #max_page_ratio = 0.15
        
    )

    print("config : ", config())

    with open(config.check_file, "a") as f:
        pass
    with open(config.error_file, "a") as f:
        pass
    

    
    
    for (path, dirs, files) in os.walk(config.input_folder):
        for file_name in files:
            if '.' + config.from_extension in file_name:
                t = make_trans_path(
                    input_path = os.path.join(path, file_name), 
                    trans_index = config.trans_index, 
                    trans_name = config.trans_name, 
                    to_extension = config.to_extension, 
                )
                if os.path.isfile(t):
                    pass
                else:
                    data_path.append(os.path.join(path, file_name))
    
    data_path.sort()
    print("data_path : ", len(data_path))
    
    num_cores = int(cpu_count() * config.use_cpu_core_ratio)  # 시스템의 CPU 코어 수 확인
    print("num_cores : ", num_cores)
    data_path_chunks = [data_path[i::num_cores] for i in range(num_cores)]  # 데이터를 CPU 코어 수만큼 청크로 나눔

    #pickle2json(data_path_chunks[0], config)


    results = parmap.map(pickle2json, data_path_chunks, config, pm_pbar=True, pm_processes=num_cores)