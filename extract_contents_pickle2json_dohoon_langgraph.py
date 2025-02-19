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
        ):
        
        self.model = model
        self.contents_extract_prompt_template = contents_extract_prompt_template
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


def make_trans_path(input_path, config):

    split_path = input_path.split('/')    
    split_path[config.trans_index] = config.trans_name
    
    trans_path = '/'
    for i in range(1, len(split_path)):
        if (i == len(split_path) - 1):            
            trans_path = os.path.join(trans_path, trans_extension(split_path[i], config.to_extension))
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

    content_extractor = content_extractor_langgraph(config.model,'','')
    # LLM with function call
    
    contents_list = []

    tpm_count = 0
    
    for i in range((len(data_list)//config.batch_size)+1):

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
        response = response['response']
        print(i, len(response))
        
        #for j in range(len(response)):
        for item in response:
            #print(item)
            if len(item.check_topics_pages) > 1:
                
                #for content in item.check_topics_pages:
                for j in range(len(item.check_topics_pages)):
                    
                    content = item.check_topics_pages[j]
                    
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
            
            #new_doc['metadata']['page'] 
            if i < len(contents_list) - 1:
                next_content_page = contents_list[i+1][1] + new_doc['metadata']['offset']
                
                if (new_doc['metadata']['page'] >= curr_content_page) and (new_doc['metadata']['page'] < next_content_page):
                    topic = [contents_list[i][0]]
                    content_page = contents_list[i][1]
                    break
                    
                elif (new_doc['metadata']['page'] >= curr_content_page) and (new_doc['metadata']['page'] == next_content_page):
                    topic = [contents_list[i][0], contents_list[i+1][0]]
                    content_page = [contents_list[i][1], contents_list[i+1][1]]
                    break
            else:
                next_content_page = new_doc['metadata']['page_count']
                
                if (new_doc['metadata']['page'] >= curr_content_page) and (new_doc['metadata']['page'] < next_content_page):
                    topic = [contents_list[i][0]]
                    content_page = contents_list[i][1]
                    break
                    
                elif (new_doc['metadata']['page'] >= curr_content_page) and (new_doc['metadata']['page'] == next_content_page):
                    topic = [contents_list[i][0]]
                    content_page = [contents_list[i][1]]
                    break

                
                
        new_doc['metadata']['topic'] = topic
        new_doc['metadata']['content_page'] = content_page
        
        documents.append(new_doc)

    
    return documents


def pickle2json(data_chunk, config):

    for dc in data_chunk:
        print(dc)



        #try:
        with open(dc, 'rb') as f:
            docs = pickle.load(f)

        #print(docs[0:5])
        if (docs[0]['metadata']['offset'] != None) and (docs[0]['metadata']['offset'] >= 0):
            contents_list = filtering_page_split_contents(docs, config)

            if len(contents_list) > 0:

                #matching_documents, contents_json = matching_contents_docs(contents_list, docs)
                matching_documents = matching_contents_docs(contents_list, docs)
                #print(matching_documents[:4])
            
            
                json_file_path = make_trans_path(
                    input_path = dc,
                    config = config,
                )
                                
                with open(json_file_path, 'w') as f:
                    json.dump(matching_documents, f, indent=4, ensure_ascii=False)
                print(f"json_file_path : {json_file_path}\n")
                    

        #except:
        #    pass


if __name__ == "__main__":
    
    data_path = []

    with open("./contents_prompt_template_dohun", "r") as f:
        contents_extract_prompt_template = f.read()

    config = SetConfig(
        model="gpt-4o-mini",
        contents_extract_prompt_template = str(contents_extract_prompt_template),
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
    

    
    
    for (path, dirs, files) in os.walk(config.input_folder):
        for file_name in files:
            if '.' + config.from_extension in file_name:
                t = make_trans_path(
                    input_path = os.path.join(path, file_name), 
                    config = config,
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