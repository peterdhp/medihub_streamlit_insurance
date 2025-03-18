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
from content_page_detector import content_detector_chain

load_dotenv()

class SetConfig():
    def __init__(
        self, 
        model = "gpt-4o-mini",
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
        check_file = "./check_file.txt",
        double_check_file = "./double_check_file.txt",
        error_file = "./error_file.txt",
        double_check_error_file = "./double_check_error_file.txt"
        ):
        
        self.model = model
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
        self.double_check_file = double_check_file
        self.error_file = error_file
        self.double_check_error_file = double_check_error_file

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


if __name__ == "__main__":
    
    data_path = []



    config = SetConfig(
        model="gpt-4o-mini",
        batch_size = 10,
        use_cpu_core_ratio = 0.5,
        input_folder = "/workspace/data1/insurance_pickle_2",
        trans_index = 3,
        trans_name = "insurace_json",
        from_extension = "pickle",
        to_extension = "json",
        #max_page = 65,
        #max_page_ratio = 0.15
        
    )
    
    with open(config.double_check_file, "r+", encoding="utf-8", errors="replace") as f:
        double_check_files = f.read().splitlines()
    
    count = 0
        
    for (path, dirs, files) in os.walk(config.input_folder):
        for file_name in files:
            if os.path.join(path, file_name) not in double_check_files:
                if '.' + config.from_extension in file_name:
                    
                    
                    json_file_path = make_trans_path(
                            input_path = os.path.join(path, file_name),
                            trans_index = config.trans_index, 
                            trans_name = "insurace_json", 
                            to_extension = config.to_extension, 
                        )
    
                    
                    contents_file_path = make_trans_path(
                        input_path = os.path.join(path, file_name),
                        trans_index = config.trans_index, 
                        trans_name = "contents_extract", 
                        to_extension = config.to_extension, 
                    )
                    
                    
                    if os.path.isfile(json_file_path):
                        os.remove(json_file_path)
                    if os.path.isfile(contents_file_path):
                        os.remove(contents_file_path)
                        
                        count += 1
    print("count:", count)
    
    
    
    
    