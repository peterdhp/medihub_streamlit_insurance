import os
import re
import random
import pickle
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from marker.config.parser import ConfigParser



def make_trans_path(input_path, trans_index_list, output_path_name_list, extension):

    split_path = input_path.split('/')

    for i in range(len(trans_index_list)):
        split_path[int(trans_index_list[i])] = str(output_path_name_list[i])
        
    
    #split_path[int(trans_index)] = str(output_path_name)
    
    trans_path = '/'
    for i in range(1, len(split_path)):
        if (i == len(split_path) - 1):
            #print('split_path : ', split_path, len(split_path))
            extension_index = split_path[i].rfind(".")
            #print('extension_index : ', extension_index)
            trans_path = os.path.join(trans_path, split_path[i][:extension_index] + extension)
            #print("trans_path : ", trans_path)
            #print()
        else:        
            trans_path = os.path.join(trans_path, split_path[i])
            os.makedirs(trans_path, exist_ok=True)

    #print("trans_path : ", trans_path)
    return trans_path


def error_file_select_and_extract():

    pdf_data_path_list = []
    output_data_path_list = []

    txt_file_list = ["교보생명.txt", "삼성화재.txt"]

    for txt_file in txt_file_list:
        

        with open(txt_file, "r") as f:
            data_lines = f.readlines()
            
        count = 0
        for i in range(len(data_lines)):
            tmp = re.split('[\|\n]', data_lines[i])
            
            if (("목차에 쪽수 명시 X" in tmp[0]) or \
                ("offset 충족, contents 결과" in tmp[0]) or \
                ("offset 미충족" in tmp[0]) or \
                ("pickle 안열림" in tmp[0]) or \
                ("pickle 목차 누락" in tmp[0]) or \
                ("pickle 안열림" in tmp[0])):
                
                pdf_path = make_trans_path(input_path = tmp[1],
                                           trans_index_list = [3],
                                           output_path_name_list = ['insurance'],
                                           extension = '.pdf'
                )
                
                output_path = make_trans_path(input_path = tmp[1],
                                              trans_index_list = [3],
                                              output_path_name_list = ['insurance_pickle_2_re'],
                                              extension = '.pickle'
                )
    
                if os.path.isfile(output_path):
                    pass
                else:
                    pdf_data_path_list.append(pdf_path)
                    output_data_path_list.append(output_path)
                
    
                count += 1
    
        print(f"{count} / {len(data_lines)}" )

    print(len(pdf_data_path_list))
    print(len(output_data_path_list))

    #random_index = random.randrange(0,len(pdf_data_path_list))
    #print(pdf_data_path_list[random_index])
    #print(output_data_path_list[random_index])
    
    return pdf_data_path_list, output_data_path_list


def markdown_page_split(markdown_data):

    index = 0
    page_text_list = []
    
    
    add_index = len('{' + f'{index}'+ '}' + '-'*48 + '\n\n')
    page_separate_index_st = markdown_data.find('{' + f'{index}'+ '}' + '-'*48 + '\n\n') + add_index
    page_separate_index_ed = markdown_data.find('{' + f'{index+1}'+ '}' + '-'*48 + '\n\n')
    
    while(page_separate_index_st >= 0):    
        
        #print("index : ", index)
        #print(page_separate_index_st, page_separate_index_ed)
    
        add_index = len('{' + f'{index}'+ '}' + '-'*48 + '\n\n')
    
        index += 1
        
    
        if page_separate_index_ed < 0:
            page_text_list.append(markdown_data[page_separate_index_st:])
            
            page_separate_index_st = markdown_data.find('{' + f'{index}'+ '}' + '-'*48 + '\n\n')
            
        else:
            page_text_list.append(markdown_data[page_separate_index_st:page_separate_index_ed])
    
            page_separate_index_st = markdown_data.find('{' + f'{index}'+ '}' + '-'*48 + '\n\n')  + add_index
    
        
        page_separate_index_ed = markdown_data.find('{' + f'{index+1}'+ '}' + '-'*48 + '\n\n')

    
    return page_text_list


def pdf2text(converter, pdf_data_path_list, output_data_path_list):

    
    for i in range(len(pdf_data_path_list)):

        try:
            rendered = converter(pdf_data_path_list[i])

            page_text_list = markdown_page_split(rendered.markdown)
            
            docs = []

            #{'metadata': }
            for page in range(len(page_text_list)):
                doc = {}
                doc['page'] = page + 1    # start page가 0
                doc['page_count'] = len(page_text_list)
                doc['offset'] = None
                doc['file_path'] = pdf_data_path_list[i]
                doc['file_name'] = pdf_data_path_list[i].split('/')[-1]
            
                docs.append({'metadata': doc,
                             'text': page_text_list[page]
                            })
                
                # Save updated data back to the pickle file
                with open(output_data_path_list[i], 'wb') as f:
                    pickle.dump(docs, f, pickle.HIGHEST_PROTOCOL)

        except:                
            with open("pdf_error.txt", "r+") as f:
                error_lines = f.read().splitlines()
                if pdf_data_path_list[i] not in error_lines:
                    f.write(pdf_data_path_list[i] + '\n')

        print(pdf_data_path_list[i])
        print(output_data_path_list[i])
        print()
            
        #break


if __name__ == "__main__":

    with open("pdf_error.txt", "a") as f:
        pass

    config = {
        "output_format": "markdown",
        "ADDITIONAL_KEY": "VALUE",
        "paginate_output": True,
    }
    config_parser = ConfigParser(config)
    
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        #llm_service=config_parser.get_llm_service()
    )

    pdf_data_path_list, output_data_path_list = error_file_select_and_extract()

    pdf2text(converter, pdf_data_path_list, output_data_path_list)

    
    

    
