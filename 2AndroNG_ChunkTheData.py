"""
To Chank Docs we have to do some notices 




"""

from langchain.text_splitter import CharacterTextSplitter

Test_text_splitter = CharacterTextSplitter(
    separator = "\n\n" , 
    #Size of CHunk for every Select !
    chunk_size = 4000,
    #Size of text for overlaping !
    chunk_overlap = 200, 
    #len of words : 
    lenght_function = " <built function len >," 
    )

"""
#Methods for chunking in Langchain: 

#Getting list of Documents : 
creat_document()

#Getting a Document ! 
quit_document()

#Types Of SPlitters : 
https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/

"""


#Make a Groups :

#LOading Data for reading from Data :

""" 
we can read File or Documents from PDF, Youtube, DataBases and ..."""

import os 
import openai
import sys 

sys.path.append(('../..'))

from dotenv import load_dotenv, find_dotenv

#We search for files with .env fromats ! 
_ = load_dotenv((find_dotenv()))

#API keys .!

openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

chunk_size = 26 
chunk_overlap = 4

r_spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size , chunk_overlap= chunk_overlap)
r_spliter = CharacterTextSplitter(chunk_size=chunk_size , chunk_overlap= chunk_overlap)

text1 = "THis is just for testing !"
r_spliter.split_text((text1))
""" 
Here we see if len of Text be more than chunk_size ,
the spliter give us a List with datas that their lenght is 26 .
and the Over_lap is also same (end of last one is first of news one !)
"""

#CharacterTextSplitter: 
"With this we can count the Spaces as a charector !!"
c_spliter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap= chunk_overlap,
    separator= " "
    )

c_spliter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap= chunk_overlap,
    separators= ["\n\n","\n"," ","(?<=\.)",""]
    )


#Working with PDF documents : 

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("Path of PDF !")
#Showing Pages ! 
pages = loader.load()

#Making Loader ! 

Text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap= chunk_overlap,
    separators= ["\n\n","\n"," ","(?<=\.)",""],
    lenght_function = len
    )

#For working with the documents and the text : 
docs = Text_spliter.split_documents(pages)


#Working with DB Documents  : 

from langchain.document_loaders import NotionDirectoryLoader 

loader = NotionDirectoryLoader("Path")
noton_db = loader.load()

docs = Text_spliter.split_documents(noton_db)


#Split with Token : 

#We spilit with Token ==> chunk_size = 1 : 

from langchain.text_splitter import TokenTextSplitter

Text_Splitter = TokenTextSplitter(chunk_size=1 , chunk_overlap= 0)

Text1 = "Its just a test for splitting with Tokens ! "

Text_spliter.split_text(Text1)


""" Split Data with Tokenize and MetaData ! """

from langchain.text_splitter import MarkdownHeaderTextSplitter 

markdown_document = """# TItle\n\n \ 
## Chapter I\n\n \
Hi tjis is Jim\n\n Hi this is Joe\n\n \ 
### Section \n\n  \
Hi this is Lance \n\n 
## Chapter 2\n\n \ 
HI this is Molly"""

headers_to_split_on = [
    ("#", "Header 1 "),
    ("##", "Header 2"),
    ("###", "Header 3 "),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

md_header_splits = markdown_splitter.split_text(markdown_document)

