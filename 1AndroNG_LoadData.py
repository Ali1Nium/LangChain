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

#For Reading PDF Files ! 
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("Path of PDF !")
#Showing Pages ! 
pages = loader.load()

#print Words ! 
print(pages[0].page_content[ :500])


"Load data from Youtube "

from langchain.document_loaders.generic import GenericLoader 
#Making voice to text and we using for Youtube Videos to Text : 
from langchain.document_loaders.parsers import OpenAIWhisperParser
#Load VOice from Youtube : 
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

#Url of viedeo 
url = " url of the Viedeo"
#Path for saving it ! 
save_dir  = "path for saving ! "

"""
We make a Loader with mixing (generic and Youtube and OPenAI)
and we can load with this ! 

"""
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()

)

#Load ! 
docs = loader.load()
#Print!
print(docs[0].page_content[ :500])


"""LOAD FROM WEBSITE AND LINKS """

#Webloader  ! 
from langchain.document_loaders import WebBaseLoader 
loader = WebBaseLoader("Url of the Website ")
docs = loader.load()
print(docs[0].page_content[ :500])


"""Load Data from Notion """

from langchain.document_loaders import NotionDirectoryLoader 
loader = NotionDirectoryLoader("Path of DB")
docs = loader.load()
print(docs[0].page_content[ :500])