
                    #Making Group and Embeding : 
                    #Symantic search !

import os 
import openai
import sys 

sys.path.append(('../..'))

from dotenv import load_dotenv, find_dotenv

#We search for files with .env fromats ! 
_ = load_dotenv((find_dotenv()))

#API keys .!
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader

                            # Load PDF :


loaders = [
    #Duplicate documents on prupose - messy data ==> unclean data ! 
    PyPDFLoader("Path of PDF1 !"),
    PyPDFLoader("Path of PDF1 !"),
    PyPDFLoader("Path of PDF3 !"),
    PyPDFLoader("Path of PDF4 !")

]

docs = []

for loader in loaders:
    docs.extend(loader.load())

from langchain.text_splitter import RecursiveCharacterTextSplitter

                        #Using Recursive for making Group b Datas .! 
Text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap= 150
    )

splits = Text_spliter.split_documents((docs))

                        #New we need to Embeding this Groups : 

from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

#For saving a Embeding : Chroma !
from langchain.vectorstores import chroma

presist_directory = "docs/chroma/"
                        
                        #To check the CHroma the be Empty : 


#in terminal : ==> !rm -rf ./docs/chroma 
import os
import shutil

# Check if directory exists, then remove it
if os.path.exists(presist_directory):
    shutil.rmtree(presist_directory)
    print(f'Directory {presist_directory} has been removed.')
else:
    print(f'Directory {presist_directory} does not exist.')

                         #Create Vector DataBase

vectordb = chroma.from_documents(
    documents = splits,
    embedding = embedding, 
    presist_directory=presist_directory
)

#Number of Groups : 
print(vectordb.collection.count())


"""                        Example for Embeding and compare                """
#After Embeding we can compare the Groups : (with numpy )
sentence1 = "Hi its first Sentence for testing !"
sentence2 = "HI its second Sentence for testing "

embedding = OpenAIEmbeddings()
embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
import numpy as np 

#print a Compare precent .! 
print(np.dot(embedding1,embedding2))



"""                          Now we Want to use it !                       """

Question = "is in this Documents somthing about test??"
docs = vectordb.similarity_search(Question, k=3)

#presist the Database:
vectordb.presist()