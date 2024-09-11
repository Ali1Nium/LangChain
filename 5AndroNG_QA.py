"""
FOr asking question from our Documents ,
This job after complition the Storags , 
after retrieve the all Groups .

We need to pass them into LLM  
"""
import langchain_openai
import os 
import openai
import sys 

sys.path.append(('../..'))

from dotenv import load_dotenv, find_dotenv

#We search for files with .env fromats ! 
_ = load_dotenv((find_dotenv()))

#API keys .!
openai.api_key = os.environ['OPENAI_API_KEY']


from langchain.vectorstores import Chroma 
from langchain.embeddings.openai import OpenAIEmbeddings 


persist_directory = "docs/chroma/"
embedding = OpenAIEmbeddings()

vectordb = Chroma(persist_directory=persist_directory, embedding_function= embedding)

print(vectordb._collection.count())

question = ""
docs = vectordb.similarity_search(question, k=3)
len(docs)

#importing LLM :

from langchain_openai import ChatOpenAI


llm = ChatOpenAI( 

    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=1000,
    verbose=True
)

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval_qa.base import RetrievalQA


qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

resualt = qa_chain({"query":question})

#Build prompt :
"""  HERE WE PASS THE PROMPT WITH ALL OF INFORMATIONS TO LLM
    AND WE WRITE THE DUTY IN  TEMPLATE (WHAT SHOULD THE LLM DO !) 
    """

from langchain.prompts import PromptTemplate

template = """use the following pieces to context to answer the question at the 
{context}
Question : {question}
Helpful answer :"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


#Read the db better with return_source_documents  . 
#Adding prompt with chain_type_kwargs .
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriver =vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
)

question = " "

resualt = qa_chain({"query":question})


                          #Second way  (for Larg Documents !)

#for larg Data and DOcuments read about the chain_type = refine