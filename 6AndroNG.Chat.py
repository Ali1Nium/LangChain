import langchain_openai
import os 
import openai
import pdfminer.pdftypes
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

question = ""
docs = vectordb.similarity_search(question, k=3)
len(docs)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model= "gpt-3.5-turbo",
    temperature=0
)

llm.predict(("Hello world!"))




from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


template = """use the following pieces to context to answer the question at the 
{context}
Question : {question}
Helpful answer :"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)


#Read the db better with return_source_documents  . 
#Adding prompt with chain_type_kwargs .

question = " "

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriver =vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
)

resualt = qa_chain({"query":question})

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

from langchain.chains import ConversationalRetrievalChain
retriever = vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever= retriever,
    memory=memory
)
question = " "
resualt = qa({"query":question})

"""                           THis will intialize your database and reciever chain                        """

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter 
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader 
from langchain.chains import RetrivalQA, ConversationalRetrivalChain
from langchain.openai import ChatOpenAI 
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

def load_db(file, chain_type, k):
    #Load documents 
    loader = PyPDFLoader(file)
    documents = loader.load()
    #spilit documents : 
    Text_spliter = CharacterTextSplitter(chunk_size = 650 , chunk_overlap = 0, separator="")
    docs = Text_spliter.split_documents(documents)
    #define embedding :
    embedding = OpenAIEmbeddings()
    #create vecctor database for data 
    db = DocArrayInMemorySearch.from_documents(docs, embedding)
    #define retriver 
    retriver = db.as_retriever(search_type = "similarity", search_kwargs = {"k":k})
    #create a chatbot chain. Memory is managed separately 
    qa = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(
            model = "gpt",
            temperatur = 0
        ),
        chain_type = chain_type,
        retriever=retriever,
        return_source_documents = True, 
        return_generated_question=True,

    )
    return qa 

.

 