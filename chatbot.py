##Imports:
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
#text SPilter and Chunking : 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
#Jasazi Data : 
from langchain_openai import OpenAIEmbeddings
#Sample data base for saving data : 
from langchain_community.vectorstores.faiss import FAISS
##Amade kardan Data baray namayesh :
from langchain.chains.retrieval import create_retrieval_chain 
#Saving Chat History :
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import MessagesPlaceholder
#Saving Chat History : Second and better way : 
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from dotenv import load_dotenv
load_dotenv()


#Load Data From url : 

def get_document_from_web(url):
    loeader = WebBaseLoader(url)
    docs = loeader.load()
    spliter = RecursiveCharacterTextSplitter(
        chunk_size = 400, 
        chunk_overlap = 20 
    )
    SplitDocs = spliter.split_documents(docs)
    return SplitDocs

#Create_DB:

def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore 

#Create Model : 

def creat_chain(vectorStore):
    model = ChatOpenAI(
        model = "gpt-3.5-turbo",
        temperature=0.4
    )
    
    prompt = ChatMessagePromptTemplate.from_messages([
        ("system", "Answer the users questions based on the context : {context}"),
        MessagesPlaceholder(variable_name="chat_history")
        ("human","{input}")
    ])
    

    chain = create_stuff_documents_chain(
        llm = model,
        prompt=prompt
    )


    retriever = vectorStore.as_retriever(search_kwargs = {"k": 3})

    retriver_prompt = ChatMessagePromptTemplate.format_messages([
        MessagesPlaceholder(variable_name="chat_history "),
        ("human","{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        
    ])
    history_awar_retriever = create_history_aware_retriever(
        llm = model ,
        retriever = retriever,
        prompt = retriver_prompt
    )

    retrieval_chain = create_retrieval_chain(
        # retriever,history_awar_retriever
        history_awar_retriever,
        chain
    )

    return retrieval_chain 

##New : CHat !
def prosses_chat(chain,question,chat_history):
    response = chain.invoke({
        "input": question ,
        "chat_history":chat_History
    })

    return response["answer"]

#Creating main Loop : 

if __name__ =="__main__":
    docs = get_document_from_web("Urls !")
    vectorStor = create_db(docs)
    chain = creat_chain(vectorStor)

    #Saving CHat History :
    chat_History = []

    while True:
        user_input  = input("You : ") 
        if user_input.lower == "exit":
            break

    response  = prosses_chat(chain, user_input,chat_History)
    chat_History.append(HumanMessage(content=user_input))
    chat_History.append(AIMessage(content=response))

    print("Assistant:",response)

"""
yek chat bot k be Data haye Khareji vasl ast sakhte shode .
va agar az on Soali porside shavad ba tavajo b on etlaat va Data ha javab midahad ..

Nokte digar : CHat history b on Ezafe shode k chat ra zakhire karde va ba tavajo b mokaleme mitavanad 
b soalat pasokh dahad . !

"""


