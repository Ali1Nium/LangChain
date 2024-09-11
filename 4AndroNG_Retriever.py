
                    #Maximum marginal relevance(MMR)
"""
When we get a Queryset (question) is very important that we find 
most good Data  about it . 
similarity_search method for searching wasnot too good !
"""

"""
when the question is not just about our document Data 
and its also with Meta data conected !

for example when the question has 2 parts :

"What movies about aliens were made in 1980?" 
part 1 : semantic part ==> about aliens 
part 2 : MetaData ==> move should be for 1980 .
"""

# We controll the number of anserws that we get with fetch_k
import os 
import openai
import sys 

sys.path.append(('../..'))

from dotenv import load_dotenv, find_dotenv

#We search for files with .env fromats ! 
_ = load_dotenv((find_dotenv()))

#API keys .!
openai.api_key = os.environ['OPENAI_API_KEY']





#For saving a Embeding : Chroma !
from langchain.vectorstores import chroma
from langchain.embeddings.openai import OpenAIEmbeddings

presist_directory = "docs/chroma/"
embedding = OpenAIEmbeddings()

vectordb = chroma.from_documents(
    embedding = embedding, 
    presist_directory=presist_directory
)

print(vectordb._collection.count())

#Example  ;

text = [
    "this is just for testing but i will speack abot Mashrooms !",
    "Mashooms are actualy good for the vision !",
    "Do not eat evry Mashroom that you see in Earth "
]

smalldb = chroma.from_text(text,embedding=embedding)
Question = "tell me about mashroms "
smalldb.similarity_search(Question, k=2)

"""Here we get 50% of anserw and may to get a Doplicated anserws maybe ."""

#Now we use MMR :

# We controll the number of anserws that we get with fetch_k
smalldb.max_marginal_relevance_search(Question,k=2,fetch_k=3)



#Self Query : 
#for filtering the Question maybe{third one !}

from langchain_openai import ChatOpenAI
from langchain.llms import openai
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo 

# we Send this Informationen to LLm and it can be better with more Informationen ! 
metadata_filds_info =[
    AttributeInfo(
        name ="source",
        description="The lecture the chunk is form, should be one of ",
        type="string",    
    ),
    AttributeInfo(
        name ="page",
        description="The page from the lecture",
        type="integer",  
    )
]

#Document Store ! : 

document_conten_description = "Lecture notes"
llm = ChatOpenAI(temperature=0)
retriver = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_conten_description,
    metadata_filds_info,
    verboser =True
)

#MOghayese Mafhomi ==>

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor 

def pretty_print_docs(docs):
    print(f"\n{"-"*100}\n".join([f"Document {d+1}:\n\n" + d.page_content for d in docs]))

llm =ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    best_retriver=vectordb.as_retriever()

)

question = ""
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs((compressed_docs))

"""WE GET A DOPLICATED ANSERW """
#CAHNGE TYPE :
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    best_retriver=vectordb.as_retriever(search_type="mmr")

)


#POP Liens :

from langchain.retrievers import SVMRetriever 
from langchain.retrievers import TFIDFRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Load PDF : 

loader = PyPDFLoader("path")
pages = loader.load()
all_page_text = [p.page_content for p in pages]
joined_page_text = " ".join(all_page_text)

#Split:
Text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 1500 ,
    chunk_overlap = 200 
)
splits = Text_spliter.split_text(joined_page_text)

#Retrivers :

svm_retrivers = SVMRetriever.from_texts(splits, embedding)
question = ""
compressed_docs = svm_retrivers.get_relevant_documents(question)
pretty_print_docs((compressed_docs))


tfid_retriver = TFIDFRetriever.from_text(splits)
question = ""
compressed_docs = tfid_retriver.get_relevant_documents(question)
pretty_print_docs((compressed_docs))