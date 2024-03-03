# %%capture


# # # # TEXT TO CHAT PART
# !pip install openai
# !pip install langchain
# !pip install unstructured
# !pip install tiktoken
# !pip install chromadb
# !pip install langchain-community
# !pip install environs
# !pip install pypdf
# !pip install gradio


from environs import Env

from langchain.chains import RetrievalQA

from langchain.chat_models import ChatOpenAI

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.prompts import PromptTemplate

from langchain.vectorstores import Chroma

from langchain_community.document_loaders.pdf import PyPDFLoader

import openai

from openai import OpenAI


api = "OPEN AI API"
openai.api_key = api #API
loader = PyPDFLoader("book.pdf") #KITOB MANZILI
pages = loader.load()
embedding = OpenAIEmbeddings(openai_api_key=api)
llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0,openai_api_key=api)

vectordb = Chroma.from_documents(pages, embedding)


#TEMPLETE. Buni o'zgartirish orqali Botni qanday ishlashini nazorat qilish mumkin (eng) 
template_uz = """answer the question completely, that is, with 2-3 sentences. 
Do not answer questions outside the book,
In whatever language the question is asked, give the answer in that language
"

{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT_uz = PromptTemplate(
   input_variables=["context", "question"], 
    template=template_uz
)

# Run chain
qa_chain_uz = RetrievalQA.from_chain_type(llm, 
                                          retriever=vectordb.as_retriever(), 
                                          return_source_documents=False,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT_uz})



def yes_man(question, history=False):
   return str(qa_chain_uz({"query": question})['result'])

question = "What is the book about?" #Savolni qabul qilish
yes_man(question) #JAVOB



#AUDIO TO CHAT PART
client = OpenAI(api_key=api)

audio_file= open("audio.wav", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file
)


print(transcription.text) #CHAT UCHUN UZATILGAN TEXT
print(yes_man(transcription.text)) #TEXTGA CHATNING JAVOBI