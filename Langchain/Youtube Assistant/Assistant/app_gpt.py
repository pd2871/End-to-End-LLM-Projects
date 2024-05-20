
from credentials import openai_api_key
import os, openai
from PyPDF2 import PdfReader
import warnings
warnings.filterwarnings("ignore")
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

openai.api_key = openai_api_key
os.environ['openai_api_key'] = openai_api_key


def main(pdf_path, prompt):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # split into chunks
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=300,
    length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Index documents into search engine
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)  #FAISS Vector Store
    
    llm = ChatOpenAI(openai_api_key=openai_api_key,
        model="gpt-4-1106-preview", temperature=0,
    )
    
    # RAG Chain
    retriever = vector_db.as_retriever()
    # retriever = VectorStoreRetriever(vectorstore=vector_db)
    # retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    chain = RetrievalQA.from_llm(llm=llm,
                                    retriever=retriever)
    response = chain.invoke(prompt)
    print("Assistant: ", response['result'])
    print()


if __name__ == '__main__':
    pdf_dir = str(input("Enter PDF path: "))
    print()
    while True:
        prompt = str(input("User: "))
        main(pdf_dir, prompt)
    