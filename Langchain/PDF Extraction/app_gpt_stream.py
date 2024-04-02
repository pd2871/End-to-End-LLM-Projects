
from credentials import openai_api_key, model_name
import os, openai, time
from timeit import default_timer as timer
from PyPDF2 import PdfReader
import warnings, tiktoken
warnings.filterwarnings("ignore")
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

openai.api_key = openai_api_key
os.environ['openai_api_key'] = openai_api_key

tokenizer_name = tiktoken.encoding_for_model(model_name)
tokenizer = tiktoken.get_encoding(tokenizer_name.name)


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def create_retriever_chain(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=2000,
    chunk_overlap=300,
    length_function=tiktoken_len,
    )
    chunks = text_splitter.split_text(text)
    
    template = """
                Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
                ------
                <ctx>
                {context}
                </ctx>
                ------
                <hs>
                {history}
                </hs>
                ------
                {question}
                Answer:
                """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    
    # Index documents into search engine
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)
    
    memory = ConversationBufferWindowMemory(
                                            memory_key='history',
                                            input_key='question',
                                            k=1,
                                            return_messages=True
                                            )
    
    # RAG Chain
    retriever = vector_db.as_retriever()
    
    llm = ChatOpenAI(streaming=True,
                    callbacks=[StreamingStdOutCallbackHandler()],
                    openai_api_key=openai_api_key,
                    model=model_name, temperature=0
    )

    chain = RetrievalQA.from_chain_type(llm=llm, verbose=False,
                                        retriever=retriever, chain_type_kwargs={
                                                                            "verbose": False,
                                                                            "prompt": prompt,
                                                                            "memory": memory
                                                                            })
    return chain


def main(chain, prompt):
    response = chain.invoke(prompt)  #get response from GPT
    # print("Assistant: ", response['result'])
    print()
    print()


if __name__ == '__main__':
    pdf_path = 'summary.pdf'
    chain = create_retriever_chain(pdf_path)
    while True:
        prompt = str(input("User: "))
        main(chain, prompt)
    