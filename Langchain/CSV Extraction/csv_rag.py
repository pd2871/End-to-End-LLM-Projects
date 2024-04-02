
from credentials import openai_api_key, model_name
import os, openai
import warnings, tiktoken

warnings.filterwarnings("ignore")
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

openai.api_key = openai_api_key
os.environ['openai_api_key'] = openai_api_key

tokenizer_name = tiktoken.encoding_for_model(model_name)
tokenizer = tiktoken.get_encoding(tokenizer_name.name)



def create_retriever_chain(csv_path):
    loader = UnstructuredCSVLoader(file_path=csv_path, mode="elements")
    docs = loader.load()
    # print(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    documents = text_splitter.split_documents(docs)
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
    vector_db = FAISS.from_documents(documents, embeddings)
    
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
    csv_path = 'test.csv'
    chain = create_retriever_chain(csv_path)
    while True:
        prompt = str(input("User: "))
        main(chain, prompt)
    