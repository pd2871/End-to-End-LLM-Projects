
from credentials import openai_api_key, model_name
import os, openai
import warnings, tiktoken, tempfile
warnings.filterwarnings("ignore")
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader, UnstructuredExcelLoader

openai.api_key = openai_api_key
os.environ['openai_api_key'] = openai_api_key


class Chain:
    def __init__(self, file_path):
        self.file_path = file_path
        self.template = """
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
        
    def pdf_loader(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(self.file_path.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(file_path=tmp_file_path)
        data = loader.load()
        return data
    
    def csv_excel_loader(self):
        if self.file_path.endswith('.csv'):
            loader = UnstructuredCSVLoader(file_path=self.file_path, mode="elements")
            data = loader.load()
        elif self.file_path.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path=self.file_path, mode="elements")
            data = loader.load()
        return data
    
    def create_chain(self):
        if self.file_path.endswith('.pdf'):
            data = self.pdf_loader()
        elif self.file_path.endswith('.csv') or self.file_path.endswith('.xlsx'):
            data = self.csv_excel_loader()
        else:
            return None
        #memory for chat history
        memory = ConversationBufferWindowMemory(
                                            memory_key='history',
                                            input_key='question',
                                            k=1,
                                            return_messages=True
                                            )
        prompt = PromptTemplate(
                            input_variables=["history", "context", "question"],
                            template=self.template,
                            )
        # Create embeddings using Sentence Transformers
        embeddings = OpenAIEmbeddings()
        # Create a FAISS vector store and save embeddings
        retriever = FAISS.from_documents(data, embeddings).as_retriever()
        # Load the language model
        llm = ChatOpenAI(streaming=True,
                    callbacks=[StreamingStdOutCallbackHandler()],
                    openai_api_key=openai_api_key,
                    model=model_name, temperature=0
                    )
        # Create a conversational chain
        chain = RetrievalQA.from_chain_type(llm=llm, verbose=False,
                                        retriever=retriever, chain_type_kwargs={
                                                                            "verbose": False,
                                                                            "prompt": prompt,
                                                                            "memory": memory
                                                                            })
        return chain
    
    