
import os, openai
import warnings
warnings.filterwarnings("ignore")
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader



class Chain:
    def __init__(self, file_path, openai_api_key, model_name, memory):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        os.environ['openai_api_key'] = openai_api_key
        self.file_path = file_path
        self.file_name = file_path.name
        self.memory = memory
        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings()
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
        self.prompt = PromptTemplate(
                            input_variables=["history", "context", "question"],
                            template=self.template,
                            )
        
    def pdf_loader(self, tmp_file_path):
        loader = PyMuPDFLoader(file_path=tmp_file_path)
        data = loader.load()
        return data
    
    def csv_excel_loader(self, tmp_file_path):
        if tmp_file_path.endswith('.csv'):
            loader = UnstructuredCSVLoader(file_path=self.file_path, mode="elements")
            data = loader.load()
        elif tmp_file_path.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path=self.file_path, mode="elements")
            data = loader.load()
        return data
            
    def create_chain(self):
        os.makedirs('data', exist_ok=True)
        with open(os.path.join('data', self.file_name), mode='wb') as tmp_file:
            tmp_file.write(self.file_path.getvalue())
            tmp_file_path = tmp_file.name
            if self.file_name.endswith('.pdf'):
                data = self.pdf_loader(tmp_file_path)
            elif tmp_file_path.endswith('.csv') or tmp_file_path.endswith('.xlsx'):
                data = self.csv_excel_loader(tmp_file_path)
            else:
                return None
        
        # Create a FAISS vector store and save embeddings
        retriever = FAISS.from_documents(data, self.embeddings).as_retriever()
        # Load the LLM
        llm = ChatOpenAI(streaming=True,
                    callbacks=[StreamingStdOutCallbackHandler()],
                    openai_api_key=self.openai_api_key,
                    model=self.model_name, temperature=0
                    )
        # Create a conversational chain
        chain = RetrievalQA.from_chain_type(llm=llm, verbose=False,
                                        retriever=retriever, chain_type_kwargs={
                                                                            "verbose": False,
                                                                            "prompt": self.prompt,
                                                                            "memory": self.memory
                                                                            })
        
        return chain
    
    