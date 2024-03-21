
import os, openai
from PyPDF2 import PdfReader
import warnings
warnings.filterwarnings("ignore")
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain import PromptTemplate



custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


def load_llm_embeddings():
    llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGUF",
    model_type="llama",
    config={'max_new_tokens': 2048,
                            'temperature': 0,
                            'context_length': 2048},
    
    temperature=0
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'cpu'})
    return llm, embeddings


def main(pdf_path, prompt):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # split into chunks
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Index documents into search engine
    llm, embeddings = load_llm_embeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)
    qa_prompt = set_custom_prompt()
    # RAG Chain
    retriever = vector_db.as_retriever()
    # retriever = VectorStoreRetriever(vectorstore=vector_db)
    # retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        return_source_documents=True,
                                       chain_type_kwargs={'prompt': qa_prompt},
                                    retriever=retriever)
    response = chain.invoke(prompt)
    print("Assistant: ", response['result'])
    print()


if __name__ == '__main__':
    while True:
        prompt = str(input("User: "))
        main(os.path.join('9cf7eda3-full.pdf'), prompt)
    