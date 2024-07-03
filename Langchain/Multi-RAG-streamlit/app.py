
from chains import Chain
import shutil, os, asyncio
import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

# def ask_question(chain, query):
#     result = chain.invoke(query)
#     return result["result"]

os.makedirs('data', exist_ok=True)


st.set_page_config(page_title='RAG - Q&A Bot', page_icon='🔗')
st.title("Chat with PDF, Excel or CSV - 📚🚀")

openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", placeholder='OpenAI GPT-4 API Key...', type='password')
model_name = st.sidebar.selectbox(
    "Model Name",
    ("gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-turbo"))
file = st.sidebar.file_uploader("Upload File")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    st.session_state['chat_history'].append(AIMessage("Hello ! Ask me about your uploaded files 🤗"))
           
           
#write history
for message in st.session_state['chat_history']:
    if isinstance(message, HumanMessage):
        with st.chat_message('user'):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message('assistant'):
            st.markdown(message.content)
    
if openai_api_key and file:
    try:
        shutil.rmtree('data')
    except:
        pass
    if 'memory' not in st.session_state:
        st.session_state['memory'] = ConversationBufferWindowMemory(
                                            memory_key='history',
                                            input_key='question',
                                            k=1,
                                            return_messages=True
                                            )
    # st.session_state['file'] = file
    chain = Chain(file, openai_api_key, model_name, st.session_state['memory'])
    rag_chain = chain.create_chain()
    if rag_chain is None:
        st.sidebar.error('Error: Upload only PDF, Excel or CSV file', icon="🚨")   
    else:
        # with st.chat_message("assistant"):
        #     st.write("Hello ! Ask me about your uploaded files 🤗")
            
        prompt = st.chat_input("Ask about your PDF, CSV or Excel data 🧮")
        if prompt:
            st.session_state['chat_history'].append(HumanMessage(prompt))
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                ai_response = rag_chain.invoke(prompt)['result']
                st.write(ai_response)
                # rag_chain.stream({'query':prompt}))
            st.session_state['chat_history'].append(AIMessage(ai_response))
    