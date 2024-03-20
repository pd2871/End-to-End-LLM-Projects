from credentials import openai_api_key, model_name
from tools import ImageDescriberTool
import os
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import YouTubeSearchTool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

tools = [YouTubeSearchTool()]  #, ImageDescriberTool()

prompt = PromptTemplate(
    input_variables=[
        "user_question",
        "image_path",
        "chat_history",
        "human_input",
        "agent_scratchpad"

    ],
    template=(
        '''
        Previous conversation:
        {chat_history}

        Begin!

        User Question: {user_question}
        Image Path: {image_path}
        Thought:{agent_scratchpad}
        Human Input: {human_input}
        '''
    )
)


memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
    input_key="human_input"
)


llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0,
    model_name='gpt-3.5-turbo'
)

# Construct the OpenAI Tools agent
openai_agent = create_openai_tools_agent(llm, tools, prompt)

agent = AgentExecutor(agent=openai_agent, tools=tools, verbose=True, memory=memory)
# agent.invoke({"user_question": 'Hello?',
#                                     "image_path": '',
#                                     'human_input': '',
#                                     "chat_history": [
#                                                         HumanMessage(content="hi! how are you?"),
#                                                         AIMessage(content="Hello user! I am fine. How can I assist you today?"),
#                                                         ],})
#Who is lex fridman, describe him and Give useful youtube links related to lex fridman.
while True:
    promp = input("Ask questions: ")
    response = agent.invoke({"user_question": promp,
                                    "image_path": '',
                                    'human_input': '',
                                    "chat_history": [
                                                        HumanMessage(content="hi! how are you?"),
                                                        AIMessage(content="Hello user! I am fine. How can I assist you today?"),
                                                        ],})

    print("Response", response['output'])



