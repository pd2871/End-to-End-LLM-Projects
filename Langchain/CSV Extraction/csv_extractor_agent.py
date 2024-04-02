
from credentials import openai_api_key, model_name
import os, openai
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

openai.api_key = openai_api_key
os.environ['openai_api_key'] = openai_api_key


def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


def main(df, prompt):
    llm = ChatOpenAI(
                temperature=0,
                model_name=model_name
            )
    agent = create_pandas_dataframe_agent(llm, df, agent_type="openai-tools", verbose=False)
    response = agent.invoke(
        {
            "input": prompt
        }
    )
    print("Assistant: ", response['output'])
    print()
    print()



if __name__ == '__main__':
    csv_path = 'test.csv'
    df = read_csv(csv_path)
    while True:
        prompt = str(input("User: "))
        main(df, prompt)
    