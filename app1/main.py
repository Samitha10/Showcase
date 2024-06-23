from llama_index.llms.groq import Groq
import os
import pandas as pd
import streamlit as st

from llama_index.core.query_pipeline import (
        QueryPipeline as QP,
        Link,
        InputComponent,
    )
from llama_index.experimental.query_engine.pandas import (
        PandasInstructionParser,
    )
from llama_index.core import PromptTemplate

groq_key = os.getenv("GROQ_KEY")
chat = Groq(model="Llama3-70b-8192", api_key=groq_key)

instruction_str = (
        "1. Convert the query to executable Python code using Pandas.\n"
        "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
        "3. The code should represent a solution to the query.\n"
        "4. PRINT ONLY THE EXPRESSION.\n"
        "5. Do not quote the expression.\n"
    )

pandas_prompt_str = (
        "You are working with a pandas dataframe in Python.\n"
        "The name of the dataframe is `df`.\n"
        "This is the result of `print(df.head())`:\n"
        "{df_str}\n\n"
        "Follow these instructions:\n"
        "{instruction_str}\n"
        "Query: {query_str}\n\n"
        "Expression:"
    )
response_synthesis_prompt_str = (
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n\n"
        "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
        "Pandas Output: {pandas_output}\n\n"
        "Response: "
    )

def agent(query_str,df):
    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
            instruction_str=instruction_str, df_str=df.head(5)
        )
    pandas_output_parser = PandasInstructionParser(df)
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
    llm = chat


    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": chat,
            "pandas_output_parser": pandas_output_parser,
            "response_synthesis_prompt": response_synthesis_prompt,
            "llm2": chat,
        },
        verbose=True,
    )
    qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    qp.add_links(
            [
                Link("input", "response_synthesis_prompt", dest_key="query_str"),
                Link(
                    "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
                    ),
                Link(
                    "pandas_output_parser",
                    "response_synthesis_prompt",
                    dest_key="pandas_output",
                    ),
            ]
        )
    # add link from response synthesis prompt to llm2
    qp.add_link("response_synthesis_prompt", "llm2")


    response = qp.run(
        query_str= query_str,
    )

    return response


def main():
    st.subheader("Pandas Dataframe Query Agent - Experimental", divider="rainbow", anchor=False)
    st.sidebar.title("Chat with your CSV File")
    st.markdown(
        """
        **Pandas Query Engine :**   
        This guide shows you how to use our PandasQueryEngine: convert natural language to Pandas python code using LLMs.
        The input to the PandasQueryEngine is a Pandas dataframe, and the output is a response. 
        The LLM infers dataframe operations to perform in order to retrieve the result.

        :red[**WARNING :**]  
        This tool provides the LLM access to the eval function.
        Arbitrary code execution is possible on the machine running this tool. While some level of filtering is done on code, this tool is not recommended to be used in a production setting without heavy sandboxing or virtual machines.
        """
    )

    path = st.sidebar.file_uploader("Upload a CSV file.", type="csv", key="file_uploader")
    if path is not None:
        df = pd.read_csv(path)
        st.dataframe(df,height=250)

    else:
        df = pd.read_csv('train.csv')
        st.dataframe(df,height=250)

    if st.chat_input("Enter your query:", key="query_input"):
        query_str = st.session_state.query_input
        response = agent(query_str,df)
        st.write(response)
            
        

main()