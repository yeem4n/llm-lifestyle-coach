#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import regex as re
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph

from data_processing import format_datum_column, merge_activities

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
system_prompt_fp = "prompt_templates/system_prompt.txt"
user_prompt_fp = "prompt_templates/user_prompt.txt"

# Load questions from a CSV file as pandas DataFrame
def load_questions(filepath: str) -> tuple[pd.DataFrame, list[str]]:
    """Load the questions dataset from CSV."""
    df = pd.read_csv(filepath, sep=',')
    print(df.head())
    print(f"Loaded {len(df)} questions.")
    questions = df['question'].tolist()
    return df, questions

# Extract user IDs from filenames in a directory
def get_user_ids_from_directory(directory: str) -> list[str]:
    """Extract user IDs from filenames in the given directory.
    Assumes filenames are in the format 'user_id.csv'.

    Args:
        directory (str): Path to the directory containing user files.
    Returns:
        list[str]: List of user IDs extracted from filenames.
    """
    user_ids = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            user_id = filename.split('.')[0]
            user_ids.append(user_id)
    print(f"Found {len(user_ids)} user files.")
    return user_ids


# Scrape documents from kanker.nl
def scrape_rag_docs(database_path: str) -> list[str]:
    """Load documents from a web source and create a vector database."""
    

    df = pd.read_csv(database_path, sep=',')
    urls = df['url'].tolist()
    
    bs4_strainer = bs4.SoupStrainer(class_=("text-formatted field field--name-text-element-body field--type-text-long field--label-hidden field__item"))
    loader = WebBaseLoader(
        web_paths=(urls),
        bs_kwargs={"parse_only": bs4_strainer},
        )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from the web source.")
    return docs

# Create a vector database from the documents
def create_vector_database(docs: list[str]) -> InMemoryVectorStore:
    """Create a vector database from the provided documents.
    Args:
        docs (list[str]): List of documents to be embedded.
    Returns:
        InMemoryVectorStore: Vector store containing the embedded documents."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # chunk size (characters)
        chunk_overlap=200, # chunk overlap (characters)
        add_start_index=True, # track index in original document
        )
    
    split_docs = text_splitter.split_documents(docs)

    vectorstore = InMemoryVectorStore.from_documents(
       documents = split_docs,
        embedding = AzureOpenAIEmbeddings(
            model = "text-embedding-3-small",
            api_key = api_key,
            api_version = "2024-07-01-preview",
            azure_endpoint = azure_endpoint
    ),
)

    print(f"Loaded {len(split_docs)} documents into the vector store.")
    return vectorstore

# Load system and user prompts from files
def load_prompt(system_prompt_fp: str, user_prompt_fp) -> ChatPromptTemplate:
    """Load system and user prompts from files and combine them into a ChatPromptTemplate."""  
    with open(system_prompt_fp, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open(user_prompt_fp, "r", encoding="utf-8") as f:
        user_prompt = f.read()
    # Combine system and user prompt
    full_prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    print("Loaded system and user prompts.")
    return full_prompt

# Initalize the LLM using langchain
llm = init_chat_model(
    "gpt-4o", 
    model_provider="azure_openai", 
    api_key = api_key,
    api_version = "2024-12-01-preview",
    azure_endpoint = azure_endpoint)

class State(TypedDict):
    """State for the question-answering system."""
    user_id: str
    question: str
    table: str
    tabledf: pd.DataFrame
    context: List[Document]
    classic_rag: dict
    answer: str
    token_count: dict

def retrieve_userdata(state: State):
    reformatted_table = state["tabledf"].to_markdown()
    # reformatted_table = df_to_json(df)
    return {"table": reformatted_table}

def retrieve_context(state: State):
    retrieved_docs = vectorstore.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    unstructured_rag = {"urls": "\n\n".join(doc.metadata['source'] for doc in state["context"]), 
                       "content": docs_content}
    prompt = load_prompt(system_prompt_fp, user_prompt_fp)
    messages = prompt.invoke({"question": state["question"], "table": state["table"], "context": docs_content})
    response = llm.invoke(messages)
    token_usage = response.usage_metadata
    return {"answer": response.content, "token_count": token_usage, "classic_rag": unstructured_rag}


def add_dict_to_df(df, new_dict):
    # This creates a new DataFrame with one row from the dictionary
    # and concatenates it with the existing DataFrame
    return pd.concat([df, pd.DataFrame([new_dict])], ignore_index=True)

vectorstore = create_vector_database(scrape_rag_docs("../data/rag_database_kankernl.csv"))

def main():
    questions_filepath = "../data/final_questions.csv"
    userdata_directory = "../data/userdata/"
    results_fp = userdata_directory+ "./results/results_finaldataset.csv"

    # Load data
    selected_ids = []
    
    questions_df, selected_questions = load_questions(questions_filepath)
    user_ids = get_user_ids_from_directory(userdata_directory)
    for user_id in user_ids:
        if str(user_id) in user_ids:
            # if the user is in our user dataset
            # print(user_id, question)
            selected_ids.append(user_id)

    count = 0

    graph_builder = StateGraph(State).add_sequence([retrieve_userdata, retrieve_context, generate]) 
    graph_builder.add_edge(START, "retrieve_userdata") 
    graph = graph_builder.compile() 

    for user, input_q in zip(selected_ids, selected_questions):
        filepath = userdata_directory + str(user) + '.csv'
        tabledf = (
            pd.read_csv(filepath, sep=",")
            .head(21)                                 # Only first 21 days
            .dropna(how="all", axis=1)              # Drop columns where all values in the row are NaN
            .fillna("nan")                          # Fill remaining NaNs with 'nan'
            .pipe(merge_activities)                 # Merge activities into one column
            .pipe(format_datum_column)             # Reformat 'Datum' column with Dutch date format and days of week
            .set_index("Datum")                     # Set 'Datum' column as index
            )
    

        result = graph.invoke({"user_id": user,"question": input_q, "tabledf": tabledf})
        if count == 0:
            df_result = pd.DataFrame([result], columns=['user_id','question', 'answer', 'token_count', 'classic_rag', 'table'])
        else:
            df_result = add_dict_to_df(df_result, result)
        count+=1

    # print(f'Token count: {result["token_count"]}\n\n')
    # print(f'Table: {result["table"]}\n\n')
    # print(f'Context: {result["context"]}\n\n')
        # print(f'Question: {result["question"]}\n')
        # print(f'Answer: {result["answer"]}')

    print(f"Processed {count} questions for {len(selected_ids)} users.")
    df_result.to_csv(results_fp, index=False)

if __name__ == "__main__":
    main()
