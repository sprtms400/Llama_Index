import os, streamlit as st

# Uncomment to specify your OpenAI API key here (local testing only, not in production!), or add corresponding environment variable (recommended)
os.environ['OPENAI_API_KEY']= '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'

# from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
## importError: cannot import name 'GPTSimpleVectorIndex' from 'llama_index' >>>> https://github.com/jerryjliu/llama_index/issues/1900
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.llms.openai import OpenAI

# Define a simple Streamlit app
st.title("Ask Llama")
query = input("What would you like to ask? (source: data/paul_graham_essay.txt) : ");

try:
    # This example uses text-davinci-003 by default; feel free to change if desired
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

    # Configure prompt parameters and initialise helper
    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    # Load documents from the 'data' directory !!
    documents = SimpleDirectoryReader('data').load_data()
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    # response = index.query(query)
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")