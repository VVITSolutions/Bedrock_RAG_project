import json
import os
import sys
import boto3
import streamlit as st


from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate  
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock
)


## Data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data") #folder path of pdf files
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    from langchain_aws import ChatBedrock

    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",   # or haiku, opus, etc.
        # model_id="ai21.j2-mid-v1",                          
        client=bedrock,
        model_kwargs={"max_tokens": 512}
    )
    return llm

def get_llama2_llm():
    from langchain_aws import ChatBedrock

    llm = ChatBedrock(
        model_id="meta.llama3-70b-instruct-v1:0",   
        client=bedrock,
        model_kwargs={"max_gen_len": 512}
    )
    return llm

prompt_template = """Use the following pieces of context to provide a concise answer to the question at the end but use at least summarize with 250 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.

<context>
{context}
</context>

Question: {input}

Assistant:"""

# Now use ChatPromptTemplate (handles {input} for questions, {context} for docs)
PROMPT = ChatPromptTemplate.from_template(prompt_template)

def get_response_llm(llm, vectorstore_faiss, query):
    # Document chain: Stuff context into prompt + run LLM
    doc_chain = create_stuff_documents_chain(llm, PROMPT)

    # Retrieval chain: Retrieve docs + pass to doc_chain
    retrieval_chain = create_retrieval_chain(
        vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        doc_chain
    )

    # Run it (returns dict with 'answer' and 'context')
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]


def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with Vector store using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_claude_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_llama2_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()


#usage: streamlit run pdf_app.py