import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Sidebar contents
with st.sidebar:
    st.title('🗨️ LLM Chat App 🗨️')
    st.write("Welcome to the LLM Chat App powered by LangChain and OpenAI!")
    st.write("About this App:")
    st.markdown(
        "This app allows you to chat with a language model powered by LLM. "
        "Ask questions and get answers from a large corpus of text data. "
        "Try it out and explore the capabilities of modern language models."
    )
    st.markdown("[Streamlit](https://streamlit.io/)")
    st.markdown("[LangChain](https://python.langchain.com/)")
    st.markdown("[OpenAI LLM](https://platform.openai.com/docs/models)")
    st.write("Made by [Sampik Kumar Gupta](https://www.linkedin.com/in/sampik-gupta-41544bb7/)")



os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


def main():
    st.header("Learn your PDF without reading all of it")
    
    #upload pdf
    pdf=st.file_uploader("Upload your PDF",type="pdf")
    
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        # splitting the document in smaller chunks
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,
                                                    length_function=len)

        chunks = text_splitter.split_text(text=text)
        

        #embeddings 
        store_name=pdf.name[:-4]

        #checking if embeddings file already exists in disk
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
        
        #creating file if not exists
        else:
            embeddings=OpenAIEmbeddings()
            VectorStore=FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)
        
        #accept user questions or queries
        query=st.text_input("Ask question about your file: ")

        if query:
            docs=VectorStore.similarity_search(query=query,k=3)
            llm=OpenAI(model_name = 'gpt-3.5-turbo')
            chain=load_qa_chain(llm=llm, chain_type="stuff")


            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs,question=query)
                print(cb)
            st.write(response)

if __name__=='__main__':
    main()
