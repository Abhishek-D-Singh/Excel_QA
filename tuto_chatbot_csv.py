#pip install streamlit langchain openai faiss-cpu tiktoken
#pip install streamlit langchain openai faiss-cpu tiktoken

import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
import tempfile
import pandas as pd

user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your OpenAI API key, sk-",
    type="password")

uploaded_file = st.sidebar.file_uploader("Upload", type=["csv", "xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    demo_ex_loader = pd.read_excel(uploaded_file, engine="openpyxl")
    demo_ex_loader.to_csv("file1.csv", encoding='utf-8', index=True)
    loader = pd.read_csv(tmp_file_path, encoding="utf-8")

    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])

    # Create a question-answering chain using the index
    chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

    # Pass a query to the chain
    query = "Do you have a column called age?"
    response = chain({"question": query})

    print(response['result'])
    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(loader, embeddings)
    chain = ConversationalRetrievalChain.from_llm(llm=OpenAI(), retriever=vectors.as_retriever())

    #def conversational_chat(query):
        #result = chain({"question": query, "chat_history": st.session_state['history']})
        #st.session_state['

#import streamlit as st
#from streamlit_chat import message
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.chat_models import ChatOpenAI
#from langchain.chains import ConversationalRetrievalChain
#from langchain.document_loaders.csv_loader import CSVLoader
#from langchain.indexes import VectorstoreIndexCreator
#from langchain.chains import RetrievalQA
#from langchain.llms import OpenAI
#from langchain.vectorstores import FAISS
#import tempfile
#import pandas as pd

#user_api_key = st.sidebar.text_input(
   # label="#### Your OpenAI API key 👇",
  #  placeholder="Paste your openAI API key, sk-",
 #   type="password")

#uploaded_file = st.sidebar.file_uploader("upload", type=["csv","xlsx"])

#if uploaded_file :
 #   with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
  #      tmp_file.write(uploaded_file.getvalue())
   #     tmp_file_path = tmp_file.name

    #demo_ex_loader = pd.read_excel(uploaded_file, engine="openpyxl")
    #demo_ex_loader.to_csv("file1.csv", encoding='utf-8', index=True) 
    #loader = CSVLoader(file_path="file1.csv", encoding="utf-8")
    #data = loader.load()

    #index_creator = VectorstoreIndexCreator()
    #docsearch = index_creator.from_loaders([loader])

    # Create a question-answering chain using the index
    #chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

    # Pass a query to the chain
    #query = "Do you have a column called age?"
    #response = chain({"question": query})

    #print(response['result'])
    #embeddings = OpenAIEmbeddings()
    #vectors = FAISS.from_documents(data, embeddings)
    #chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=user_api_key),
                                                 #retriever=vectors.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]
        
    # Container for the chat history
    response_container = st.container()
    # Container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
