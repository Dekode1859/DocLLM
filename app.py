import os
import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub

def make_vectorstore(embeddings):
    loader = PyPDFDirectoryLoader("data") 
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    docsearch = FAISS.from_documents(texts, embeddings)
    
    return docsearch

def get_conversation(vectorstore, model):
    
    memory = ConversationBufferMemory(memory_key="messages", return_messages=True)

    conversation_chain = RetrievalQA.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory)
    
    return conversation_chain

def get_response(conversation_chain, query):
    # get the response
    response = conversation_chain.invoke(query)
    response = response["result"]
    answer = response.split('\nHelpful Answer: ')[1]
    return answer

def main():
    st.title("Chat LLM")
    # create a folder named data
    if not os.path.exists("data"):
        os.makedirs("data")
    
    print("Downloading Embeddings Model")
    with st.spinner('Downloading Embeddings Model...'):
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs = {'device': 'cpu'})
    
    print("Loading LLM from HuggingFace")
    with st.spinner('Loading LLM from HuggingFace...'):
        llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature":0.7, "max_new_tokens":512, "top_p":0.95, "top_k":50})
    
    # multiple pdfs uploader in the side bar
    st.sidebar.title("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with open(f"data/{file.name}", "wb") as f:
                f.write(file.getbuffer())
        with st.spinner('making a vectorstore database...'):
            vectorstore = make_vectorstore(embeddings)
        with st.spinner('making a conversation chain...'):
            conversation_chain = get_conversation(vectorstore, llm)
        st.sidebar.success("PDFs uploaded successfully")
    else:
        st.sidebar.warning("Please upload PDFs")
    # add a clear chat button which will clear the session state
    if st.button("Clear Chat"):
        st.session_state.messages = []
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("bot").markdown(message["content"])
    
    user_prompt = st.chat_input("ask a question", key="user")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        response = get_response(conversation_chain, user_prompt)
        st.chat_message("bot").markdown(response)
        st.session_state.messages.append({"role": "bot", "content": response})
        
if __name__ == "__main__":
    main()