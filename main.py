from llm_model import generate_response
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader

st.title("News Research Tool")
st.sidebar.title("News Article URLs")
main_placeholder = st.empty()

# Initialize session state for data_dict if not present
if 'data_dict' not in st.session_state:
    st.session_state.data_dict = {
        'instruction': '',
        'input': '',
        'output': ''
    }

# Input URLs
urls = st.sidebar.text_input("URL")
process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked and urls:
    loader = UnstructuredURLLoader([urls])  # wrap in list
    main_placeholder.text("Data Loading... Started...")
    data = loader.load()

    # Optional: You can split the text using a splitter if needed
    # text_splitter = RecursiveCharacterTextSplitter(
    #     separators=['\n\n', '\n', '.', ','],
    #     chunk_size=1000
    # )
    # docs = text_splitter.split_documents(data)

    main_placeholder.text("Text Processing... Done")
    documents_text = "\n\n".join([dt.page_content for dt in data])
    st.session_state.data_dict.update({"input": documents_text})

# Input question
query = main_placeholder.text_input("Question:")

if query:
    st.session_state.data_dict.update({
        "instruction": str(query),
        "output": ""
    })
    st.header("Answer")
    response = generate_response(st.session_state.data_dict)
    st.write(response)
