from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.create_documents([text])
