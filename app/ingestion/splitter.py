from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def split_text(text, metadata=None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.create_documents([text], metadatas=[metadata] if metadata else None)
