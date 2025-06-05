from langchain.vectorstores import FAISS

def build_vectorstore(docs, embedder):
    return FAISS.from_documents(docs, embedder)
