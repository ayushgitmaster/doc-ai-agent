from langchain.chains import RetrievalQA

def get_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever.as_retriever(),
        chain_type="stuff"
    )
