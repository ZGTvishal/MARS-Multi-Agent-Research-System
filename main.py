from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(load_max_docs=2, get_full_documents=False)
docs = retriever.invoke("transformer architecture")
print(docs[0].metadata)
print(docs[0].page_content[:200])