# Langchain RAG system with Llama3 and ChromaDB

## Introduction
In this project, we implement a RAG system with Llama3 and ChromaDB. The RAG system is a system that can answer questions based on the given context. The RAG system is composed of three components: retriever, reader, and generator.
The retriever retrieves relevant documents from the given context. The reader reads the retrieved documents and generates the answer. The generator generates the answer based on the retrieved documents and the answer generated by the reader. In this project, we use Llama3 as the retriever and ChromaDB as the reader. We use the Llama3 to retrieve the relevant documents from the given context. We use the ChromaDB to read the retrieved documents and generate the answer. The RAG system is implemented in Python.

# Using FAISS for vector search
I have used FAISS for vector search instead of CHROME here in the repository because of the limitations of the CHROME. The CHROME is not able to handle the large documents and the large number of documents. The FAISS is able to handle the large documents and the large number of documents. The FAISS is a library for efficient similarity search and clustering of dense vectors

Just try both and see how they perform and then choose best.

# Important Links
- [Ollama](https://ollama.com/library)
- [Langchain HuggingFace Embedding](https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/)
- [Langchain Chrome](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/)
- [Langchain Retriever QA Chain](https://docs.smith.langchain.com/old/cookbook/hub-examples/retrieval-qa-chain)
- [Chunking large documents for vector search](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents)