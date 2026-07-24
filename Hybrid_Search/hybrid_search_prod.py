import os
import tempfile
from dotenv import load_dotenv

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def create_knowledge_base(pdf_path):
    loader = PyPDFLoader(pdf_path)
    loaded_documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(loaded_documents)
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings_model, persist_directory=tempfile.mkdtemp())
    return vector_store,chunks

def retriver(vector_store,chunks):
    semantic_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(chunks,k=3)
    
    hybrid_retriever = EnsembleRetriever(retrievers=[semantic_retriever, bm25_retriever], weights=[0.7, 0.3])
    return hybrid_retriever


def demo_rag(pdf_path):
    vector_store, chunks = create_knowledge_base(pdf_path)
    retriever = retriver(vector_store, chunks)

    prompt =  ChatPromptTemplate.from_template(
        """
        Answer the question based only on the following context:
        {context}
        Question: {question}

        Answer:
        
        Make sure to answer in a concise manner, 
        and if you don't know the answer, just say "I don't know."""
        )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    question = ["What is the main topic of the research paper?",
                    "What are the key findings of the research paper?",
                    "What methodology was used in the research paper?",
                    "give the complete techinal flow of the research paper",]
    
    for q in question:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

if __name__ == "__main__":
    pdf_path = r"C:\Users\shiva\Downloads\Shivam_Sarang_UG_Research_Fellowship_2nd round_final.pdf"
    demo_rag(pdf_path)