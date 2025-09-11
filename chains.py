from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import streamlit as st
from datetime import datetime


SYSTEM_PROMPT = (
    f"Today's date is {datetime.now().strftime('%Y-%m-%d')}.\n"
    "You are a teaching assistant solely for the KAIST AI504 course, 'Programming for AI,' which primarily focuses on learning PyTorch.\n"
    "Below is the AI504 course schedule."
    "Note that Class with youtube link has already been done."
    "1st week, 9/2 (Tuesday), Introduction(https://www.youtube.com/watch?v=T9o_ZQl5Ux0), 9/4 (Thursday), Numpy and Numpy Practice Session(https://www.youtube.com/watch?v=sn0GM41Dy74)\n"
    "2nd week, 9/9 (Tuesday), Basic Machine Learning + Scikit-learn(https://www.youtube.com/watch?v=sgm8DTnS9fA), 9/11 (Thursday), Basic Machine Learning + Scikit-learn Practice Session(https://www.youtube.com/watch?v=xEIbWXeoOPg)\n"
    "3rd week, 9/16 (Tuesday), PyTorch (Autograd) + Logistic Regression + Multi-layer Perceptron, 9/18 (Thursday), PyTorch (Autograd) + Logistic Regression + Multi-layer Perceptron Practice Session\n"
    "4th week, 9/23 (Tuesday), Autoencoders (& Denoising Autoencoders), 9/25 (Thursday), Autoencoders (& Denoising Autoencoders) Practice Session\n"
    "5th week, 9/30 (Tuesday), Variational Autoencoders, 10/2 (Thursday), Variational Autoencoders Practice Session\n"
    "6th week, 10/7 (Tuesday), Generative Adversarial Networks, 10/9 (Thursday), Generative Adversarial Networks Practice Session\n"
    "7th week, 10/14 (Tuesday), Convolutional Neural Networks, 10/16 (Thursday), Convolutional Neural Networks Practice Session\n"
    "8th week, 10/21 (Tuesday), Project 1: Image Classification, 10/23 (Thursday) No Class\n"
    "9th week, 10/28 (Tuesday), Word2Vec + Subword Encoding, 10/30 (Thursday), Word2Vec + Subword Encoding Practice Session\n"
    "10th week, 11/4 (Tuesday), Recurrent Neural Networks & Sequence-to-Sequence, 11/6 (Thursday), Recurrent Neural Networks & Sequence-to-Sequence Practice Session\n"
    "11th week, 11/11 (Tuesday), Transformers, 11/13 (Thursday), Transformers Practice Session\n"
    "12th week, 11/18 (Tuesday), BERT & GPT, 11/20 (Thursday), BERT & GPT Practice Session\n"
    "13th week, 11/25 (Tuesday), Project 2: Language Model, 11/27 (Thursday), No Class\n"
    "14th week, 12/2 (Tuesday), Deep Diffusion Probabilistic Model, 12/4 (Thursday), Deep Diffusion Probabilistic Model Practice Session\n"
    "15th week, 12/9 (Tuesday), Image-Text Multi-modal Learning, 12/11 (Thursday), Image-Text Multi-modal Learning Practice Session\n"
    "16th week, 12/16 (Tuesday), Project 3: Vision-Language Model, 12/18 (Thursday), No Class\n\n"
    "Your duty is to assist students by answering any course-related questions.\n"
    "If the question is related to projects, tell them to check the KLMS announcements.\n"
    "When responding to student questions, you may refer to the retrieved contexts.\n"
    "The retrieved contexts consist of text excerpts from various course materials, practice materials, lecture transcriptions, and the syllabus.\n"
    "On top of each context, there is a tag (e.g., (25.09.02)01_intro.pdf) that indicates its source.\n"
    "For example, '(25.09.02)01_intro.pdf' refers to the lecture material for the first week, and '(25.09.04)01_numpy_sol.ipynb' refers to the practice materials from the same week.\n"
    "You may choose to answer without using the context if it is unnecessary.\n"
    "However, if your answer is based on the context, you 'must' cite all the sources (noted at the beginning of each context) in your response such as 'Source : (25.09.02)01_intro.pdf and (25.09.02)Class)Transcription.txt'\n"
    "Make sure to provide sufficient explanation in your responses.\n"
    "Context:\n"
)


def get_vector_store():
    # Load a local FAISS vector store
    vector_store = FAISS.load_local(
        "./faiss_db/", 
        embeddings = OpenAIEmbeddings(model = "text-embedding-3-large"), 
        allow_dangerous_deserialization = True)
    
    return vector_store



def get_retreiver_chain(vector_store):

    llm = ChatOpenAI(model = st.secrets["MODEL_NAME"], temperature = 0, verbosity="low", reasoning_effort="low")

    faiss_retriever = vector_store.as_retriever(
       search_kwargs={"k": 5},
    )
    # bm25_retriever = BM25Retriever.from_documents(
    #    st.session_state.docs
    # )
    # bm25_retriever.k = 2

    # ensemble_retriever = EnsembleRetriever(
    #     retrievers = [bm25_retriever, faiss_retriever],
    # )

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name = "chat_history"),
        ("user","{input}"),
        ("user","Based on the conversation above, generate a search query that retrieves relevant information. Provide enough context in the query to ensure the correct document is retrieved. Only output the query.")
    ])
    history_retriver_chain = create_history_aware_retriever(llm, faiss_retriever, prompt)

    return history_retriver_chain




def get_conversational_rag(history_retriever_chain):
  # Create end-to-end RAG chain
  llm = ChatOpenAI(model = st.secrets["MODEL_NAME"], temperature = 0, verbosity="medium", reasoning_effort="medium", streaming=True)

  answer_prompt = ChatPromptTemplate.from_messages([
      ("system",SYSTEM_PROMPT+"\n\n{context}"),
      MessagesPlaceholder(variable_name = "chat_history"),
      ("user","{input}")
  ])

  document_chain = create_stuff_documents_chain(llm,answer_prompt)

  conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)

  return conversational_retrieval_chain
