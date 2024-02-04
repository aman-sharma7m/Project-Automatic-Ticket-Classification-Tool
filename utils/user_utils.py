from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone 
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import joblib

def get_embedding():
  return SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-V2')


def get_index(embedding):
  index_name='pdf-store'
  return Pinecone.from_existing_index(index_name,embedding)

def get_relevant_docs(index,query,k=2):
  return index.similarity_search(query,k)

def get_llm_ans(docs,query):
  chain=load_qa_chain(OpenAI(),chain_type='map_reduce',verbose=True)
  return chain.run(input_documents=docs,question=query)

def predict(query):
  model=joblib.load('./models/model_1.pkl')
  result=model.predict([query])
  return result


