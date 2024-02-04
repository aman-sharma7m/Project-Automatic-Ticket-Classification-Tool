from user_utils import *
import streamlit as st 
from dotenv import load_dotenv

#getting keys 
load_dotenv()

def main():
  st.set_page_config(page_title='Automatic ticket tool')
  st.header('Automatic Ticket Classification Tool')
  st.write("We are here to help you. Ask your Q's, We got your A's")
  query=st.text_input('🔍')

  if query:

    #get_embedding instance
    embedding=get_embedding()

    #get index from pinecone
    index=get_index(embedding)

    #get relevant docs
    docs=get_relevant_docs(index,query)

    #get llm answer
    answer=get_llm_ans(docs,query)

    st.write('Answer:')
    st.write(answer)


if __name__=='__main__':
  main()
    