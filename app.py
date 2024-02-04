from utils.user_utils import *
import streamlit as st 
from dotenv import load_dotenv

#getting keys 
load_dotenv()

if 'HR_tickets' not in st.session_state:
  st.session_state['HR_tickets']=[]
if 'IT_tickets' not in st.session_state:
  st.session_state['IT_tickets']=[]
if 'TRANS_tickets' not in st.session_state:
  st.session_state['TRANS_tickets']=[]

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

    button=st.button('Raise ticket?')

    if button:
      embedding=get_embedding()
      e_query=embedding.embed_query(query)
      query_result=predict(e_query)[0]
      st.write(f'Tickets has been assigned : {query_result}')

      if query_result=='HR':
        st.session_state['HR_tickets'].append(query)
      elif query_result=='IT':
        st.session_state['IT_tickets'].append(query)
      else:
        st.session_state['TRANS_tickets'].append(query)




if __name__=='__main__':
  main()
    