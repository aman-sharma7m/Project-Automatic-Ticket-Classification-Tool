from dotenv import load_dotenv
import streamlit as st
from admin_utils import *

#loading the keys
load_dotenv()

def main():
  st.set_page_config(page_title='Dump pdf to pinecone vector store')
  st.header('Please upload your pdf files here !!!!')

  file=st.file_uploader('Drop your file: (*.pdf)',type=['pdf'])

  if file is not None:
    with st.spinner('In progress......'):
      
      #reading 
      pdf_docs=read_pdf(file)
      st.write('👉 Reading the pdf file done')

      #create chunks
      chunk_text=chunking(pdf_docs)
      st.write('👉 Splitting data into chunks')

      #create embeddings
      embeddings=get_embeddings()
      st.write('👉 Create embedding instance done!!!!')

      #vector store
      st.write('👉 Uploading vectors to pinecone....')
      store_embeddings(chunk_text,embeddings)
      
    st.success('file upload to pinecone is done successfully!')





if __name__=='__main__':
  main()