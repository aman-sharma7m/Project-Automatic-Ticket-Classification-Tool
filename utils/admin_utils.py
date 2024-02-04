from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone 
from pinecone import Pinecone as pc ,PodSpec 
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2



############## loading the documents function######################
def read_pdf(pdf_file):
  pdf_obj=PyPDF2.PdfReader(pdf_file)

  text=''

  for page in pdf_obj.pages:
    page_text=page.extract_text()
    text+=page_text
  
  docs=Document(page_content=text)
  return docs

def chunking(docs):
  text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
  texts=text_splitter.split_documents([docs])
  return texts


def get_embeddings():
  embeddings=SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-V2')
  return embeddings

def store_embeddings(chunk_text,embeddings):
  pc_config=pc()
  index_name='pdf-store'

  for name in pc_config.list_indexes().names():
    if name!=index_name:
      try:
        pc_config.delete_index(name)
        print(f'delete index {index_name}')
      except Exception as e:
        print('no index is there')
  
  if pc_config.list_indexes().names()==[]:
    print('creating new index')
    pc_config.create_index(name=index_name,
                    dimension=embeddings.client.get_sentence_embedding_dimension(),
                    metric='dotproduct',
                    spec=PodSpec(environment='gcp-starter'))

      
  Pinecone.from_documents(chunk_text,embeddings,index_name=index_name)

###################################################################