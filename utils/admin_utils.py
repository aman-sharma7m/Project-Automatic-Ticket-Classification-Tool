from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone 
from pinecone import Pinecone as pc ,PodSpec 
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split



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
  
#################model training#################################

def read_csv_file(csv_file,embed):
  #reading the file
  raw_data=pd.read_csv(csv_file,names=['Query','Class'])


  sample_data=[]

  #getting min count for samples
  min_sample_value=raw_data.Class.value_counts(ascending=True).values[0]

  for cls in raw_data.Class.unique():
    cls_data=raw_data[raw_data['Class']==cls]

    sample_class_data=cls_data.sample(n=min_sample_value,random_state=42)

    sample_data.append(sample_class_data)
  

  sample_df=pd.concat(sample_data)
  sample_df['embeddings']=sample_df['Query'].apply(lambda x:embed.embed_query(x))
  return sample_df

def train_model(df):
  x_train,x_test,y_train,y_test=train_test_split(list(df['embeddings']),list(df['Class']),test_size=0.2,random_state=42)

  pipeline=make_pipeline(StandardScaler(), SVC(class_weight='balanced'))

  pipeline.fit(x_train,y_train)

  return pipeline,x_test,y_test

def get_score(model,x_test,y_test):
  score=model.score(x_test,y_test)
  return score



  ################################################################