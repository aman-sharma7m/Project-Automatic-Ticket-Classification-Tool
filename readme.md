# Project Automatic-Ticket-Classification-Tool

## Introduction

In this project, we will explore an automatic ticket classification tool that can help categorize and assign tickets based on user queries. The tool uses natural language processing techniques to analyze user queries and classify them into different categories such as HR, IT, or TRANS (Transportation).

## Installation & create environment

Clone the project

```bash
  git clone link_to_copy
```

Go to the project directory

```bash
  cd proj_dir
```

Create the enviroment

```bash
  conda create --prefix ./lang_env
  conda activate {path}/lang_env
  python -m ipykernel install --user --name=lang_env
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Insert Open-ai api key

```bash
  touch .env
  insert your key as following
  OPENAI_API_KEY=""
  HUGGINGFACEHUB_API_TOKEN=""
  PINECONE_API_KEY=""
```

Start the server

```bash
  streamlit run app.py
```

## Intro to libraries

Ticket Classification: Ticket classification is the process of categorizing user queries or tickets into different categories based on their content. This helps in efficiently assigning tickets to the appropriate department or team for resolution.

Natural Language Processing (NLP): NLP is a branch of artificial intelligence that focuses on the interaction between computers and human language. It involves techniques for understanding, analyzing, and generating human language.

Embeddings: Embeddings are numerical representations of words or sentences that capture their semantic meaning. They are used in NLP tasks such as text classification, information retrieval, and machine translation.

Pinecone: Pinecone is a vector database that allows fast and efficient similarity search on high-dimensional embeddings. It provides a scalable solution for storing and querying embeddings.

### Detailed Explanation:

app.py
This code is a script that creates a Streamlit application for an automatic ticket classification tool.

First, it imports the necessary modules: `user_utils` from the `utils` package, `streamlit`, and `load_dotenv` from the `dotenv` package.

Then, it calls the `load_dotenv()` function to load any environment variables from a `.env` file.

Next, it checks if the keys `'HR_tickets'`, `'IT_tickets'`, and `'TRANS_tickets'` are not already present in the `st.session_state` dictionary. If any of these keys are not present, it initializes them with empty lists.

After that, it defines the `main()` function, which is the entry point of the script.

Inside the `main()` function, it sets the page configuration for the Streamlit app and displays a header and a text input field for the user to enter their query.

If the user enters a query, the following steps are executed:

1. It calls the `get_embedding()` function to get an instance of the embedding model.
2. It calls the `get_index()` function to get the index from the Pinecone service.
3. It calls the `get_relevant_docs()` function to get the relevant documents from the index based on the user's query.
4. It calls the `get_llm_ans()` function to get the answer from the Language Model based on the relevant documents and the user's query.
5. It displays the answer to the user.
6. It displays a button labeled "Raise ticket?".
7. If the user clicks the button, the following steps are executed:
   - It calls the `get_embedding()` function again to get an instance of the embedding model.
   - It calls the `embed_query()` method of the embedding model to embed the user's query.
   - It calls the `predict()` function to predict the ticket category based on the embedded query.
   - It displays a message indicating the assigned ticket category.
   - It appends the user's query to the corresponding ticket category list in the `st.session_state` dictionary based on the predicted category.

Finally, it checks if the script is being run directly (i.e., not imported as a module) and calls the `main()` function if it is.

user_utils.py
The code provided is a Python script that imports several modules and defines several functions. Let's go through each part of the code and explain what it does.

First, the code imports the following modules:

- `SentenceTransformerEmbeddings` from `langchain.embeddings.sentence_transformer`
- `Pinecone` from `langchain.vectorstores`
- `load_qa_chain` from `langchain.chains.question_answering`
- `OpenAI` from `langchain.llms`
- `joblib`

Next, the code defines a function called `get_embedding()`. This function returns an instance of the `SentenceTransformerEmbeddings` class with the model name set to `'all-MiniLM-L6-V2'`. This function is used to get the sentence embeddings for the input text.

The next function defined is `get_index(embedding)`. This function takes an `embedding` parameter and returns an instance of the `Pinecone` class with the index name set to `'pdf-store'`. The `Pinecone` class is used for similarity search on the embeddings.

The next function defined is `get_relevant_docs(index, query, k=2)`. This function takes an `index`, `query`, and an optional `k` parameter (default value is 2). It performs a similarity search on the `index` using the `query` and returns the top `k` most similar documents.

The next function defined is `get_llm_ans(docs, query)`. This function takes `docs` and `query` parameters. It loads a question answering chain using the `load_qa_chain` function with an instance of the `OpenAI` class and the chain type set to `'map_reduce'`. It then runs the chain with the input documents set to `docs` and the question set to `query`. The function returns the result of the chain execution.

The last function defined is `predict(query)`. This function takes a `query` parameter. It loads a model from a file called `'./models/model_1.pkl'` using the `joblib.load()` function. It then uses the loaded model to predict the result for the input `query` and returns the result.

Overall, this code defines several functions that can be used for various natural language processing tasks such as getting sentence embeddings, performing similarity search, running question answering chains, and making predictions using a pre-trained model.

admin_utils.py
Loading the Documents
The first part of the code focuses on loading the text documents that we want to classify. It uses the PyPDF2 library to extract text from PDF files. The read_pdf function takes a PDF file as input and returns a Document object that contains the extracted text.

Next, the chunking function splits the document into smaller chunks of text. This is done to improve the efficiency of the embedding process and to handle long documents. The RecursiveCharacterTextSplitter class is used for this purpose.

The get_embeddings function initializes the SentenceTransformer model, which is a pre-trained model for generating sentence embeddings. Sentence embeddings are used to represent chunks of text in a numerical format.

Finally, the store_embeddings function stores the embeddings of the chunked text in the Pinecone vector store. It creates a new index if it doesn't exist or deletes the existing index and creates a new one.

Model Training
The second part of the code focuses on training the text classification model. It starts with the read_csv_file function, which reads a CSV file containing the training data. The CSV file should have two columns: "Query" and "Class". The "Query" column contains the text data, and the "Class" column contains the corresponding class labels.

The function performs some preprocessing steps, such as balancing the number of samples for each class and generating embeddings for the text data using the embed object.

Next, the train_model function splits the data into training and testing sets and trains a support vector machine (SVM) model using the StandardScaler for feature scaling. The model is returned along with the testing data.

Finally, the get_score function calculates the accuracy score of the trained model on the testing data.

## libraries

```
import streamlit as st
from dotenv import load_dotenv
import joblib
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as pc ,PodSpec
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
```

## Conclusion

we explored an automatic ticket classification tool that can help categorize and assign tickets based on user queries. The tool uses natural language processing techniques, embeddings, and Pinecone for efficient ticket classification and assignment. By automating this process, organizations can streamline their ticket management and improve customer support efficiency.
