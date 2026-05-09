# Project Explanation

## 1. Project Idea

This project is an AI-powered movie recommendation system.

The user can write a natural language request, such as:

> Recommend me old classic movies with good ratings.

The system then searches for the most relevant movies in the dataset and generates a clear recommendation answer.

The project combines Machine Learning, Deep Learning, NLP, Embeddings, RAG, LLMs, and a Gradio web interface.

---

## 2. Dataset and Data Preprocessing

The project uses an IMDb movie dataset that contains movie information such as:

- Movie title
- Director
- Genre
- Release year
- Runtime
- IMDb rating
- Metascore
- Gross revenue

Before building the models, the data was cleaned and prepared.

The preprocessing steps included:

- Handling missing values
- Removing duplicate records
- Converting numeric columns to the correct data types
- Handling outliers
- Preparing features for Machine Learning models
- Creating text data for NLP and RAG

This step is important because clean data helps the models make better predictions and recommendations.

---

## 3. Machine Learning Part

In the Machine Learning part, the project uses traditional ML models to analyze and predict movie-related information.

The project includes regression models for predicting IMDb ratings, such as:

- Linear Regression
- Random Forest Regressor
- Tuned Random Forest Regressor

It also includes classification models for predicting whether a movie has a high rating, such as:

- Logistic Regression
- Random Forest Classifier
- Tuned Random Forest Classifier

The models were evaluated using metrics such as:

- Mean Squared Error
- RMSE
- R2 Score
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The best model was selected based on performance.

---

## 4. Deep Learning Part

The project also includes a Deep Learning model using PyTorch.

A Multi-Layer Perceptron (MLP) was implemented for binary classification.

The goal of this part is to add a deep learning component to the system and compare it with traditional Machine Learning models.

Because the dataset mainly contains structured movie metadata, traditional models such as Random Forest may perform better than the MLP. This is normal because tree-based models often work very well with tabular data.

---

## 5. NLP and Text Processing

The movie metadata was converted into text so the system could understand movie information in a language-like format.

The text processing part included:

- Text normalization
- Tokenization
- Combining important movie features into one text field
- Preparing text for embeddings

This step helps the system search for movies based on meaning, not only exact keywords.

---

## 6. Embeddings and Semantic Search

The project uses SentenceTransformer embeddings to convert movie text into numerical vectors.

These vectors represent the meaning of each movie.

For example, if the user asks for:

> emotional drama movies with strong story

the system does not only search for the exact words. Instead, it searches for movies with similar meaning.

This makes the recommendation system smarter and more flexible.

---

## 7. RAG System

RAG stands for Retrieval-Augmented Generation.

In this project, the RAG pipeline works as follows:

1. The user writes a movie recommendation query.
2. The query is converted into an embedding.
3. The system searches the ChromaDB vector database.
4. The most relevant movies are retrieved.
5. The retrieved movie metadata is used as context.
6. The context is sent to the LLM.
7. The LLM generates a final recommendation answer.

This makes the answer grounded in the dataset instead of being generated randomly.

---

## 8. LLM Integration

The project integrates an LLM using OpenRouter.

The LLM receives the retrieved movie context and generates a human-friendly recommendation response.

The LLM is instructed to answer only using the retrieved movie information.

This helps reduce hallucination and keeps the answer connected to the actual dataset.

---

## 9. Gradio Demo Application

The project includes a Gradio web application.

The user can:

- Enter a natural language movie request
- Choose the number of recommendations
- View the AI-generated recommendation answer
- View the retrieved movies from the vector database

The Gradio app makes the project easier to test and present.

---

## 10. Project Limitations

The dataset mainly contains movie metadata.

It does not include full movie plots or user reviews.

Because of this:

- The RAG system is based on metadata only.
- The system cannot deeply analyze movie stories.
- Sentiment analysis was not fully implemented because there are no user reviews in the dataset.

A future improvement would be to add movie plots and user reviews, then use them for better recommendations and sentiment analysis.

---

## 11. Conclusion

This project is an end-to-end AI movie recommendation system.

It includes:

- Data preprocessing
- Machine Learning models
- Deep Learning using PyTorch MLP
- NLP preprocessing
- SentenceTransformer embeddings
- ChromaDB vector database
- RAG pipeline
- LLM integration using OpenRouter
- Gradio demo application

The final system can understand user queries, retrieve relevant movies, and generate clear recommendations using real dataset information.
