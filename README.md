# Movie Recommendation System using ML, DL, LLM, and RAG

## Author
Khaled Metwally

## Project Overview
This project is an end-to-end AI-powered IMDb movie recommendation system.

It combines Data Preprocessing, Machine Learning, Deep Learning, NLP, Embeddings, ChromaDB Vector Database, LLM Integration using OpenRouter, Retrieval-Augmented Generation (RAG), and Gradio.

## Features
- Data cleaning and preprocessing
- Missing value handling
- Duplicate removal
- Outlier detection and handling
- Exploratory Data Analysis and visualization
- Feature engineering
- Regression models for IMDb rating prediction
- Classification models for high-rating prediction
- Hyperparameter tuning using RandomizedSearchCV
- Deep Learning using PyTorch MLP
- Text normalization and tokenization
- SentenceTransformer embeddings
- Semantic search
- ChromaDB vector database
- LLM integration using OpenRouter
- Full RAG pipeline
- Interactive Gradio web app

## Machine Learning Models
Regression:
- Linear Regression
- Random Forest Regressor
- Tuned Random Forest Regressor

Classification:
- Logistic Regression
- Random Forest Classifier
- Tuned Random Forest Classifier

## Deep Learning
A PyTorch Multi-Layer Perceptron (MLP) was implemented for binary classification.

The tuned Random Forest Classifier achieved the best performance overall, while the MLP added the deep learning component required for the project.

## RAG Pipeline
The RAG system works as follows:

1. Convert movie metadata into text.
2. Generate embeddings using SentenceTransformers.
3. Store movie embeddings and metadata in ChromaDB.
4. Retrieve top-k relevant movies based on user query.
5. Build a structured context.
6. Send the context to an LLM through OpenRouter.
7. Generate a grounded movie recommendation response.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch
- SentenceTransformers
- ChromaDB
- OpenRouter
- OpenAI Python SDK
- Gradio
- Jupyter Notebook

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/khaldon1/imdb-movie-recommendation-system.git
cd imdb-movie-recommendation-system
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Create Environment File

Create a file named `.env` in the project root directory.

Add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

> Do not upload the `.env` file to GitHub.

You can use `.env.example` as a template.

### 4. Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/imdb_complete_ai_project.ipynb
```

Run all cells from top to bottom.

### 5. Run the Gradio App

From the project root, go to the app folder:

```bash
cd app
```

Run the app:

```bash
python gradio_app.py
```

The Gradio app will start and provide a local URL in the terminal.

Open the URL in your browser and enter a movie recommendation query, such as:

```text
recommend me old classic movies with good ratings
```

## Limitations

- The dataset mainly contains movie metadata.
- The dataset does not include full plot descriptions or user reviews.
- The RAG response is grounded on metadata only.
- Free OpenRouter models may sometimes be rate-limited.


## Future Improvements

- Add movie plot descriptions.
- Add user reviews.
- Perform sentiment analysis.
- Deploy the Gradio app online.