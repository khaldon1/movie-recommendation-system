import os
from pathlib import Path

import pandas as pd
import gradio as gr
import chromadb

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ============================================================
# Project Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
DATA_PATH = BASE_DIR / "data" / "processed" / "movie_nlp_ready.csv"
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"


# ============================================================
# Load Environment Variables
# ============================================================
print("Looking for .env file at:", ENV_PATH)
print(".env file exists:", ENV_PATH.exists())

load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = OPENROUTER_API_KEY.strip()

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing. Please add it to your .env file."
    )

print("OpenRouter API key loaded successfully!")


# ============================================================
# OpenRouter Client
# ============================================================
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

LLM_MODELS = [
    "openrouter/free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "deepseek/deepseek-r1:free",
]


# ============================================================
# Load Dataset
# ============================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Please make sure movie_nlp_ready.csv exists in data/processed/."
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    "title",
    "director",
    "genre",
    "release_year",
    "runtime",
    "rating",
    "metascore",
    "gross",
    "normalized_text",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# Helper Functions
# ============================================================
def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


# ============================================================
# Load Embedding Model
# ============================================================
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded successfully!")


# ============================================================
# ChromaDB Setup
# ============================================================
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection_name = "imdb_movies"

metadata_keys = [
    "title",
    "director",
    "genre",
    "release_year",
    "runtime",
    "rating",
    "metascore",
    "gross",
]


def build_chroma_collection():
    print("Building ChromaDB collection...")

    try:
        chroma_client.delete_collection(name=collection_name)
        print("Old ChromaDB collection deleted.")
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)

    documents = df["normalized_text"].astype(str).tolist()
    ids = [f"movie_{i}" for i in range(len(df))]

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True,
    ).tolist()

    metadatas = []
    for _, row in df.iterrows():
        metadatas.append(
            {
                "title": str(row["title"]),
                "director": str(row["director"]),
                "genre": str(row["genre"]),
                "release_year": safe_int(row["release_year"]),
                "runtime": safe_float(row["runtime"]),
                "rating": safe_float(row["rating"]),
                "metascore": safe_float(row["metascore"]),
                "gross": safe_float(row["gross"]),
            }
        )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("ChromaDB collection built successfully!")
    print("Total movies in collection:", collection.count())

    return collection


def load_or_rebuild_chroma_collection():
    try:
        collection = chroma_client.get_collection(name=collection_name)
        collection_count = collection.count()

        needs_rebuild = collection_count != len(df)

        if collection_count > 0:
            sample = collection.peek(limit=1)
            sample_metadata = sample.get("metadatas", [{}])[0]

            for key in metadata_keys:
                if key not in sample_metadata:
                    needs_rebuild = True
                    break
        else:
            needs_rebuild = True

        if needs_rebuild:
            print("Existing ChromaDB collection is outdated. Rebuilding...")
            collection = build_chroma_collection()
        else:
            print("Existing ChromaDB collection loaded successfully!")
            print("Total movies in collection:", collection.count())

        return collection

    except Exception:
        return build_chroma_collection()


movie_collection = load_or_rebuild_chroma_collection()


# ============================================================
# Retrieval Function
# ============================================================
def chroma_movie_search(query, top_k=5):
    top_k = int(top_k)
    top_k = max(1, min(top_k, 10))

    query_embedding = embedding_model.encode([query])

    results = movie_collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
    )

    rows = []

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        document = results["documents"][0][i]
        distance = results["distances"][0][i]

        rows.append(
            {
                "Title": metadata.get("title", "Unknown"),
                "Director": metadata.get("director", "Unknown"),
                "Genre": metadata.get("genre", "Unknown"),
                "Year": metadata.get("release_year", "Unknown"),
                "Runtime": metadata.get("runtime", "N/A"),
                "IMDb Rating": metadata.get("rating", "Unknown"),
                "Metascore": metadata.get("metascore", "Unknown"),
                "Gross": metadata.get("gross", "Unknown"),
                "Distance": round(distance, 4),
                "Document": document,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Context Builder
# ============================================================
def build_movie_context(results_df):
    context = ""

    for i, row in results_df.iterrows():
        context += f"Movie {i + 1}:\n"
        context += f"Title: {row['Title']}\n"
        context += f"Director: {row['Director']}\n"
        context += f"Genre: {row['Genre']}\n"
        context += f"Release Year: {row['Year']}\n"
        context += f"Runtime: {row['Runtime']} minutes\n"
        context += f"IMDb Rating: {row['IMDb Rating']}\n"
        context += f"Metascore: {row['Metascore']}\n"
        context += f"Gross Revenue: {row['Gross']}\n"
        context += f"Movie Text: {row['Document']}\n"
        context += "-" * 70 + "\n"

    return context


# ============================================================
# Prompt Builder
# ============================================================
def build_rag_prompt(query, context):
    prompt = f"""
You are an intelligent movie recommendation assistant.

You must answer using ONLY the retrieved movie context below.
Do not invent plot details, themes, actors, awards, or story information that are not present in the context.

User Query:
{query}

Retrieved Movie Context:
{context}

Instructions:
1. Recommend the most relevant movies from the retrieved context.
2. For each movie, mention only the available information:
   title, director, genre, release year, runtime, rating, metascore, and gross revenue.
3. Explain why each movie matches the user's request based only on metadata.
4. If plot or review details are not available, clearly say that the dataset does not include plot/review information.
5. Do not recommend movies outside the retrieved context.
6. Format the answer using clean markdown.

Final Answer:
"""
    return prompt


# ============================================================
# LLM Function with Fallback
# ============================================================
def generate_llm_response(prompt):
    last_error = None

    for model_name in LLM_MODELS:
        try:
            print(f"Trying model: {model_name}")

            completion = openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful movie recommendation assistant. "
                            "Answer only using the retrieved movie context."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.5,
                max_tokens=700,
            )

            answer = completion.choices[0].message.content

            if answer is None or str(answer).strip() == "":
                raise ValueError("The model returned an empty response.")

            print(f"Model used successfully: {model_name}")
            return answer

        except Exception as e:
            print(f"Model failed: {model_name}")
            print("Error:", e)
            print("-" * 80)
            last_error = e

    return f"""
## LLM Error

All free OpenRouter models failed.

**Last error:**

```text
{last_error}
```

Please try again later or use another OpenRouter model.
"""


# ============================================================
# Full RAG Pipeline
# ============================================================
def rag_movie_recommendation(user_query, top_k=5):
    if user_query is None or user_query.strip() == "":
        return "Please enter a movie recommendation query.", pd.DataFrame()

    try:
        retrieved_movies = chroma_movie_search(user_query, top_k=top_k)

        display_movies = retrieved_movies[
            [
                "Title",
                "Director",
                "Genre",
                "Year",
                "Runtime",
                "IMDb Rating",
                "Metascore",
                "Gross",
                "Distance",
            ]
        ].copy()

        context = build_movie_context(retrieved_movies)
        prompt = build_rag_prompt(user_query, context)
        answer = generate_llm_response(prompt)

        final_answer = f"""
# AI Movie Recommendations

**Query:** {user_query}

{answer}

---

**Dataset Note:** This answer is based on retrieved IMDb metadata only. The dataset does not include full plots or user reviews.
"""

        return final_answer, display_movies

    except Exception as e:
        return f"Application Error:\n\n{e}", pd.DataFrame()


# ============================================================
# Clean Professional Gradio UI
# ============================================================
custom_css = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background: #f7f8fb !important;
}

#hero {
    background: #111827;
    color: #ffffff;
    border-radius: 18px;
    padding: 30px 28px;
    margin: 12px 0 24px 0;
    border: 1px solid #1f2937;
}

#hero h1 {
    margin: 0 0 8px 0;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.4px;
}

#hero p {
    margin: 6px 0;
    color: #d1d5db;
    font-size: 15px;
}

.badges {
    margin-top: 16px;
}

.badge {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px 5px 0 0;
    border-radius: 999px;
    background: #1f2937;
    color: #e5e7eb;
    border: 1px solid #374151;
    font-size: 13px;
    font-weight: 600;
}

.section-title {
    color: #111827;
    font-weight: 800;
}

textarea, input {
    border-radius: 10px !important;
}

button {
    border-radius: 10px !important;
    font-weight: 700 !important;
}

#footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 18px;
    padding: 12px;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="gray",
).set(
    body_background_fill="#f7f8fb",
    block_background_fill="#ffffff",
    block_border_color="#e5e7eb",
    block_label_text_color="#111827",
    input_background_fill="#ffffff",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#e5e7eb",
    button_secondary_background_fill_hover="#d1d5db",
    button_secondary_text_color="#111827",
)


with gr.Blocks(
    theme=theme,
    css=custom_css,
    title="IMDb Movie Recommendation System",
) as demo:
    gr.HTML(
        """
        <div id="hero">
            <h1>IMDb Movie Recommendation System</h1>
            <p>Clean AI-powered movie recommendations using ML, DL, ChromaDB, OpenRouter, and RAG.</p>
            <p>Type a natural language request and get grounded recommendations based on retrieved IMDb metadata.</p>
            <div class="badges">
                <span class="badge">Machine Learning</span>
                <span class="badge">Deep Learning</span>
                <span class="badge">ChromaDB</span>
                <span class="badge">OpenRouter</span>
                <span class="badge">RAG</span>
                <span class="badge">Gradio</span>
            </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("## Search Settings")

            query_input = gr.Textbox(
                label="Movie Query",
                placeholder="Example: recommend me old classic movies with good ratings",
                lines=3,
            )

            top_k_input = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Number of Recommendations",
            )

            with gr.Row():
                submit_btn = gr.Button("Generate Recommendations", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

            gr.Examples(
                examples=[
                    ["emotional drama movies with strong story", 5],
                    ["recommend me old classic movies with good ratings", 5],
                    ["recommend me action crime thriller movies", 5],
                    ["funny comedy movies with high rating", 5],
                    ["high rated drama romance movies", 5],
                ],
                inputs=[query_input, top_k_input],
            )

        with gr.Column(scale=7):
            gr.Markdown("## AI Recommendation Response")
            answer_output = gr.Markdown(
                value="Your AI recommendation will appear here after you submit a query."
            )

    gr.Markdown("## Retrieved Movies")

    retrieved_movies_output = gr.Dataframe(
        label="Top Retrieved Movies from ChromaDB",
        interactive=False,
        wrap=True,
    )

    gr.HTML(
        """
        <div id="footer">
            Pipeline: User Query -> Embedding -> ChromaDB Retrieval -> RAG Context -> OpenRouter LLM -> Final Recommendation
        </div>
        """
    )

    submit_btn.click(
        fn=rag_movie_recommendation,
        inputs=[query_input, top_k_input],
        outputs=[answer_output, retrieved_movies_output],
    )

    clear_btn.click(
        fn=lambda: (
            "",
            5,
            "Your AI recommendation will appear here after you submit a query.",
            pd.DataFrame(),
        ),
        inputs=[],
        outputs=[query_input, top_k_input, answer_output, retrieved_movies_output],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(show_error=True)
