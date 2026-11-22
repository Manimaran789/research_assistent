# Research Dataset Discovery Assistant
# Hugging Face Spaces deployment version


import re, requests, pandas as pd, numpy as np, gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print("Downloading/loading model (first run may take a few minutes)...")
embedder = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded.")

FALLBACK_DATASETS = [
    {"id":"uciml/iris","name":"Iris","description":"Iris flower measurements for predicting flower species","tags":["tabular","classification","biology"],"downloads":0,"likes":0,"lastModified":"","url":"https://huggingface.co/datasets/uciml/iris"},
    {"id":"imdb","name":"IMDb","description":"Movie reviews with sentiment labels for text classification","tags":["text","sentiment","classification"],"downloads":0,"likes":0,"lastModified":"","url":"https://huggingface.co/datasets/imdb"},
    {"id":"ag_news","name":"AG News","description":"News articles grouped into news categories for text classification","tags":["text","news","classification"],"downloads":0,"likes":0,"lastModified":"","url":"https://huggingface.co/datasets/ag_news"},
    {"id":"heart-disease-search","name":"Heart disease dataset search","description":"Search results for datasets about cardiovascular risk and heart disease prediction using health indicators","tags":["health","medical","classification"],"downloads":0,"likes":0,"lastModified":"","url":"https://huggingface.co/datasets?search=heart+disease"},
    {"id":"student-performance-search","name":"Student performance dataset search","description":"Datasets about student grades, attendance, study habits and academic performance prediction","tags":["education","tabular","regression"],"downloads":0,"likes":0,"lastModified":"","url":"https://huggingface.co/datasets?search=student+performance"},
    {"id":"fake-news-search","name":"Fake news dataset search","description":"Datasets with news text and labels for fake news and misinformation classification research","tags":["news","text","classification"],"downloads":0,"likes":0,"lastModified":"","url":"https://huggingface.co/datasets?search=fake+news"},
    {"id":"air-quality-search","name":"Air quality dataset search","description":"Environmental measurements including air pollution, particulate matter, temperature and humidity","tags":["environment","sensor","time-series"],"downloads":0,"likes":0,"lastModified":"","url":"https://huggingface.co/datasets?search=air+quality"},
    {"id":"customer-churn-search","name":"Customer churn dataset search","description":"Customer usage and account information for predicting churn in business or telecom","tags":["business","customer","classification"],"downloads":0,"likes":0,"lastModified":"","url":"https://huggingface.co/datasets?search=customer+churn"},
]


HF_API = "https://huggingface.co/api/datasets"

# Fix punctuation in the stopword string (normal Python string)
STOPWORDS = set("i want to build develop make a an the using based on for from with in of and or predict prediction detect detection analyze analysis study data dataset model machine learning ai ml project research into by about that this it people person".split())

def make_search_terms(query, max_terms=5):
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", query.lower())
    useful = [w for w in words if w not in STOPWORDS and len(w) > 2]
    terms = [query.strip()]
    if len(useful) >= 2:
        terms.append(" ".join(useful[:4]))
    terms.extend(useful[:max_terms])
    return list(dict.fromkeys(t for t in terms if t))

def fetch_hf_datasets(query, per_query=25, timeout=20):
    records, seen, errors = [], set(), []
    for term in make_search_terms(query):
        try:
            response = requests.get(
                HF_API,
                params={"search": term, "limit": per_query},
                timeout=timeout,
                headers={"User-Agent": "ResearchDatasetDiscoveryAssistant/1.0"}
            )
            response.raise_for_status()
            items = response.json()
            if not isinstance(items, list):
                continue
            for item in items:
                dataset_id = item.get("id") or item.get("name")
                if not dataset_id or dataset_id in seen:
                    continue
                seen.add(dataset_id)
                card = item.get("cardData") or {}
                description = card.get("description") or card.get("pretty_name") or "Dataset repository; open the dataset card for details."
                records.append({
                    "id": dataset_id,
                    "name": dataset_id.split("/")[-1],
                    "description": str(description),
                    "tags": item.get("tags") or [],
                    "downloads": item.get("downloads", 0) or 0,
                    "likes": item.get("likes", 0) or 0,
                    "lastModified": item.get("lastModified", "") or "",
                    "url": "https://huggingface.co/datasets/" + dataset_id
                })
        except Exception as exc:
            errors.append(f"{term}: {exc}")
    return records, errors


def dataset_text(row):
    tags = row.get("tags", [])
    if isinstance(tags, list):
        tags = " ".join(map(str, tags))
    return f"Dataset: {row.get('name', '')}. Description: {row.get('description', '')}. Tags: {tags}"

def rank_datasets(query, records):
    if not records:
        return pd.DataFrame()
    query_vec = embedder.encode([query], normalize_embeddings=True)
    dataset_texts = [dataset_text(r) for r in records]
    dataset_vecs = embedder.encode(dataset_texts, normalize_embeddings=True, show_progress_bar=False)
    semantic_scores = cosine_similarity(query_vec, dataset_vecs)[0]
    ranked = []
    for record, similarity in zip(records, semantic_scores):
        completeness = sum(bool(record.get(k)) for k in ["name", "description", "tags", "url"]) / 4
        score = 0.90 * float(similarity) + 0.10 * completeness
        ranked.append({
            **record,
            "semantic_similarity": float(similarity),
            "metadata_completeness": float(completeness),
            "ranking_score": float(score)
        })
    return pd.DataFrame(ranked).sort_values("ranking_score", ascending=False).reset_index(drop=True)


def discover(query, top_k=10):
    if not query or not query.strip():
        return pd.DataFrame(), "Enter a research problem first."
    live, errors = fetch_hf_datasets(query)
    combined = live + FALLBACK_DATASETS
    unique_records, seen = [], set()
    for record in combined:
        if record["id"] not in seen:
            seen.add(record["id"])
            unique_records.append(record)
    ranked = rank_datasets(query, unique_records).head(int(top_k))
    status = f"Live Hugging Face results: {len(live)}. Candidates ranked (including fallback examples): {len(unique_records)}."
    if errors:
        status += " Some API requests failed; results may be incomplete."
    return ranked, status

def format_results(results):
    if results.empty:
        return pd.DataFrame(), "No results found. Try a broader query.", None
    table = results[["name", "ranking_score", "semantic_similarity", "downloads", "likes", "lastModified", "url"]].copy()
    table.columns = ["Dataset", "Ranking score (%)", "Semantic similarity", "Downloads", "Likes", "Updated", "Source URL"]
    table["Ranking score (%)"] = (table["Ranking score (%)"] * 100).round(1)
    table["Semantic similarity"] = table["Semantic similarity"].round(3)
    details = []
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        tags = row.get("tags", [])
        if isinstance(tags, list):
            tags = ", ".join(map(str, tags[:20]))
        details.append(
            f"{rank}. {row['name']}\n"
            f"Ranking score: {row['ranking_score'] * 100:.1f}% (heuristic)\n"
            f"Description: {row['description']}\nTags: {tags}\n"
            f"Downloads: {row.get('downloads', 0)} | Likes: {row.get('likes', 0)}\n"
            f"Last modified: {row.get('lastModified') or 'not provided'}\n"
            f"Source: {row['url']}"
        )
    details_text = "\n\n".join(details)
    csv_path = "recommended_datasets.csv"
    results.to_csv(csv_path, index=False)
    return table, details_text, csv_path


def search_ui(query, top_k):
    ranked, status = discover(query, top_k)
    table, details, csv_path = format_results(ranked)
    return status, table, details, csv_path


with gr.Blocks(title="Research Dataset Discovery Assistant", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🔎 Research Dataset Discovery Assistant\nEnter your research idea and find relevant dataset repositories.")
    query_input = gr.Textbox(
        label="Research problem / idea",
        placeholder="Example: predict diabetes using patient health indicators",
        lines=3
    )
    top_k_input = gr.Slider(3, 20, value=10, step=1, label="Number of results")
    search_button = gr.Button("Find datasets", variant="primary")
    status_output = gr.Textbox(label="Search status", interactive=False)
    table_output = gr.Dataframe(label="Ranked datasets", interactive=False, wrap=True)
    details_output = gr.Textbox(label="Dataset details", lines=18, interactive=False)
    csv_output = gr.File(label="Download results CSV")
    search_button.click(
        fn=search_ui,
        inputs=[query_input, top_k_input],
        outputs=[status_output, table_output, details_output, csv_output]
    )
    gr.Examples(
        examples=[
            ["Predict diabetes using age, glucose, BMI, and blood pressure"],
            ["Detect fake news from article text and headlines"],
            ["Predict student academic performance from attendance and study habits"],
            ["Forecast air pollution using environmental sensor time series"],
            ["Predict customer churn from telecom usage data"]
        ],
        inputs=query_input
    )

# Share link is public to anyone who has it while the Colab runtime is active.
# Do not enter confidential research text into this public demo.
app.launch()
