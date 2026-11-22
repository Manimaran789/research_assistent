---
title: Research Dataset Discovery Assistant
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Research Dataset Discovery Assistant

Search for candidate research datasets based on a natural-language research idea. The application retrieves dataset metadata from Hugging Face, then ranks candidates using Sentence Transformer embeddings and cosine similarity.

**Important:** live retrieval is keyword-based; semantic similarity reranks the retrieved candidates. Scores are heuristic and are not probabilities, model accuracy, or proof of dataset quality. The app analyzes metadata only, not the underlying dataset files. Fallback entries are examples and should be verified.
