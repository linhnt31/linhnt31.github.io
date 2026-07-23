---
title:
layout: default
permalink: /research/rag
published: true
---

> Here are my notes on Retrieval-Augmented Generation (RAG), covering core concepts from embeddings and semantic search to the full RAG pipeline.

# Table of Contents

1. [What Are Embeddings?](#embeddings)

    1.1. [Meaning as Geometry](#geometry)

    1.2. [Measuring Similarity](#similarity)

2. [Semantic Representation](#semantic)

    2.1. [Why Semantic Search Beats Keyword Search](#semantic-search)

    2.2. [Key Properties of Good Embeddings](#properties)

3. [The RAG Pipeline](#pipeline)

    3.1. [Indexing](#indexing)

    3.2. [Retrieval](#retrieval)

    3.3. [Generation](#generation)

4. [Embedding Models](#models)

5. [References](#ref)

---

## 1. What Are Embeddings? [1]

<div id='embeddings'/>

Embeddings are **dense numerical vectors** that represent text — words, sentences, or entire documents — in a high-dimensional space. Rather than treating words as arbitrary symbols, embeddings encode *meaning*: semantically similar texts end up geometrically close to each other in that space.

For example, the sentence `"The dog chased the ball"` might be represented as a vector like:

$$\mathbf{v} = [0.23, -0.81, 0.54, \ldots] \in \mathbb{R}^d$$

where $d$ is typically 384, 768, or 1536 dimensions depending on the model.

The critical property is that **distance in vector space corresponds to semantic distance in meaning**. Two texts that mean the same thing (even if worded differently) will have vectors that are close together, while unrelated texts will be far apart.

### 1.1. Meaning as Geometry

<div id='geometry'/>

The core insight behind embeddings is that **semantic similarity = geometric proximity**. In the vector space:

- `"car"` and `"automobile"` are near neighbors
- `"king"`, `"queen"`, `"man"`, and `"woman"` satisfy: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$
- Unrelated concepts like `"pizza"` and `"quantum physics"` are geometrically distant

This transforms a fuzzy linguistic question — *"are these texts similar?"* — into a precise mathematical one: *"how close are these vectors?"*

### 1.2. Measuring Similarity

<div id='similarity'/>

The standard metric is **cosine similarity**, which measures the *angle* between two vectors rather than their absolute distance:

$$\text{similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{A}| \times |\mathbf{B}|}$$

| Score | Interpretation |
|-------|---------------|
| `1.0` | Identical meaning |
| `0.5 – 0.9` | Highly related |
| `0.0` | Completely unrelated |
| `-1.0` | Opposite meaning |

Cosine similarity is preferred over Euclidean distance because it is **scale-invariant** — it doesn't matter how long the vectors are, only the direction they point. This makes it robust to variable-length texts.

---

## 2. Semantic Representation

<div id='semantic'/>

### 2.1. Why Semantic Search Beats Keyword Search

<div id='semantic-search'/>

Traditional information retrieval methods like **BM25** or **TF-IDF** rely on exact keyword matching. They fail when the query and document use different words to express the same idea.

| Query | Relevant Document | BM25 match? | Embedding match? |
|-------|-----------------|-------------|-----------------|
| "how to fix a flat tyre" | "puncture repair guide" | No | Yes |
| "cardiac arrest symptoms" | "heart attack warning signs" | No | Yes |
| "affordable accommodation" | "cheap hotels" | No | Yes |

Embeddings enable **semantic search**: finding relevant content based on *meaning*, not surface-level word overlap. This is the foundation that makes RAG possible.

### 2.2. Key Properties of Good Embeddings

<div id='properties'/>

| Property | Why It Matters for RAG |
|----------|----------------------|
| **Semantic coherence** | Similar-meaning chunks cluster together, improving retrieval precision |
| **Isotropy** | Vectors spread uniformly in space; avoids degenerate clustering |
| **Cross-lingual alignment** | Multilingual models match concepts across languages |
| **Domain awareness** | Domain-specific models (legal, medical, code) outperform general-purpose ones on niche corpora |
| **Fixed-size output** | Enables fast vector arithmetic and efficient ANN indexing |

---

## 3. The RAG Pipeline

<div id='pipeline'/>

Retrieval-Augmented Generation combines a **retriever** (embedding-based search) with a **generator** (LLM) to answer questions grounded in a specific knowledge base. The pipeline has three stages:

```
Documents → Chunking → Embedding → Vector DB
                                       ↑
Query → Embedding → kNN Search ────────┘
                        ↓
              Top-k chunks → LLM prompt → Answer
```

### 3.1. Indexing

<div id='indexing'/>

Before any query can be answered, the knowledge base must be indexed:

1. **Chunking** — Split documents into passages (e.g., 256–512 tokens). Chunk size is a key hyperparameter: too small loses context, too large dilutes relevance.
2. **Embedding** — Each chunk is passed through an embedding model to produce a vector.
3. **Storage** — Vectors are stored in a **vector database** (FAISS, Pinecone, Weaviate, ChromaDB, Qdrant) that supports fast approximate nearest-neighbor (ANN) search.

### 3.2. Retrieval

<div id='retrieval'/>

At query time:

1. The user's question is embedded using the *same model* used during indexing.
2. A **k-nearest-neighbor search** finds the top-$k$ most similar document vectors (typically $k = 3$–$10$).
3. The corresponding text chunks are retrieved.

The quality of retrieval directly determines the quality of the final answer — garbage in, garbage out.

### 3.3. Generation

<div id='generation'/>

The retrieved chunks are injected into the LLM's prompt as context:

```
System: You are a helpful assistant. Answer using only the provided context.

Context:
[chunk 1] ...
[chunk 2] ...
[chunk 3] ...

User: {question}
```

The LLM then generates an answer grounded in the retrieved evidence, dramatically reducing hallucinations compared to vanilla prompting.

---

## 4. Embedding Models

<div id='models'/>

| Model | Provider | Dimensions | Notes |
|-------|----------|-----------|-------|
| `text-embedding-3-small` | OpenAI | 1536 | Fast, strong general baseline |
| `text-embedding-3-large` | OpenAI | 3072 | Higher accuracy, higher cost |
| `all-MiniLM-L6-v2` | sentence-transformers | 384 | Lightweight, open-source |
| `bge-m3` | BAAI | 1024 | Multilingual, strong RAG performance |
| `nomic-embed-text` | Nomic AI | 768 | Open-source, competitive quality |

The embedding model used during indexing and retrieval **must be the same** — mixing models produces incompatible vector spaces and breaks retrieval entirely.

---

## 5. References

<div id='ref'/>

[1] Hesamation, *LLM Embeddings Explained: A Visual and Intuitive Guide*, Hugging Face Spaces. [link](https://huggingface.co/spaces/hesamation/primer-llm-embedding)

[2] Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS. [link](https://arxiv.org/abs/2005.11401)

[3] Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP. [link](https://arxiv.org/abs/1908.10084)
