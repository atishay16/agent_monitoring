
# Agent Loop Detector

This project implements a local prototype of the multi-level hashing + semantic similarity detector for agentic AI loops.

It instruments an (example) agent loop, logs events, hashes actions and responses, computes embeddings with SentenceTransformers, and uses FAISS for similarity search.

## Setup

1. Create a new PyCharm project from this folder.
2. In PyCharm, configure a Python virtual environment for the project.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the example:

```bash
python main.py
```

You can then replace the dummy `run_example_loop` implementation with your real agent logic.
