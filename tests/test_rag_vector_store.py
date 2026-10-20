import pickle

import faiss

from rag.retriever.retriever import VECTOR_STORE_DIR


def test_tracked_vector_store_assets_match():
    index = faiss.read_index(str(VECTOR_STORE_DIR / "index.faiss"))
    with (VECTOR_STORE_DIR / "metadata.pkl").open("rb") as file:
        payload = pickle.load(file)

    assert index.d == 384
    assert index.ntotal == len(payload["chunks"]) == len(payload["metadata"])
