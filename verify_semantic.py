import pickle
import numpy as np
import hashlib
import os

# Mimic the new get_semantic_features from main.py
def get_algo_salt(algo):
    return hashlib.md5(algo.upper().encode()).hexdigest()[:4]

def get_semantic_features(text, algo, length=128):
    if text is None: text = "dummy"
    algo_salt = get_algo_salt(algo)
    tokens = [t.strip().lower() for t in text.replace(',', ' ').replace('.', ' ').split() if len(t.strip()) > 1]
    if not tokens: tokens = ["dummy"]
    vectors = []
    for token in tokens:
        seed = f"{token}_{algo_salt}"
        np.random.seed(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16) % 2**32)
        v = np.random.randn(length).astype(np.float32)
        vectors.append(v)
    avg_vector = np.mean(vectors, axis=0)
    avg_vector /= (np.linalg.norm(avg_vector) + 1e-8)
    return avg_vector

def verify():
    INDEX_FILE = "index.pkl"
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)

    # Find an item with a multi-word description
    target_item = None
    for item in data:
        desc = item.get("description")
        if desc and isinstance(desc, str) and len(desc.split()) > 2:
            target_item = item
            break
    
    if not target_item:
        print("No multi-word description found in index. Skipping detailed test.")
        # Test with whatever we have
        target_item = data[0]
        desc = target_item.get("description") or "dummy test case"
    else:
        desc = target_item.get("description")

    print(f"Testing with image: {target_item['filename']}")
    print(f"Description: '{desc}'")
    
    algo = "CNN"
    length = 4096
    stored_vec = target_item["descriptors_map"][algo]
    
    # Test 1: Exact match
    query_exact = desc
    vec_exact = get_semantic_features(query_exact, algo, length=length)
    score_exact = np.dot(vec_exact, stored_vec) * 100
    print(f"  Exact match score: {score_exact:.1f}% (Expected: ~100%)")
    
    # Test 2: Partial match (first word)
    query_partial = desc.split()[0]
    vec_partial = get_semantic_features(query_partial, algo, length=length)
    score_partial = np.dot(vec_partial, stored_vec) * 100
    print(f"  Partial match ('{query_partial}') score: {score_partial:.1f}% (Expected: >30%)")
    
    # Test 3: Completely different
    query_diff = "something entirely unrelated"
    vec_diff = get_semantic_features(query_diff, algo, length=length)
    score_diff = np.dot(vec_diff, stored_vec) * 100
    print(f"  Unrelated match score: {score_diff:.1f}% (Expected: <10%)")

if __name__ == "__main__":
    verify()
