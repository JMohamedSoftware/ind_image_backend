import numpy as np
import hashlib

# logic from main.py
def get_algo_salt(algo):
    return hashlib.md5(algo.upper().encode()).hexdigest()[:4]

def get_algo_vector_length(algo):
    algo = algo.upper()
    if algo == "CNN": return 4096
    if algo == "CLIP": return 512
    if algo == "SVM": return 64
    if algo == "MEANSHIFT": return 16
    return 32

def get_semantic_features(text, algo, length=None):
    if length is None: length = get_algo_vector_length(algo)
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

def final_test():
    algo = "CNN"
    
    # 1. Indexing: "a red ferrari sports car"
    index_desc = "a red ferrari sports car"
    stored_vec = get_semantic_features(index_desc, algo)
    
    # 2. Search exact
    query_exact = "a red ferrari sports car"
    q_vec_exact = get_semantic_features(query_exact, algo)
    score_exact = np.dot(q_vec_exact, stored_vec) * 100
    print(f"Exact match ('{query_exact}'): {score_exact:.1f}%")
    
    # 3. Search partial: "ferrari"
    query_p1 = "ferrari"
    q_vec_p1 = get_semantic_features(query_p1, algo)
    score_p1 = np.dot(q_vec_p1, stored_vec) * 100
    print(f"Partial match ('{query_p1}'): {score_p1:.1f}%")
    
    # 4. Search partial: "red car"
    query_p2 = "red car"
    q_vec_p2 = get_semantic_features(query_p2, algo)
    score_p2 = np.dot(q_vec_p2, stored_vec) * 100
    print(f"Partial match ('{query_p2}'): {score_p2:.1f}%")

    # 5. Search unrelated: "ocean beach"
    query_u = "ocean beach"
    q_vec_u = get_semantic_features(query_u, algo)
    score_u = np.dot(q_vec_u, stored_vec) * 100
    print(f"Unrelated match ('{query_u}'): {score_u:.1f}%")

if __name__ == "__main__":
    final_test()
