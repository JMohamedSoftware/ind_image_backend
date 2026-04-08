import requests
import time

BASE = "http://127.0.0.1:8010"
IMG  = "https://images.unsplash.com/photo-1541443131876-44b03de101c5?w=500"

CBIR_ALGOS = ["ORB", "SIFT", "AKAZE", "HOG"]
ML_ALGOS   = ["SVM", "K-MEANS", "PCA", "MEANSHIFT"]
DL_ALGOS   = ["CNN", "CLIP"]

def OK(ok): return "OK" if ok else "FAIL"

def reset():
    r = requests.delete(f"{BASE}/api/index")
    return r.status_code == 200

def index(algo, desc="Test image"):
    r = requests.post(f"{BASE}/api/index", data={"url": IMG, "algorithm": algo, "description": desc})
    return r.status_code == 200

def search(algo):
    r = requests.post(f"{BASE}/api/search", data={"url": IMG, "algorithm": algo})
    if r.status_code == 200:
        d = r.json()
        results = d.get("results", [])
        if results:
            return True, results[0]["similarityScore"], None
        return False, 0, d.get("message", "No results")
    return False, 0, f"HTTP {r.status_code}"

def search_text(algo, query="test"):
    r = requests.post(f"{BASE}/api/search/text", data={"query": query, "algorithm": algo})
    if r.status_code == 200:
        d = r.json()
        results = d.get("results", [])
        if results:
            return True, results[0]["similarityScore"], None
        return False, 0, d.get("message", "No results")
    return False, 0, f"HTTP {r.status_code}"

print("\n" + "="*55)
print("   FULL ALGORITHM TEST - ALL CATEGORIES")
print("="*55)

print("\n[*] Resetting index...")
ok = reset()
print(f"  {OK(ok)} Index cleared")
time.sleep(1)

# CBIR
print("\n[CBIR] Classical Visual Algorithms")
print("-"*40)
for algo in CBIR_ALGOS:
    idx_ok = index(algo)
    time.sleep(0.5)
    found, score, err = search(algo)
    status = f"{score:.1f}% similarity" if found else f"NOT found -- {err}"
    print(f"  {OK(found)} {algo:<8}  index={OK(idx_ok)}  search: {status}")
time.sleep(0.5)

# ML
print("\n[ML] Machine Learning Algorithms")
print("-"*40)
for algo in ML_ALGOS:
    idx_ok = index(algo, desc="machine learning test")
    time.sleep(0.5)
    found, score, err = search(algo)
    status = f"{score:.1f}% similarity" if found else f"NOT found -- {err}"
    print(f"  {OK(found)} {algo:<10}  index={OK(idx_ok)}  search: {status}")
time.sleep(0.5)

# DL
print("\n[DL] Deep Learning Algorithms")
print("-"*40)
for algo in DL_ALGOS:
    idx_ok = index(algo, desc="red flower nature")
    time.sleep(0.5)
    found, score, err = search(algo)
    status = f"{score:.1f}% similarity" if found else f"NOT found -- {err}"
    print(f"  {OK(found)} {algo:<8}  index={OK(idx_ok)}  search: {status}")

found, score, err = search_text("CLIP", "flower")
status = f"{score:.1f}% similarity" if found else f"NOT found -- {err}"
print(f"  {OK(found)} CLIP text ('flower')  : {status}")

# TBIR
print("\n[TBIR] Text-Based Retrieval")
print("-"*40)
idx_ok = index("TBIR_BASIC", desc="nature flower blossom vibrant")
time.sleep(0.5)
found, score, err = search_text("TBIR_BASIC", "flower")
status = f"{score:.1f}% similarity" if found else f"NOT found -- {err}"
print(f"  {OK(found)} TBIR_BASIC  index={OK(idx_ok)}  search: {status}")

print("\n[*] Cleanup...")
reset()
print("  OK Done.\n")
print("="*55 + "\n")
