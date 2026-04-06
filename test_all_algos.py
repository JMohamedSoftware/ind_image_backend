import requests
import time
import sys

BASE_URL = "http://localhost:8007"
TEST_IMAGE = "https://images.unsplash.com/photo-1541443131876-44b03de101c5?w=500"

def test():
    print("--- 🧪 COMPREHENSIVE ALGORITHM TESTING 🧪 ---")
    
    # 1. RESET
    print("\n[1/5] Resetting index...")
    r = requests.delete(f"{BASE_URL}/api/index")
    if r.status_code == 200:
        print("✅ Index cleared.")
    else:
        print(f"❌ Failed to clear index: {r.text}")
        sys.exit(1)
    time.sleep(1) # Wait for stability

    # 2. CBIR TEST (ORB)
    print("\n[2/5] Testing CBIR (ORB)...")
    requests.post(f"{BASE_URL}/api/index", data={"url": TEST_IMAGE, "algorithm": "ORB", "description": "CBIR Test Image"})
    time.sleep(1)
    
    algos_cbir = ["ORB", "SIFT", "AKAZE", "HOG"]
    for algo in algos_cbir:
        r = requests.post(f"{BASE_URL}/api/search", data={"url": TEST_IMAGE, "algorithm": algo})
        if r.status_code == 200:
            res = r.json()
            if res['results']:
                score = res['results'][0]['similarityScore']
                print(f"  ✅ {algo:8} : {score}% similarity")
            else:
                # For isolated collections, only ORB should return results if only indexed with ORB
                print(f"  ℹ️ {algo:8} : No results (Expected if isolated)")
        else:
            print(f"  ❌ {algo:8} : Error {r.status_code}")

    # 3. DL TEST (CNN)
    print("\n[3/5] Testing DL (CNN)...")
    # First, test visual identity match (No description)
    print("  Indexing CNN without description...")
    requests.post(f"{BASE_URL}/api/index", data={"url": TEST_IMAGE, "algorithm": "CNN"})
    time.sleep(1)
    
    r = requests.post(f"{BASE_URL}/api/search", data={"url": TEST_IMAGE, "algorithm": "CNN"})
    if r.status_code == 200:
        res = r.json()
        if 'results' in res and res['results']:
            print(f"  ✅ CNN Visual Match: {res['results'][0]['similarityScore']}%")
        else:
            print(f"  ❌ CNN visual search failed: {res.get('message', res.get('error', 'No results'))}")

    # Now, test semantic search (Needs description)
    print("  Indexing CNN with description...")
    requests.post(f"{BASE_URL}/api/index", data={"url": TEST_IMAGE, "algorithm": "CNN", "description": "Deep Red Flower"})
    time.sleep(1)
    
    # Use CLIP for text-based semantic search across CNN indexed data
    r = requests.post(f"{BASE_URL}/api/search", data={"query_text": "red flower", "algorithm": "CLIP"})
    if r.status_code == 200:
        res = r.json()
        if 'results' in res and res['results']:
            # Find our CNN item
            cnn_item = next((i for i in res['results'] if i['algorithm'] == 'CNN'), None)
            if cnn_item:
                print(f"  ✅ CNN Semantic Match via CLIP: {cnn_item['similarityScore']}% for 'red flower'")
            else:
                print("  ❌ CNN semantic match failed (not in results).")
        else:
            print(f"  ❌ CLIP semantic search failed: {res.get('message', res.get('error', 'No results'))}")

    # 4. TBIR TEST
    print("\n[4/5] Testing TBIR...")
    requests.post(f"{BASE_URL}/api/index", data={"url": TEST_IMAGE, "algorithm": "TBIR_BASIC", "description": "vibrant, nature, blossom"})
    
    r = requests.post(f"{BASE_URL}/api/search/text", data={"query": "nature", "algorithm": "TBIR_BASIC"})
    if r.status_code == 200:
        res = r.json()
        if res['results']:
            print(f"  ✅ TBIR Match: {res['results'][0]['similarityScore']}% for 'nature'")
        else:
            print("  ❌ TBIR failed to find 'nature'.")

    # 5. ISOLATION VERIFICATION
    print("\n[5/5] Verifying Isolation...")
    # Search for "nature" using CNN (which shouldn't find the TBIR image if isolated correctly)
    r = requests.post(f"{BASE_URL}/api/search/text", data={"query": "nature", "algorithm": "CNN"})
    if r.status_code == 200:
        res = r.json()
        # Filter for the image that was only indexed with TBIR
        tbir_matches = [res for res in res['results'] if res['algorithm'] == "TBIR_BASIC"]
        if not tbir_matches:
            print("  ✅ Isolation confirmed: CNN search did not return TBIR-only results.")
        else:
            print("  ⚠️ Warning: CNN search returned TBIR-only indexed items. Isolation might be loose.")

    # FINAL CLEAR
    print("\n🧹 Cleaning up test data...")
    requests.delete(f"{BASE_URL}/api/index")
    print("✅ Index cleaned. Done.")

if __name__ == "__main__":
    test()
