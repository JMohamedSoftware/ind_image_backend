import requests
import os

BASE_URL = "http://localhost:8007" # Adjusted to match frontend and my started server
TEST_IMAGE = "https://images.unsplash.com/photo-1541443131876-44b03de101c5?w=500"

def test_indexing():
    algos = ["CNN", "ORB", "SIFT", "HOG", "TBIR_BASIC"]
    
    for algo in algos:
        print(f"\n🚀 Testing indexing for: {algo}")
        try:
            r = requests.post(f"{BASE_URL}/api/index", data={
                "url": TEST_IMAGE,
                "algorithm": algo,
                "filename": f"test_{algo}.jpg",
                "description": f"Testing {algo} indexing"
            }, timeout=10)
            
            if r.status_code == 200:
                print(f"✅ Success: {r.json().get('status')}")
                
                # Check if it appears in the list
                list_r = requests.get(f"{BASE_URL}/api/index/list?algorithm={algo}")
                items = list_r.json()
                if any(item['algorithm'] == algo for item in items):
                    print(f"✅ Found in {algo} pool list.")
                else:
                    print(f"❌ NOT found in {algo} pool list! (Pool contains {len(items)} items)")
            else:
                print(f"❌ Failed: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"❌ Request error: {e}")

if __name__ == "__main__":
    test_indexing()
