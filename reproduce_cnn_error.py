import requests
import os

def test_cnn_flow():
    BASE_URL = "http://localhost:8007/api"
    
    # Check health
    try:
        requests.get("http://localhost:8007/", timeout=2)
    except:
        print("Server down on 8007")
        return

    # 1. Index an image with CNN
    img_url = "https://images.unsplash.com/photo-1592194996308-7b43878e84a6"
    print("Testing indexing with CNN...")
    payload = {
        "url": img_url,
        "filename": "cnn_test_cat.jpg",
        "algorithm": "CNN",
        "description": "a fluffy cat for cnn testing"
    }
    try:
        r_index = requests.post(f"{BASE_URL}/index", data=payload, timeout=20)
        print(f"Indexing Response: {r_index.status_code}")
        print(r_index.json())
    except Exception as e:
        print(f"Indexing FAILED: {str(e)}")

    # 2. Search with CNN
    print("\nTesting search with CNN...")
    search_payload = {
        "url": img_url,
        "algorithm": "CNN"
    }
    try:
        r_search = requests.post(f"{BASE_URL}/search", data=search_payload, timeout=20)
        print(f"Search Response: {r_search.status_code}")
        # print(r_search.json())
    except Exception as e:
        print(f"Search FAILED: {str(e)}")

if __name__ == "__main__":
    test_cnn_flow()
