import requests
import os

def index_test_image():
    URL = "http://localhost:8007/api/index"
    
    # Check if server is up
    try:
        requests.get("http://localhost:8007/", timeout=2)
    except:
        print("Server not running on 8007. Please start the backend.")
        return

    # Use a public image URL
    img_url = "https://images.unsplash.com/photo-1592194996308-7b43878e84a6" # A cat image
    filename = "verification_cat.jpg"
    description = "a fluffy white cat sitting on a wooden floor"
    
    payload = {
        "url": img_url,
        "filename": filename,
        "algorithm": "CNN",
        "description": description
    }
    
    print(f"Indexing '{filename}' with description: '{description}'...")
    r = requests.post(URL, data=payload)
    print(r.json())

if __name__ == "__main__":
    index_test_image()
