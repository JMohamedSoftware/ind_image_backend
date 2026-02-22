import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io
import requests
from PIL import Image
import os

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import pickle

# File-based storage for persistence
INDEX_FILE = "index.pkl"

def save_index(data):
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(data, f)

def load_index():
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading index: {e}")
    return []

# Load data on startup
indexed_data = load_index()

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def process_image_content(contents):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

@app.get("/")
async def root():
    return {"message": "Visual Search API is running with Cloudinary support"}

@app.post("/api/index")
async def index_image(
    file: UploadFile = File(None), 
    url: str = Form(None),
    filename: str = Form(None)
):
    contents = None
    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            contents = response.content
            target_filename = filename or url.split('/')[-1]
        except Exception as e:
            return {"error": f"Failed to fetch image from URL: {str(e)}"}
    elif file:
        contents = await file.read()
        target_filename = file.filename
    else:
        return {"error": "No image provided (neither file nor url)"}

    result = process_image_content(contents)
    if result is None:
        return {"status": "error", "message": "Could not decode image"}
    kp, des = result
    
    if des is not None:
        indexed_data.append({
            "filename": target_filename,
            "url": url if url else None,
            "descriptors": des,
        })
        save_index(indexed_data)
        return {
            "status": "success", 
            "filename": target_filename, 
            "keypoints": len(kp)
        }
    
    return {"status": "error", "message": "No features found"}

@app.post("/api/search")
async def search_image(file: UploadFile = File(None), url: str = Form(None)):
    contents = None
    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            contents = response.content
        except Exception as e:
            return {"error": f"Failed to fetch image from URL: {str(e)}"}
    elif file:
        contents = await file.read()
    else:
        return {"error": "No image provided"}

    result = process_image_content(contents)
    if result is None:
        return {"results": [], "message": "Could not decode image"}
    kp_query, des_query = result
    
    if des_query is None:
        return {"results": [], "message": "No features found in query image"}
    
    if not indexed_data:
        return {"results": [], "message": "No images in index yet. Please upload and index images first."}
    
    results = []
    for item in indexed_data:
        des_indexed = item["descriptors"]
        
        matches = bf.match(des_query, des_indexed)
        matches = sorted(matches, key=lambda x: x.distance)
        
        good_matches = [m for m in matches if m.distance < 50]
        score = (len(good_matches) / len(kp_query)) * 100 if len(kp_query) > 0 else 0
        score = min(score * 2.5, 100) 
        
        results.append({
            "filename": item["filename"],
            "url": item.get("url"),
            "similarityScore": round(score, 2),
            "matchesCount": len(good_matches),
        })
    
    results = sorted(results, key=lambda x: x["similarityScore"], reverse=True)
    return {"results": results[:5]}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
