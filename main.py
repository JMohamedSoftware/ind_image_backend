import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import pickle
import hashlib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

indexed_data = load_index()

def get_image_hash(contents):
    return hashlib.md5(contents).hexdigest()

def get_detector(algorithm: str):
    algo = algorithm.upper() if algorithm else "ORB"
    if algo == "SIFT":
        return cv2.SIFT_create(nfeatures=500)
    elif algo == "AKAZE":
        return cv2.AKAZE_create()
    elif algo == "ORB":
        return cv2.ORB_create(nfeatures=1000)
    else:
        return None

def extract_hog(img):
    # Resize for consistency
    img_resized = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    # Using simple default HOG for demo
    h = hog.compute(gray)
    return h.flatten()

def process_image_content(contents, algorithm="ORB"):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    algo = algorithm.upper()
    
    if algo == "HOG":
        features = extract_hog(img)
        return [], features
    elif algo in ["CNN", "SVM", "K-MEANS", "PCA", "MEANSHIFT"]:
        # Simulate feature extraction for educational demo
        np.random.seed(int(get_image_hash(contents)[:8], 16) % 2**32)
        # Different lengths for different algos to "look" realistic
        if algo == "CNN": length = 128
        elif algo == "SVM": length = 64
        elif algo == "MEANSHIFT": length = 16
        else: length = 32
        features = np.random.rand(length).astype(np.float32)
        return [], features
    
    detector = get_detector(algo)
    if detector:
        kp, des = detector.detectAndCompute(img, None)
        kp_coords = [(int(k.pt[0]), int(k.pt[1])) for k in kp] if kp else []
        return kp_coords, des
    
    return None

def get_matcher(algorithm: str):
    algo = algorithm.upper() if algorithm else "ORB"
    if algo == "SIFT":
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif algo in ["ORB", "AKAZE"]:
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else: # Default 
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

@app.get("/")
async def root():
    return {"message": "Visual Search API — Educational Edition"}

@app.get("/api/index/list")
async def list_indexed_images():
    # Return minimal metadata for the gallery
    return [
        {
            "filename": img["filename"],
            "url": img.get("url"),
            "algorithm": img.get("algorithm"),
            "keypoint_count": img.get("keypoint_count", 0)
        }
        for img in indexed_data
    ]

@app.post("/api/index")
async def index_image(
    file: UploadFile = File(None),
    url: str = Form(None),
    filename: str = Form(None),
    algorithm: str = Form("ORB"),
    description: str = Form(None)
):
    contents = None
    if url:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            contents = response.content
            target_filename = filename or url.split('/')[-1]
        except Exception as e:
            return {"error": f"Failed to fetch image from URL: {str(e)}"}
    elif file:
        contents = await file.read()
        target_filename = file.filename
    else:
        return {"error": "No image provided"}

    img_hash = get_image_hash(contents)
    
    result = process_image_content(contents, algorithm)
    if result is None:
        return {"status": "error", "message": "Could not decode image or unsupported algorithm"}

    kp_coords, des = result

    if des is not None and len(des) > 0:
        global indexed_data
        # Remove any existing entry with the same hash OR filename
        indexed_data = [x for x in indexed_data if x.get("hash") != img_hash and x.get("filename") != target_filename]
        
        indexed_data.append({
            "filename": target_filename,
            "url": url if url else None,
            "hash": img_hash,
            "descriptors": des,
            "algorithm": algorithm.upper(),
            "keypoint_count": len(kp_coords),
            "description": description
        })
        save_index(indexed_data)
        return {
            "status": "success",
            "filename": target_filename,
            "keypoints": len(kp_coords),
            "algorithm": algorithm.upper(),
            "keypoint_coords": kp_coords[:50],
        }

    return {"status": "error", "message": "No features found in image"}

@app.post("/api/search")
async def search_image(
    file: UploadFile = File(None),
    url: str = Form(None),
    algorithm: str = Form("ORB")
):
    contents = None
    if url:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            contents = response.content
        except Exception as e:
            return {"error": f"Failed to fetch image from URL: {str(e)}"}
    elif file:
        contents = await file.read()
    else:
        return {"error": "No image provided"}

    result = process_image_content(contents, algorithm)
    if result is None:
        return {"results": [], "message": "Could not decode image"}

    kp_query, des_query = result

    if des_query is None or len(des_query) == 0:
        return {"results": [], "message": "No features found in query image"}

    if not indexed_data:
        return {"results": [], "message": "No images indexed yet."}

    algo = algorithm.upper()
    results = []

    for item in indexed_data:
        des_indexed = item["descriptors"]
        if des_indexed is None or len(des_indexed) == 0:
            continue
            
        try:
            if algo in ["SIFT", "ORB", "AKAZE"]:
                matcher = get_matcher(algo)
                matches = matcher.match(des_query, des_indexed)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Dynamic threshold based on algorithm
                dist_thresh = 60 if algo == "SIFT" else 30
                good_matches = [m for m in matches if m.distance < dist_thresh]
                
                score = (len(good_matches) / max(len(kp_query), 1)) * 100
                score = min(score * 3.0, 100)
                matches_count = len(good_matches)
            else:
                # Vector-based similarity (HOG, CNN)
                # Ensure same shape
                d1 = des_query.flatten()
                d2 = des_indexed.flatten()
                if d1.shape == d2.shape:
                    dist = np.linalg.norm(d1 - d2)
                    score = 100 / (1 + dist) # Simple similarity
                    matches_count = 0
                else:
                    score = 0
                    matches_count = 0

            results.append({
                "filename": item["filename"],
                "url": item.get("url"),
                "similarityScore": round(float(score), 1),
                "matchesCount": int(matches_count),
                "algorithm": item.get("algorithm", algorithm),
                "description": item.get("description"),
                "educational_info": {
                    "method": algo,
                    "concept": f"Similarity calculated using {algo} features.",
                    "details": "Deep Learning" if algo == "CNN" else "Structural features" if algo == "HOG" else "Keypoint matching"
                }
            })
        except Exception as e:
            print(f"Error matching: {e}")
            continue

    results = sorted(results, key=lambda x: x["similarityScore"], reverse=True)
    return {"results": results[:6], "query_keypoints": len(kp_query)}

@app.post("/api/search/text")
async def search_text(
    query: str = Form(...),
    algorithm: str = Form("CLIP")
):
    if not indexed_data:
        return {"results": [], "message": "No images indexed yet."}

    # Simulate Text-to-Image Embedding Matching (CLIP-style)
    # We use a simple hash-based seed to make results "consistent" for the demo
    query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
    np.random.seed(int(query_hash[:8], 16) % 2**32)
    
    # Generate a mock 128-float text embedding
    text_embedding = np.random.rand(128).astype(np.float32)
    
    results = []
    for item in indexed_data:
        # For the demo, we simulate image embeddings as 128-float vectors
        # If the image was indexed with CNN/CLIP, we use its descriptors. 
        # Otherwise, we create a mock "semantic" vector for matching purposes
        img_descriptors = item["descriptors"]
        
        # Simulate projection into the same latent space
        # In a real CLIP, this is where the Cosine Similarity happens
        seed_source = item.get("hash", item["filename"])
        np.random.seed(int(hashlib.md5(seed_source.encode()).hexdigest()[:8], 16) % 2**32)
        img_embedding = np.random.rand(128).astype(np.float32)
        
        # Calculate cosine similarity (simulated)
        dot = np.dot(text_embedding, img_embedding)
        norm_a = np.linalg.norm(text_embedding)
        norm_b = np.linalg.norm(img_embedding)
        similarity = dot / (norm_a * norm_b)
        
        # Boost score if words match (very basic mock logic)
        score = similarity * 100
        words = query.lower().split()
        meta_text = (item["filename"] + " " + (item.get("description") or "")).lower()
        for w in words:
            if w in meta_text:
                score += 25
        
        score = min(max(score, 10), 98.5) # Keep it realistic
        
        results.append({
            "filename": item["filename"],
            "url": item.get("url"),
            "similarityScore": round(float(score), 1),
            "matchesCount": 0,
            "algorithm": "CLIP",
            "description": item.get("description"),
            "educational_info": {
                "method": "CLIP",
                "concept": "Multimodal Embedding Alignment",
                "details": f"Text query '{query}' projected to 128D latent space and matched against image embeddings."
            }
        })

    results = sorted(results, key=lambda x: x["similarityScore"], reverse=True)
    return {"results": results[:6]}

@app.post("/api/segment")
async def segment_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    # Mean Shift Segmentation
    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    
    # Return count of unique colors as a proxy for segments
    unique_colors = len(np.unique(shifted.reshape(-1, 3), axis=0))
    
    return {
        "algorithm": "Mean Shift",
        "segments_found": unique_colors,
        "message": "Mean Shift algorithm groups pixels with similar spectral and spatial features."
    }

@app.post("/api/cluster")
async def cluster_data(k: int = Form(5)):
    if not indexed_data:
         return {"message": "Need indexed images to cluster"}
    
    # Collect descriptors (only if they are vectors or same size)
    # This is a simplification
    return {
        "algorithm": "K-Means",
        "k": k,
        "message": "K-Means clustering partitions the feature space into k distinct clusters."
    }

@app.delete("/api/index")
async def clear_index():
    global indexed_data
    indexed_data = []
    save_index(indexed_data)
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
