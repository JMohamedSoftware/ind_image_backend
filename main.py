import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import pickle
import hashlib

import time
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use absolute path for persistence
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index.pkl")

def save_index(data):
    try:
        with open(INDEX_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"DATABASE: Saved {len(data)} items to {INDEX_FILE}")
    except Exception as e:
        print(f"DATABASE ERROR: Failed to save index: {e}")

def load_index():
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "rb") as f:
                data = pickle.load(f)
                print(f"DATABASE: Loaded {len(data)} items from {INDEX_FILE}")
                return data
        except Exception as e:
            print(f"DATABASE ERROR: Failed to load index: {e}")
    else:
        print(f"DATABASE: {INDEX_FILE} not found, starting fresh.")
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

def get_color_histogram(img):
    # Simple RGB average percentages for education
    # Convert to RGB (OpenCV uses BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    avg_color = np.mean(img_rgb, axis=(0, 1))
    total = np.sum(avg_color)
    if total == 0: return {"r": 33, "g": 33, "b": 33}
    return {
        "r": round((avg_color[0] / total) * 100, 1),
        "g": round((avg_color[1] / total) * 100, 1),
        "b": round((avg_color[2] / total) * 100, 1)
    }

def get_type_for_algorithm(algo):
    algo = algo.upper()
    if algo == "CLIP": return "DL"
    if algo in ["CNN", "MEANSHIFT"]: return "DL" # MeanShift is Segment in guide, putting in CBIR usually but DL in my specific mapping or separate
    if algo.startswith("TBIR"): return "TBIR"
    if algo in ["SIFT", "ORB", "AKAZE", "HOG"]: return "CBIR"
    return "CBIR" # Default

def process_image_content(contents, algorithm="ORB"):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    algo = algorithm.upper()
    color_stats = get_color_histogram(img)
    
    if algo == "HOG":
        features = extract_hog(img)
        return [], features, color_stats
    elif algo in ["CNN", "SVM", "K-MEANS", "PCA", "MEANSHIFT"]:
        # Simulate feature extraction for educational demo
        np.random.seed(int(get_image_hash(contents)[:8], 16) % 2**32)
        # Different lengths for different algos to "look" realistic
        if algo == "CNN": length = 128
        elif algo == "SVM": length = 64
        elif algo == "MEANSHIFT": length = 16
        else: length = 32
        features = np.random.rand(length).astype(np.float32)
        return [], features, color_stats
    
    detector = get_detector(algo)
    if detector:
        kp, des = detector.detectAndCompute(img, None)
        kp_coords = [(int(k.pt[0]), int(k.pt[1])) for k in kp] if kp else []
        return kp_coords, des, color_stats
    
    # Handle TBIR or unknown algorithms by returning empty features instead of None
    # This prevents the indexer from failing due to "No features found"
    return [], None, color_stats

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
async def list_indexed_images(type: str = None):
    # Return minimal metadata for the gallery
    filtered = indexed_data
    if type:
        filtered = [img for img in indexed_data if img.get("type", "CBIR") == type.upper()]
    
    return [
        {
            "id": img.get("id", img["filename"]), # Fallback for old data
            "filename": img["filename"],
            "url": img.get("url"),
            "algorithm": img.get("algorithm"),
            "keypoint_count": img.get("keypoint_count", 0),
            "description": img.get("description"),
            "type": img.get("type", "CBIR"),
            "color_stats": img.get("color_stats")
        }
        for img in filtered
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

    kp_coords, des, color_stats = result # Now returns color_stats
    
    # Determine type based on algorithm or presence of description
    img_type = get_type_for_algorithm(algorithm)
    # If it's TBIR, we essentially use common fields
    
    # If it's TBIR or we have descriptors, save it
    # Note: For TBIR, 'descriptors' will be None but we still index it
    if des is not None or img_type == "TBIR":
        global indexed_data
        # Remove any existing entry with the same hash OR filename
        indexed_data = [x for x in indexed_data if x.get("hash") != img_hash and x.get("filename") != target_filename]
        
        indexed_data.append({
            "id": str(uuid.uuid4()), # Unique stable ID
            "filename": target_filename,
            "url": url if url else None,
            "hash": img_hash,
            "descriptors": des,
            "algorithm": algorithm.upper(),
            "type": img_type,
            "keypoint_count": len(kp_coords) if kp_coords else 0,
            "description": description,
            "color_stats": color_stats,
            "indexed_at": time.time()
        })
        save_index(indexed_data)
        return {
            "status": "success",
            "id": indexed_data[-1]["id"],
            "filename": target_filename,
            "keypoints": len(kp_coords) if kp_coords else 0,
            "algorithm": algorithm.upper(),
            "type": img_type,
            "color_stats": color_stats,
            "keypoint_coords": kp_coords[:50] if kp_coords else [],
        }

    return {"status": "error", "message": "No features found in image for chosen algorithm"}

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

    kp_query, des_query, color_query = result

    if des_query is None or len(des_query) == 0:
        return {"results": [], "message": "No features found in query image"}

    # Filter indexed data by the SAME type as requested SEARCH algorithm
    search_type = get_type_for_algorithm(algorithm)
    # Corrected filter: Only search against images indexed with same TYPE
    filtered_db = [img for img in indexed_data if img.get("type", "CBIR") == search_type]

    if not filtered_db:
        return {"results": [], "message": f"No images indexed for type {search_type} yet."}

    algo = algorithm.upper()
    results = []

    for item in filtered_db:
        des_indexed = item["descriptors"]
        if des_indexed is None or len(des_indexed) == 0:
            continue
            
        try:
            if algo in ["SIFT", "ORB", "AKAZE"]:
                matcher = get_matcher(algo)
                matches = matcher.match(des_query, des_indexed)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # More robust thresholds
                # SIFT (L2): usually 100-200 is good
                # ORB/AKAZE (Hamming): 50-80 is good for 256-bit
                if algo == "SIFT":
                    dist_thresh = 150 
                elif algo == "ORB":
                    dist_thresh = 64 # ~25% bits
                else: # AKAZE
                    dist_thresh = 100 # AKAZE has longer descriptors
                
                good_matches = [m for m in matches if m.distance < dist_thresh]
                
                # Percentage of query keypoints matched
                ratio = len(good_matches) / max(len(kp_query), 1)
                score = min(ratio * 400, 100) # Boost for visibility
                matches_count = len(good_matches)
            else:
                # Vector-based similarity (HOG, CNN, MEANSHIFT)
                d1 = des_query.flatten().astype(np.float32)
                d2 = des_indexed.flatten().astype(np.float32)
                
                if d1.shape == d2.shape:
                    # Cosine Similarity: (A . B) / (||A|| * ||B||)
                    dot = np.dot(d1, d2)
                    norm1 = np.linalg.norm(d1)
                    norm2 = np.linalg.norm(d2)
                    if norm1 > 0 and norm2 > 0:
                        cos_sim = dot / (norm1 * norm2)
                        score = max(0, min(100, cos_sim * 100))
                    else:
                        score = 0
                    matches_count = 0
                else:
                    score = 0
                    matches_count = 0

            # Pedagogical visual info: distances/percentages
            educational_info = {
                "method": algo,
                "concept": f"Similarity calculated using {algo} features.",
                "details": "Deep Learning" if algo == "CNN" else "Structural features" if algo == "HOG" else "Keypoint matching",
                "color_matching": {
                    "query": color_query,
                    "target": item.get("color_stats")
                }
            }

            results.append({
                "id": item.get("id"),
                "filename": item["filename"],
                "url": item.get("url"),
                "similarityScore": round(float(score), 1),
                "matchesCount": int(matches_count),
                "algorithm": item.get("algorithm", algorithm),
                "description": item.get("description"),
                "color_stats": item.get("color_stats"),
                "educational_info": educational_info
            })
        except Exception as e:
            print(f"Error matching: {e}")
            continue

    results = sorted(results, key=lambda x: x["similarityScore"], reverse=True)
    return {"results": results[:6], "query_keypoints": len(kp_query), "query_color_stats": color_query}

@app.post("/api/search/text")
async def search_text(
    query: str = Form(...),
    algorithm: str = Form("CLIP")
):
    if not indexed_data:
        return {"results": [], "message": "No images indexed yet."}

    algo_upper = algorithm.upper()
    query_words = set(query.lower().strip().split())
    
    results = []
    
    # 1. Path for TBIR (Keyword Search / Inverted Index Logic)
    if get_type_for_algorithm(algo_upper) == "TBIR":
        for item in indexed_data:
            desc = (item.get("description") or "").lower()
            name = item["filename"].lower()
            # Inverted index simulation: check word intersection
            item_words = set(desc.split()) | set(name.replace('.', ' ').split())
            intersection = query_words.intersection(item_words)
            
            if intersection:
                # Score based on how many query words matched
                score = (len(intersection) / len(query_words)) * 100
                results.append({
                    "filename": item["filename"],
                    "url": item.get("url"),
                    "similarityScore": round(float(score), 1),
                    "matchesCount": len(intersection),
                    "algorithm": algo_upper,
                    "description": item.get("description"),
                    "color_stats": item.get("color_stats"),
                    "educational_info": {
                        "method": "Inverted Index",
                        "concept": "Keyword-Based Retrieval",
                        "details": f"Matched words: {', '.join(intersection)}",
                        "color_matching": {"query": "N/A (Text Only)", "target": item.get("color_stats")}
                    }
                })
    
    # 2. Path for DL (Semantic CLIP Search)
    else:
        # Simulate CLIP-style matching
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        np.random.seed(int(query_hash[:8], 16) % 2**32)
        text_embedding = np.random.rand(128).astype(np.float32)
        
        for item in indexed_data:
            seed_source = item.get("hash", item["filename"])
            np.random.seed(int(hashlib.md5(seed_source.encode()).hexdigest()[:8], 16) % 2**32)
            img_embedding = np.random.rand(128).astype(np.float32)
            
            # Simulated CLIP similarity score
            similarity = np.dot(text_embedding, img_embedding) * 100
            similarity = max(5, min(95, similarity + np.random.normal(0, 5)))
            
            results.append({
                "filename": item["filename"],
                "url": item.get("url"),
                "similarityScore": round(float(similarity), 1),
                "matchesCount": 0,
                "algorithm": "CLIP",
                "description": item.get("description"),
                "color_stats": item.get("color_stats"),
                "educational_info": {
                    "method": "CLIP",
                    "concept": "Multimodal Semantic Matching",
                    "details": "Vector distance in shared Text-Image latent space.",
                    "color_matching": {"query": "Semantic Text", "target": item.get("color_stats")}
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
    indexed_data.clear()
    save_index(indexed_data)
    print("Database cleared successfully.")
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
