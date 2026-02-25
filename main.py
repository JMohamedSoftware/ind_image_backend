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
import threading
import pymongo
from dotenv import load_dotenv

load_dotenv()

# ─── DATABASE CONFIG ───────────────────────────────────────────────────────
MONGO_URL = os.environ.get("MONGO_URL")
client = None
db = None
collection = None

if MONGO_URL:
    try:
        # Use a timeout so it doesn't hang if URL is wrong
        client = pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        # Force a connection to verify
        client.admin.command('ping')
        db = client["test"]
        collection = db["image_index"]
        print("DATABASE: Connected to MongoDB Atlas (DB: test)")
    except Exception as e:
        print(f"DATABASE ERROR: MongoDB connection failed: {e}")

# ─── UNIVERSAL FEATURE EXTRACTION ──────────────────────────────────────────

# Define which algorithms belong to which category for "Universal Pool" processing
CATEGORY_MAP = {
    "CBIR": ["ORB", "SIFT", "AKAZE", "HOG"],
    "DL": ["CNN", "SVM", "K-MEANS", "PCA", "MEANSHIFT", "CLIP"],
    "TBIR": ["TBIR_BASIC"]
}

app = FastAPI()

# Thread lock for persistence stability
save_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use absolute path for persistence
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index.pkl")

def save_index(data):
    # Save to MongoDB if available
    if collection is not None:
        try:
            # We skip heavy threading here for MongoDB as it handles concurrent writes better
            # but we still want to be safe with the list itself
            with save_lock:
                # Prepare data for MongoDB (convert numpy to lists)
                serializable_data = []
                for item in data:
                    new_item = item.copy()
                    if "descriptors_map" in new_item:
                        new_item["descriptors_map"] = {
                            k: v.tolist() if isinstance(v, np.ndarray) else v 
                            for k, v in new_item["descriptors_map"].items()
                        }
                    # Convert any other numpy if found
                    if isinstance(new_item.get("descriptors"), np.ndarray):
                        new_item["descriptors"] = new_item["descriptors"].tolist()
                    serializable_data.append(new_item)

                # Upsert approach to keep DB clean
                for doc in serializable_data:
                    # Use hash as a semi-unique identifier to prevent duplicates
                    collection.replace_one({"hash": doc["hash"]}, doc, upsert=True)
                print(f"DATABASE: Upserted {len(serializable_data)} items to MongoDB")
        except Exception as e:
            print(f"DATABASE ERROR: MongoDB save failed: {e}")

    # Always keep a local backup for safety
    with save_lock:
        try:
            with open(INDEX_FILE, "wb") as f:
                pickle.dump(data, f)
            print(f"DATABASE: Saved local backup to {INDEX_FILE}")
        except Exception as e:
            print(f"DATABASE ERROR: Local save failed: {e}")

def load_index():
    data = []
    
    # 1. Try Loading from MongoDB
    if collection is not None:
        try:
            docs = list(collection.find({}))
            if docs:
                for doc in docs:
                    doc.pop("_id", None) # Remove Mongo specific ID
                    # Convert lists back to numpy for CV2
                    if "descriptors_map" in doc:
                        doc["descriptors_map"] = {
                            k: np.array(v, dtype=np.float32 if k == "SIFT" else np.uint8) if isinstance(v, list) else v
                            for k, v in doc["descriptors_map"].items()
                        }
                    data.append(doc)
                print(f"DATABASE: Loaded {len(data)} items from MongoDB Atlas")
                return data
        except Exception as e:
            print(f"DATABASE ERROR: MongoDB load failed: {e}")

    # 2. Fallback to Local + Migration
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "rb") as f:
                data = pickle.load(f)
                print(f"DATABASE: Loaded {len(data)} items from local backup")
                
                # If we have local data but Mongo was empty/reachable, migrate it
                if collection is not None and len(data) > 0:
                    print("DATABASE: Migrating local data to MongoDB...")
                    save_index(data)
                
                return data
        except Exception as e:
            print(f"DATABASE ERROR: Local load failed: {e}")

    print("DATABASE: No existing data found.")
    return []
    return []

indexed_data = load_index()

def get_image_hash(contents):
    return hashlib.md5(contents).hexdigest()

def get_detector(algorithm: str):
    algo = algorithm.upper() if algorithm else "ORB"
    if algo == "SIFT":
        return cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.04)
    elif algo == "AKAZE":
        return cv2.AKAZE_create()
    elif algo == "ORB":
        return cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    else:
        return None

def extract_hog(img):
    # Resize to 64x128 to match frontend documentation of 3,780 features
    img_resized = cv2.resize(img, (64, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    # 64x128 image results in 3,780 features with default params
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
    if algo in ["CNN", "SVM", "K-MEANS", "PCA", "MEANSHIFT"]: return "DL"
    if algo.startswith("TBIR"): return "TBIR"
    if algo in ["SIFT", "ORB", "AKAZE", "HOG"]: return "CBIR"
    return "CBIR" # Default

def process_image_content(contents, algorithm="ORB", description=None):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    algo = algorithm.upper()
    color_stats = get_color_histogram(img)
    
    if algo == "HOG":
        features = extract_hog(img)
        return [], features, color_stats
    elif algo in ["CNN", "SVM", "K-MEANS", "PCA", "MEANSHIFT", "CLIP"]:
        # Simulate feature extraction for educational demo
        # Salted Seed: adding algo name ensures different algos produce different rankings (DIVERSITY)
        algo_salt = hashlib.md5(algo.encode()).hexdigest()[:4]
        
        # Ensure seed_text is ALWAYS defined for robustness
        img_hash = get_image_hash(contents) if contents else "00000000"
        if algo == "CLIP" and description:
            seed_text = f"{description.lower().strip()}_{algo_salt}"
        else:
            seed_text = f"{img_hash[:8]}_{algo_salt}"
            
        np.random.seed(int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16) % 2**32)
            
        # Different lengths for different algos
        if algo == "CNN": length = 4096
        elif algo == "CLIP": length = 512
        elif algo == "SVM": length = 64
        elif algo == "MEANSHIFT": length = 16
        else: length = 32
        
        # Use randn (not rand) for centered distributions and normalize to unit length
        features = np.random.randn(length).astype(np.float32)
        features /= (np.linalg.norm(features) + 1e-8) # L2 Normalization
        return [], features, color_stats
    
    detector = get_detector(algo)
    if detector:
        kp, des = detector.detectAndCompute(img, None)
        kp_coords = [(int(k.pt[0]), int(k.pt[1])) for k in kp] if kp else []
        return kp_coords, des, color_stats
    
    # Handle TBIR or unknown algorithms by returning empty features instead of None
    # This prevents the indexer from failing due to "No features found"
    return [], None, color_stats

# ──────────────────────────────────────────────────────────────────────────

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
    return {"status": "online", "message": "Visual Search API — Educational Edition", "version": "1.1.0"}

@app.get("/health")
async def health():
    return {"status": "ok", "indexed_count": len(indexed_data)}

def format_indexed_item(item):
    """
    Formats an indexed item for the frontend, respecting pedagogical purity.
    If the algorithm is TBIR, we strip visual features like color_stats and keypoints.
    """
    algo = item.get("algorithm", "ORB")
    is_tbir = get_type_for_algorithm(algo) == "TBIR"
    
    formatted = {
        "id": item.get("id", item["filename"]),
        "filename": item["filename"],
        "url": item.get("url"),
        "algorithm": item.get("algorithm", algo), # Keep original index algorithm for metadata
        "description": item.get("description"),
        "type": item.get("type", "CBIR"),
    }
    
    # Pedagogical Purity: In TBIR, the system "doesn't see" pixels/colors
    if not is_tbir:
        formatted["keypoint_count"] = item.get("keypoint_count", 0)
        formatted["color_stats"] = item.get("color_stats")
    
    return formatted

@app.get("/api/index/list")
async def list_indexed_images(type: str = None):
    # Return images filtered by category (TBIR, CBIR, DL)
    # This allows all images in a category to show up regardless of current algo
    filtered = indexed_data
    if type:
        filtered = [img for img in indexed_data if img.get("type", "CBIR") == type.upper()]
    
    return [format_indexed_item(img) for img in filtered]

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
            target_filename = filename or url.split('/')[-1].split('?')[0]
        except Exception as e:
            return {"error": f"Failed to fetch image from URL: {str(e)}"}
    elif file:
        contents = await file.read()
        target_filename = file.filename
    else:
        return {"error": "No image provided"}

    img_hash = get_image_hash(contents)
    
    result = process_image_content(contents, algorithm, description)
    if result is None:
        return {"status": "error", "message": "Could not decode image or unsupported algorithm"}

    kp_coords, des, color_stats = result # Standard result for the indexing algo
    
    # Determine type based on algorithm
    img_type = get_type_for_algorithm(algorithm)
    
    # UNIVERSAL DESCRIPTOR EXTRACTION:
    # Compute descriptors for ALL known algorithms to populate a truly universal pool
    descriptors_map = {}
    total_kp_count = 0
    
    # Flatten all algos from CATEGORY_MAP
    all_algos = [algo for group in CATEGORY_MAP.values() for algo in group]
    
    for algo_id in all_algos:
        try:
            # Re-process for each algo (efficient enough for demo)
            algo_res = process_image_content(contents, algo_id, description)
            if algo_res:
                _, algo_des, _ = algo_res
                descriptors_map[algo_id] = algo_des
                if algo_id == algorithm.upper():
                    total_kp_count = len(algo_res[0])
        except Exception as e:
            print(f"Failed to extract {algo_id}: {e}")

    global indexed_data
    # Remove any existing entry with the same hash OR filename
    indexed_data = [x for x in indexed_data if x.get("hash") != img_hash and x.get("filename") != target_filename]
    
    indexed_data.append({
        "id": str(uuid.uuid4()),
        "filename": target_filename,
        "url": url if url else None,
        "hash": img_hash,
        "descriptors": des, # Legacy field for safety
        "descriptors_map": descriptors_map, # New multi-descriptor storage
        "algorithm": algorithm.upper(),
        "type": img_type,
        "keypoint_count": total_kp_count,
        "description": description,
        "color_stats": color_stats,
        "indexed_at": time.time()
    })
    save_index(indexed_data)
    
    return {
        "status": "success",
        "id": indexed_data[-1]["id"],
        "filename": target_filename,
        "keypoints": total_kp_count,
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
    algorithm: str = Form("ORB"),
    query_text: str = Form(None)
):
    contents = None
    kp_query, des_query, color_query = [], None, [0, 0, 0]
    if url:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            contents = response.content
        except Exception as e:
            return {"error": f"Failed to fetch image from URL: {str(e)}"}
    elif file:
        contents = await file.read()
    
    # Allow proceeding if we have text and it's CLIP
    if not contents and not (query_text and algorithm.upper() == "CLIP"):
        return {"error": "Provide an image or text (for CLIP)"}

    kp_query, des_query, color_query = [], None, None
    if contents:
        res = process_image_content(contents, algorithm)
        if res:
            kp_query, des_query, color_query = res
            
    algo = algorithm.upper()
    search_type = get_type_for_algorithm(algorithm)
    results = []

    # 1. SPECIAL CASE: Multimodal CLIP Fusion
    if algo == "CLIP" and query_text:
        # Text vector
        query_hash = hashlib.md5(query_text.lower().strip().encode()).hexdigest()
        np.random.seed(int(query_hash[:8], 16) % 2**32)
        text_embedding = np.random.rand(128).astype(np.float32)
        
        # Fuse if image exists
        if des_query is not None:
            final_query_embedding = (text_embedding + des_query.flatten()[:128]) / 2
            method_name = "CLIP Multimodal Fusion"
            concept_text = f"Neural fusion: Image + '{query_text}'"
        else:
            final_query_embedding = text_embedding
            method_name = "CLIP Semantic Search"
            concept_text = f"Semantic matching for '{query_text}'"
            
        for item in indexed_data:
            seed_source = item.get("hash", item["filename"])
            np.random.seed(int(hashlib.md5(seed_source.encode()).hexdigest()[:8], 16) % 2**32)
            img_embedding = np.random.rand(128).astype(np.float32)
            
            similarity = np.dot(final_query_embedding, img_embedding) * 100
            similarity = max(5, min(95, similarity + np.random.normal(0, 3)))
            
            results.append({
                **format_indexed_item(item),
                "similarityScore": round(float(similarity), 1),
                "matchesCount": 0,
                "educational_info": {
                    "method": method_name,
                    "concept": concept_text,
                    "details": "Vector alignment in multimodal latent space.",
                    "color_matching": {"query": "Neural Context", "target": item.get("color_stats") if get_type_for_algorithm(item.get("algorithm")) != "TBIR" else "N/A"}
                }
            })
        
        results = sorted(results, key=lambda x: x["similarityScore"], reverse=True)
        return {"results": results[:6]}

    # 2. STANDARD CASE: Visual Similarity
    if not contents and not (query_text and algorithm.upper() == "CLIP"):
         return {"results": [], "message": "Provide an image or text (for CLIP)"}

    # UNIVERSAL SEARCH: 
    # Search across ALL indexed images. If an image is missing descriptors for the
    # current algorithm, we'll handle it below (robust fallback).
    filtered_db = indexed_data
    
    # Robust Fallback: If visual descriptors failed to extract (e.g. blurry image)
    # we fall back to color comparison so the user always sees results.
    if des_query is None and algo != "CLIP":
        print(f"No visual features for {algo}. Using color fallback.")
        results = []
        safe_color_query = color_query if color_query is not None else [128, 128, 128]
        for item in filtered_db:
            item_color = item.get("color_stats")
            if not item_color:  # Skip items with no color stats (e.g. TBIR-indexed images)
                continue
            # Simple color distance score
            dist = np.linalg.norm(np.array(safe_color_query) - np.array(item_color))
            score = max(0, 100 - (dist / 4.41)) # 4.41 is max possible distance scaled for 0-100
            results.append({
                **format_indexed_item(item),
                "similarityScore": round(float(score), 1),
                "matchesCount": 0,
                "educational_info": {
                    "method": f"{algo} (Color Fallback)",
                    "concept": f"{algo} failed to find keypoints; falling back to color distribution.",
                    "details": "When an image is too blurry or lacks contrast, detectors skip it. We use color clusters to maintain search functionality.",
                    "color_matching": {"query": safe_color_query, "target": item_color}
                }
            })
        results = sorted(results, key=lambda x: x["similarityScore"], reverse=True)
        return {"results": results[:6], "query_keypoints": 0, "query_color_stats": safe_color_query, "message": "Low feature detection - using color-based similarity fallback."}


    if not filtered_db:
        return {"results": [], "message": f"No images in the {search_type} pool yet. Please index some first!"}

    for item in filtered_db:
        # Pull correct descriptor from the map for the requested search algorithm
        des_map = item.get("descriptors_map", {})
        des_indexed = des_map.get(algo)
        
        if des_indexed is None or len(des_indexed) == 0:
            continue
            
        try:
            if algo in ["SIFT", "ORB", "AKAZE"]:
                # Ensure correct dtype for matcher
                q = np.array(des_query, dtype=np.float32 if algo == "SIFT" else np.uint8)
                idx_d = np.array(des_indexed, dtype=np.float32 if algo == "SIFT" else np.uint8)
                
                if q.ndim != 2 or idx_d.ndim != 2:
                    continue  # Skip malformed descriptors
                
                matcher = get_matcher(algo)
                matches = matcher.match(q, idx_d)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Thresholds: distances are algorithm-specific
                if algo == "SIFT":   dist_thresh = 200  # SIFT L2 distances can be large (float)
                elif algo == "ORB":  dist_thresh = 64   # ORB Hamming distance (8-bit, 256 bits)
                elif algo == "AKAZE": dist_thresh = 80
                else: dist_thresh = 100
                
                good_matches = [m for m in matches if m.distance < dist_thresh]

                # Minimum required good matches before given a meaningful score
                min_good = 4 if algo in ["ORB", "AKAZE"] else 3

                # Score: needs enough good matches to be relevant
                if len(good_matches) >= min_good:
                    best_dist = matches[0].distance
                    # Refined normalized score: distance closer to 0 is better
                    normalized = 1 - min(best_dist / (dist_thresh * 2), 1.0)
                    # Denominator protection and better ratio
                    count_ratio = min(len(good_matches) / max(len(kp_query), 50), 1.0)
                    
                    # count_ratio weighted heavily — more shared keypoints = more similar
                    # For self-match: normalized=1.0, count_ratio=1.0 -> score=100
                    score = min((normalized * 0.4 + count_ratio * 0.6) * 100, 100)
                elif len(good_matches) > 0:
                    # Very few matches — give a very low score (< 15)
                    score = (len(good_matches) / min_good) * 15
                else:
                    score = 0
                matches_count = len(good_matches)
            else:
                # Vector-based similarity (HOG, CNN, MEANSHIFT, etc.)
                d1 = des_query.flatten().astype(np.float32)
                d2 = des_indexed.flatten().astype(np.float32)
                
                if d1.shape == d2.shape:
                    dot = np.dot(d1, d2)
                    norm1 = np.linalg.norm(d1)
                    norm2 = np.linalg.norm(d2)
                    if norm1 > 1e-6 and norm2 > 1e-6: # Prevent division by zero
                        cos_sim = dot / (norm1 * norm2)
                        # HOG similarity often naturally sits high, let's scale it for better UX
                        # (cos_sim - bias) / scale
                        score = max(0, min(100, (cos_sim - 0.5) * 200)) if algo == "HOG" else max(0, min(100, cos_sim * 100))
                    else:
                        score = 0
                    matches_count = 0
                else:
                    score = 0
                    matches_count = 0

            educational_info = {
                "method": algo,
                "concept": f"Similarity calculated using {algo} features.",
                "details": "Deep Learning Context" if algo == "CNN" else "Structural features" if algo == "HOG" else "Keypoint matching",
                "color_matching": {
                    "query": color_query,
                    "target": item.get("color_stats")
                }
            }

            results.append({
                **format_indexed_item(item),
                "similarityScore": round(float(score), 1),
                "matchesCount": int(matches_count),
                "educational_info": educational_info
            })
        except Exception as e:
            print(f"Error matching: {e}")
            continue

    safe_color_query = color_query if color_query is not None else [128, 128, 128]
    results = sorted(results, key=lambda x: x["similarityScore"], reverse=True)
    return {"results": results, "query_keypoints": len(kp_query), "query_color_stats": safe_color_query}

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
            query_words_clean = [w.strip(",.!?") for w in query.lower().split() if len(w) > 2]
            if not query_words_clean: query_words_clean = [query.lower().strip()]
            
            for item in indexed_data:
                desc = (item.get("description") or "").lower()
                name = item["filename"].lower().replace('.', ' ')
                content_pool = f"{desc} {name}"
                
                matches = []
                for word in query_words_clean:
                    # Strict word boundary match to avoid partial matches like "car" in "carpet"
                    if f" {word} " in f" {content_pool} ":
                        matches.append(word)
                
                if matches:
                    # Better scoring for exact word matches (Jaccard-ish)
                    # Unique matches / Total query words
                    score = (len(list(set(matches))) / len(list(set(query_words_clean)))) * 100
                    results.append({
                        **format_indexed_item(item),
                        "similarityScore": round(float(score), 1),
                        "matchesCount": len(matches),
                        "educational_info": {
                            "method": "Inverted Index",
                            "concept": "Keyword-Based Retrieval",
                            "details": f"Matched words: {', '.join(matches)}",
                            "color_matching": {"query": "Text Only", "target": "N/A (Hidden in TBIR)"}
                        }
                    })
    
    # 2. Path for DL (Semantic CLIP Search)
    else:
        # Simulate CLIP-style matching
        algo_salt = hashlib.md5(algo_upper.encode()).hexdigest()[:4]
        query_hash = hashlib.md5(f"{query.lower().strip()}_{algo_salt}".encode()).hexdigest()
        np.random.seed(int(query_hash[:8], 16) % 2**32)
        text_embedding = np.random.randn(128).astype(np.float32)
        text_embedding /= (np.linalg.norm(text_embedding) + 1e-8)
        
        for item in indexed_data:
            seed_source = f"{item.get('hash', item['filename'])}_{algo_salt}"
            np.random.seed(int(hashlib.md5(seed_source.encode()).hexdigest()[:8], 16) % 2**32)
            img_embedding = np.random.randn(128).astype(np.float32)
            img_embedding /= (np.linalg.norm(img_embedding) + 1e-8)
            
            # Simulated Cosine Similarity (Dot product of normalized vectors)
            similarity = np.dot(text_embedding, img_embedding) * 100
            # Scale to UX range but keep variance
            similarity = max(5, min(98, (similarity + 1) * 45)) # Range -1..1 shifted to roughly 0..90
            similarity += np.random.normal(0, 2)
            
            results.append({
                **format_indexed_item(item),
                "similarityScore": round(float(similarity), 1),
                "matchesCount": 0,
                "educational_info": {
                    "method": "CLIP",
                    "concept": "Multimodal Semantic Matching",
                    "details": "Vector distance in shared Text-Image latent space.",
                    "color_matching": {"query": "Semantic Text", "target": item.get("color_stats") if get_type_for_algorithm(item.get("algorithm")) != "TBIR" else "N/A"}
                }
            })

    results = sorted(results, key=lambda x: x["similarityScore"], reverse=True)
    return {"results": results}

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
    if collection is not None:
        try:
            collection.delete_many({})
            print("MongoDB collection cleared.")
        except Exception as e:
            print(f"MongoDB clear failed: {e}")
    save_index(indexed_data)
    print("Database cleared successfully.")
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
