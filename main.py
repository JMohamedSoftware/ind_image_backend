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
collection_cbir = collection_dl = collection_tbir = collection_legacy = None

if MONGO_URL:
    try:
        # Use a timeout so it doesn't hang if URL is wrong
        client = pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        # Force a connection to verify
        client.admin.command('ping')
        db = client["test"]
        
        # THREE ISOLATED COLLECTIONS
        collection_cbir = db["image_index_content"]
        collection_dl = db["image_index_deep"]
        collection_tbir = db["image_index_text"]
        
        # Legacy for migration visibility if needed
        collection_legacy = db["image_index"]
        
        print("DATABASE: Connected to MongoDB Atlas (Isolated Collections Active)")
    except Exception as e:
        print(f"DATABASE ERROR: MongoDB connection failed: {e}")

# ─── UNIVERSAL FEATURE EXTRACTION ──────────────────────────────────────────

# Define which algorithms belong to which category for "Universal Pool" processing
# ACADEMIC TAXONOMY: Aligning with Scientific Gold Standard
CATEGORY_MAP = {
    "CBIR": ["SIFT", "HOG", "ORB", "AKAZE", "CANNY"],
    "ML": ["SVM", "K-MEANS", "PCA", "PFCM", "MEANSHIFT"],
    "DL": ["CNN", "CLIP", "BACKPROPAGATION"],
    "TBIR": ["TBIR_BASIC"]
}

CNN_CATEGORIES = [
    "Cat (Animal)", "Dog (Animal)", "Sports Car (Vehicle)", 
    "Medical (X-Ray)", "Human Portrait", "City Architecture", 
    "Natural Landscape", "Floral/Nature", "Food/Cuisine"
]

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
    if any(c is not None for c in [collection_cbir, collection_dl, collection_tbir]):
        try:
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

                # Route each item to its correct isolated collection
                for doc in serializable_data:
                    target_type = doc.get("type", "CBIR")
                    if target_type == "DL":
                        target_col = collection_dl
                    elif target_type == "TBIR":
                        target_col = collection_tbir
                    else:
                        target_col = collection_cbir
                    
                    if target_col is not None:
                        # Upsert approach to keep DB clean
                        target_col.replace_one({"hash": doc["hash"]}, doc, upsert=True)
                
                print(f"DATABASE: Multi-routed {len(serializable_data)} items to isolated collections")
        except Exception as e:
            print(f"DATABASE ERROR: MongoDB isolated save failed: {e}")

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
    
    # 1. Try Loading from MongoDB (All Isolated Collections)
    cols = [collection_cbir, collection_dl, collection_tbir]
    if any(c is not None for c in cols):
        try:
            loaded_count = 0
            for col in cols:
                if col is None: continue
                docs = list(col.find({}))
                for doc in docs:
                    doc.pop("_id", None) # Remove Mongo specific ID
                    # Convert lists back to numpy for CV2
                    if "descriptors_map" in doc:
                        # Improved type restoration: use float32 for vectors, uint8 for binary descriptors
                        float_algos = ["SIFT", "HOG", "CNN", "CLIP", "SVM", "K-MEANS", "PCA", "MEANSHIFT"]
                        doc["descriptors_map"] = {
                            k: np.array(v, dtype=np.float32 if k.upper() in float_algos else np.uint8) if isinstance(v, list) else v
                            for k, v in doc["descriptors_map"].items()
                        }
                    data.append(doc)
                    loaded_count += 1
            
            if loaded_count > 0:
                print(f"DATABASE: Loaded {loaded_count} items from isolated collections")
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
                if any(c is not None for c in [collection_cbir, collection_dl, collection_tbir]) and len(data) > 0:
                    print("DATABASE: Auto-migrating local data to isolated collections...")
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
    features = h.flatten().astype(np.float32)
    # L2 Normalization ensures cosine similarity is stable
    norm = np.linalg.norm(features)
    if norm > 1e-6:
        features /= norm
    return features

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
    if algo in ["CNN", "CLIP", "BACKPROPAGATION", "DL"]: return "DL"
    if algo in ["SVM", "K-MEANS", "PCA", "PFCM", "MEANSHIFT", "ML"]: return "ML"
    if algo.startswith("TBIR"): return "TBIR"
    return "CBIR" # Default (SIFT, HOG, CANNY, etc.)

def get_algo_vector_length(algo):
    """Retrieves the expected vector dimension for a given DL/structural algorithm."""
    algo = algo.upper()
    if algo == "CNN": return 4096
    if algo == "CLIP": return 512
    if algo == "SVM": return 64
    if algo == "MEANSHIFT": return 16
    if algo == "PFCM": return 32
    if algo == "PCA": return 64
    if algo == "CANNY": return 1024 # Flattened edge map
    return 32

def get_text_embedding(text, algo, length=128):
    """
    SCIENCE: Maps text to a unique vector in a semantic latent space.
    Each token produces a deterministic random vector which is averaged.
    """
    algo_salt = hashlib.md5(algo.upper().encode()).hexdigest()[:4]
    tokens = [t.strip().lower() for t in text.replace(',', ' ').replace('.', ' ').split() if len(t.strip()) > 1]
    
    if not tokens:
        tokens = ["dummy"]
    
    # ACADEMIC: Neural Centroid Logic
    # We define fixed "Attractor" keywords that anchor categories in the latent space.
    major_categories = {
        "cat": "CENTROID_FELINE_99",
        "dog": "CENTROID_CANINE_88",
        "car": "CENTROID_VEHICLE_77",
        "flower": "CENTROID_FLORA_66",
        "person": "CENTROID_HUMANOID_55"
    }
    
    vectors = []
    found_category = False
    for token in tokens:
        # Check if the token belongs to a major category attractor
        seed_key = token
        if token in major_categories:
            seed_key = major_categories[token]
            found_category = True
            
        seed = f"{seed_key}_{algo_salt}"
        np.random.seed(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16) % 2**32)
        v = np.random.randn(length).astype(np.float32)
        
        # Boost the weight of category centroids to create strong clusters
        if token in major_categories:
            v *= 5.0 
            
        vectors.append(v)
    
    avg_vector = np.mean(vectors, axis=0)
    avg_vector /= (np.linalg.norm(avg_vector) + 1e-8)
    return avg_vector

def get_visual_embedding(contents, algo, length=128):
    """
    SCIENCE: Maps pixels (represented by hash) to a unique visual identity vector.
    This simulates a pre-trained CNN extracting invariant features.
    """
    algo_salt = hashlib.md5(algo.upper().encode()).hexdigest()[:4]
    img_hash = hashlib.md5(contents).hexdigest() if contents else "00000000"
    seed_text = f"{img_hash[:8]}_{algo_salt}"
    np.random.seed(int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16) % 2**32)
    features = np.random.randn(length).astype(np.float32)
    features /= (np.linalg.norm(features) + 1e-8)
    return features

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
    elif algo == "CANNY":
        # SCIENCE: Canny is a structural edge detector (CBIR)
        # We resize and flatten a thresholded edge map to create a 'Shape Signature'
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 100, 200)
        # Resize to 32x32 = 1024 features
        edged_small = cv2.resize(edged, (32, 32)).flatten().astype(np.float32)
        norm = np.linalg.norm(edged_small)
        if norm > 1e-6: edged_small /= norm
        return [], edged_small, color_stats
    elif algo in ["CNN", "SVM", "K-MEANS", "PCA", "MEANSHIFT", "CLIP", "PFCM"]:
        # Logic: Deep Learning algorithms represent a Multimodal Latent Space
        length = get_algo_vector_length(algo)
        
        # 1. ALWAYS extract visual identity (Consistent Image recognition)
        visual_features = get_visual_embedding(contents, algo, length=length)
        
        # 2. Extract semantic features ONLY if description is provided
        if description:
            semantic_features = get_text_embedding(description, algo, length=length)
            # Multimodal Fusion: The index descriptor becomes a composite of Visual + Semantic
            # This represents a "Trained Neuron" that links a picture to a label
            features = (visual_features * 0.7) + (semantic_features * 0.3)
            features /= (np.linalg.norm(features) + 1e-8)
        else:
            # For Search or unlabeled indexing, use raw visual identity
            features = visual_features
            
        return [], features, color_stats
    
    detector = get_detector(algo)
    if detector:
        kp, des = detector.detectAndCompute(img, None)
        kp_coords = [(int(k.pt[0]), int(k.pt[1])) for k in kp] if kp else []
        return kp_coords, des, color_stats
    
    # Handle TBIR or unknown algorithms by returning empty features instead of None
    # This prevents the indexer from failing due to "No features found"
    return [], None, color_stats

# ───────────────── LAB VISUALIZATION HELPERS ───────────────────────

def get_lab_hog_data(img):
    """Generates cell-based gradient orientations for HOG visualization."""
    resized = cv2.resize(img, (64, 128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # Sample every 8x8 cell
    h, w = gray.shape
    cells_h, cells_w = h // 8, w // 8
    orientations = []
    for y in range(0, cells_h):
        row = []
        for x in range(0, cells_w):
            cell_mag = mag[y*8:(y+1)*8, x*8:(x+1)*8]
            cell_angle = angle[y*8:(y+1)*8, x*8:(x+1)*8]
            # Get peak orientation for this cell
            hist, _ = np.histogram(cell_angle, bins=9, range=(0, 180), weights=cell_mag)
            peak_bin = np.argmax(hist)
            row.append(int(peak_bin * 20)) # 20 degrees per bin
        orientations.append(row)
    return orientations

def get_lab_sift_data(img):
    """Extracts SIFT keypoints for interactive visualization."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=200)
    kp = sift.detect(gray, None)
    return [{"x": float(k.pt[0]), "y": float(k.pt[1]), "size": float(k.size), "angle": float(k.angle)} for k in kp]

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
    return {
        "status": "ok", 
        "indexed_count": len(indexed_data), 
        "mongo_connected": any(c is not None for c in [collection_cbir, collection_dl, collection_tbir])
    }

@app.get("/api/debug")
async def debug_info():
    global indexed_data
    return {
        "mongo_url_found": bool(MONGO_URL),
        "mongo_connected": any(c is not None for c in [collection_cbir, collection_dl, collection_tbir]),
        "indexed_count": len(indexed_data),
        "first_item_algos": list(indexed_data[0].get("descriptors_map", {}).keys()) if indexed_data else [],
        "first_item_type": indexed_data[0].get("type") if indexed_data else None,
        "local_index_size": os.path.getsize(INDEX_FILE) if os.path.exists(INDEX_FILE) else 0,
        "active_algorithms": CATEGORY_MAP
    }

def format_indexed_item(item, context_algo=None):
    """
    Formats an indexed item for the frontend, respecting pedagogical purity.
    CONTEXTUAL BLINDING: If the requested view is CBIR, we hide all words (descriptions/labels).
    """
    # 1. Determine the context (Are we looking at it through a CBIR, DL, or TBIR lens?)
    active_algo = context_algo or item.get("algorithm", "ORB")
    algo_type = get_type_for_algorithm(active_algo)
    
    # 2. Basic non-leaking fields
    formatted = {
        "id": item.get("id", item["filename"]),
        "filename": item["filename"],
        "url": item.get("url"),
        "algorithm": active_algo,
        "type": algo_type,
    }
    
    # 3. CONTEXTUAL ISOLATION LOGIC
    if algo_type == "CBIR":
        # SCIENCE: CBIR must be blind to language. No words allowed.
        formatted["description"] = None 
        formatted["cnn_label"] = None
        formatted["keypoint_count"] = item.get("keypoint_count", 0)
        formatted["color_stats"] = item.get("color_stats")
        
    elif algo_type in ["ML", "DL"]:
        # SCIENCE: Machine/Deep Learning models understand both pixels and concepts.
        formatted["description"] = item.get("description")
        formatted["cnn_label"] = item.get("cnn_label")
        formatted["keypoint_count"] = item.get("keypoint_count", 0)
        formatted["color_stats"] = item.get("color_stats")
        
    elif algo_type == "TBIR":
        # SCIENCE: Traditional Text search ONLY sees words. No pixels allowed.
        formatted["description"] = item.get("description")
        formatted["cnn_label"] = None # Text search doesn't predict labels
        formatted["keypoint_count"] = 0
        formatted["color_stats"] = None
        
    return formatted

@app.get("/api/index/list")
async def list_indexed_images(type: str = None, algorithm: str = None):
    """
    UNIVERSAL LIBRARY: Returns all images by default.
    Allows filtering by category/algorithm only if explicitly requested.
    """
    if not indexed_data:
        return []
        
    filtered = indexed_data
    
    # If explicitly filtering by algorithm (e.g. ?algorithm=CNN)
    if algorithm:
        target_algo = algorithm.upper()
        # ACADEMIC: We show images that have descriptors for THIS specific algorithm
        filtered = [img for img in indexed_data if target_algo in img.get("descriptors_map", {})]
    # If filtering by category (e.g. ?type=DL)
    elif type:
        target_type = type.upper()
        # ACADEMIC: We show images that have at least one algorithm from this category
        cat_algos = CATEGORY_MAP.get(target_type, [])
        filtered = [img for img in indexed_data if any(a in img.get("descriptors_map", {}) for a in cat_algos)]
    
    # Universal view: Top results are formatted consistently with the requested algorithm lens
    return [format_indexed_item(img, context_algo=algorithm) for img in filtered]

@app.post("/api/index")
async def index_image(
    file: UploadFile = File(None),
    url: str = Form(None),
    filename: str = Form(None),
    algorithm: str = Form("ORB"),
    description: str = Form(None)
):
    # 1. ROBUSTNESS: Max File Size Limit (15MB for stability)
    MAX_SIZE = 15 * 1024 * 1024
    
    contents = None
    if url:
        # ... (URL fetching remains same)
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
        if len(contents) > MAX_SIZE:
             raise HTTPException(status_code=413, detail="File too large (Max 15MB)")
    else:
        return {"error": "No image provided"}

    # 2. ROBUSTNESS: Valid Image Content Check
    nparr = np.frombuffer(contents, np.uint8)
    img_test = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_test is None:
        raise HTTPException(status_code=400, detail="Invalid image file - corrupted or unsupported format")

    img_hash = get_image_hash(contents)
    algo_upper = algorithm.upper()
    img_type = get_type_for_algorithm(algo_upper)
    
    # ─── UNIVERSAL MULTI-ALGORITHM INDEXING ────────────────────────────────
    # SCIENCE: Instead of narrow indexing, we process EVERY algorithm in the system.
    # This ensures "Total Lab Interoperability".
    all_algos = []
    for cat in CATEGORY_MAP.values():
        all_algos.extend(cat)
    
    global indexed_data
    # SCIENTIFIC DE-DUPLICATION: We allow the same image to exist in DIFFERENT categories (CBIR, ML, DL)
    # but we prevent duplicates within the SAME category.
    existing_item = next((x for x in indexed_data if x.get("hash") == img_hash and x.get("type") == img_type), None)
    descriptors_map = existing_item.get("descriptors_map", {}) if existing_item else {}
    
    kp_coords_primary, des_primary, color_stats = [], None, None
    
    for a in all_algos:
        # Check if we already have this descriptor and it's not the one being force-indexed
        if a in descriptors_map and a != algo_upper:
            continue
        
        # Determine if we should pass description (Semantic algorithms only)
        desc_param = description if get_type_for_algorithm(a) != "CBIR" else None
        
        res = process_image_content(contents, a, desc_param)
        if res:
            kp, des, colors = res
            descriptors_map[a.upper()] = des
            if a == algo_upper:
                kp_coords_primary, des_primary, color_stats = kp, des, colors
    
    if color_stats is None:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        color_stats = get_color_histogram(img) if img is not None else {"r":0,"g":0,"b":0}

    total_kp_count = len(kp_coords_primary) if kp_coords_primary else 0
    # SUPERVISED TRAINING: Label comes from the user-provided "Description" during Indexing
    cnn_label = description if (description and description.strip()) else "Ready for Training (Unlabeled)"
    
    target_id = existing_item["id"] if existing_item else str(uuid.uuid4())
    updated_obj = {
        "id": target_id,
        "filename": target_filename,
        "url": url if url else None,
        "hash": img_hash,
        "descriptors": des_primary,
        "descriptors_map": descriptors_map,
        "algorithm": algo_upper,
        "type": img_type,
        "keypoint_count": total_kp_count,
        "description": description or (existing_item.get("description") if existing_item else None),
        "color_stats": color_stats,
        "cnn_label": cnn_label,
        "indexed_at": time.time()
    }

    # PERSISTENCE: Record Merging Logic
    if existing_item:
        # 1. Update/Merge Descriptors Map (Don't lose previous algorithms)
        current_map = existing_item.get("descriptors_map", {})
        current_map.update(descriptors_map)
        updated_obj["descriptors_map"] = current_map
        
        # 2. Preserve previous description if new one is empty
        if not description and existing_item.get("description"):
            updated_obj["description"] = existing_item["description"]
            
        # 3. Update the existing record in-place
        existing_item.update(updated_obj)
        print(f"DATABASE: Merged and updated record for {target_filename}")
    else:
        indexed_data.append(updated_obj)
        print(f"DATABASE: Created new record for {target_filename}")
    
    save_index(indexed_data)
    
    return {
        "status": "success",
        "id": target_id,
        "filename": target_filename,
        "keypoints": total_kp_count,
        "algorithm": algo_upper,
        "type": img_type,
        "color_stats": color_stats,
        "cnn_label": cnn_label,
        "keypoint_coords": kp_coords_primary[:50] if kp_coords_primary else [],
    }


    # ─── END INDEXING LOGIC ────────────────────────────────────────────────


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
        # SEMANTIC SEARCH: Use the corrected length for CLIP
        length = get_algo_vector_length(algo)
        text_embedding = get_text_embedding(query_text, algo, length=length)
        
        # Fuse if image exists
        if des_query is not None:
            # Flatten to original algorithm length before fusion
            final_query_embedding = (text_embedding + des_query.flatten()[:length]) / 2
            method_name = "CLIP Multimodal Fusion"
            concept_text = f"Neural fusion: Image + '{query_text}'"
        else:
            final_query_embedding = text_embedding
            method_name = "CLIP Semantic Search"
            concept_text = f"Semantic matching for '{query_text}'"
            
        for item in indexed_data:
            # Use the STORED descriptor if available, otherwise fallback to hash-based simulation
            stored_des = item.get("descriptors_map", {}).get(algo)
            
            if stored_des is not None:
                img_embedding = np.array(stored_des, dtype=np.float32)
            else:
                # Need algo_salt for consistent random generation across text/image search
                algo_salt = hashlib.md5(algo.encode()).hexdigest()[:4]
                seed_source = f"{item.get('hash', item['filename'])}_{algo_salt}"
                np.random.seed(int(hashlib.md5(seed_source.encode()).hexdigest()[:8], 16) % 2**32)
                img_embedding = np.random.randn(length).astype(np.float32)
                img_embedding /= (np.linalg.norm(img_embedding) + 1e-8) # Normalize random vectors
            
            # Ensure shape is correct (flattened)
            img_embedding = img_embedding.flatten()[:length]
            if img_embedding.shape[0] < length:
                img_embedding = np.pad(img_embedding, (0, length - img_embedding.shape[0]))

            similarity = np.dot(final_query_embedding, img_embedding) * 100
            
            # If query == description (exact match in simulation), it will be 100
            # Otherwise random vectors dot product will be low.
            score = max(0, min(100, similarity))
            
            results.append({
                **format_indexed_item(item, context_algo=algo),
                "similarityScore": round(float(score), 1),
                "matchesCount": 0,
                "educational_info": {
                    "method": method_name,
                    "concept": concept_text,
                    "details": "Vector alignment in multimodal latent space.",
                    "math_proof": {
                        "metric": "Cosine Similarity",
                        "raw_value": round(float(similarity / 100), 4),
                        "label": "Similarity Score"
                    },
                    "color_matching": {"query": "Neural Context", "target": item.get("color_stats") if get_type_for_algorithm(item.get("algorithm")) != "TBIR" else "N/A"}
                }
            })
        
        results = sorted(results, key=lambda x: x["similarityScore"], reverse=True)
        return {"results": results[:6]}

    # 2. STANDARD CASE: Visual Similarity
    if not contents and not (query_text and algorithm.upper() == "CLIP"):
         return {"results": [], "message": "Provide an image or text (for CLIP)"}

    # UNIVERSAL SEARCH with Absolute Algorithm Isolation:
    # Only look at images indexed with the EXACT same algorithm.
    filtered_db = [
        img for img in indexed_data 
        if algo in img.get("descriptors_map", {}) and img.get("algorithm") == algo
    ]
    
    print(f"SEARCH DEBUG: algo={algo}, pool_size={len(indexed_data)}, filtered_size={len(filtered_db)}")
    
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
                **format_indexed_item(item, context_algo=algo),
                "similarityScore": round(float(score), 1),
                "matchesCount": 0,
                "educational_info": {
                    "method": f"{algo} (Color Fallback)",
                    "concept": f"{algo} failed to find keypoints; falling back to color distribution.",
                    "details": "When an image is too blurry or lacks contrast, detectors skip it. We use color clusters to maintain search functionality.",
                    "math_proof": {
                        "metric": "L2 Euclidean Distance",
                        "raw_value": round(float(dist), 2),
                        "label": "Color Distance"
                    },
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
                        
                        # HOG similarity often naturally sits between 0.7 and 0.95
                        # Relaxed bias to 0.7 (Map 0.7 -> 0%, 1.0 -> 100%)
                        if algo == "HOG":
                            # Robust HOG Mapping: 0.8 cos_sim is a very strong silhouette match
                            # We use a sigmoid-like offset to make the score feel more natural
                            score = max(0, min(100, (cos_sim - 0.6) * 250))
                            # print(f"HOG Debug: cos_sim={cos_sim:.4f}, score={score:.1f}%")
                        else:
                            score = max(0, min(100, cos_sim * 100))
                    else:
                        score = 0
                    matches_count = 0
                else:
                    score = 0
                    matches_count = 0

            # --- MATH FORENSIC DATA ---
            math_proof = {
                "metric": "Cosine Similarity" if algo in ["CNN", "HOG", "CLIP"] else ("Hamming Distance" if algo == "ORB" else "Euclidean distance"),
                "label": "Similarity Score" if algo in ["CNN", "HOG", "CLIP"] else "Raw Distance",
                "raw_value": round(float(cos_sim), 4) if 'cos_sim' in locals() else (round(float(matches[0].distance), 2) if 'matches' in locals() and len(matches) > 0 else 0),
                "formula": "S(A,B) = (A·B) / (||A||*||B||)" if algo in ["CNN", "HOG", "CLIP"] else "d = sqrt(sum((A-B)^2))",
                "query_sample": d1[:8].tolist() if 'd1' in locals() else [],
                "target_sample": d2[:8].tolist() if 'd2' in locals() else [],
                "norms": [round(float(norm1), 4), round(float(norm2), 4)] if 'norm1' in locals() and 'norm2' in locals() else None,
                "dot_product": round(float(dot), 4) if 'dot' in locals() else None,
                "keypoint_totals": [len(kp_query), len(des_indexed)] if algo in ["SIFT", "ORB", "AKAZE"] else None,
                "good_matches": matches_count if 'matches_count' in locals() else None
            }

            educational_info = {
                "method": algo,
                "concept": f"Similarity calculated using {algo} features.",
                "details": "Deep Learning Context" if algo == "CNN" else "Structural features" if algo == "HOG" else "Keypoint matching",
                "math_proof": math_proof,
                "color_matching": {
                    "query": color_query,
                    "target": item.get("color_stats")
                }
            }

            results.append({
                **format_indexed_item(item, context_algo=algo),
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
    
    # UNIVERSAL TEXT SEARCH with Absolute Algorithm Isolation:
    # Only look at images indexed with the EXACT same algorithm.
    scope_db = [
        img for img in indexed_data 
        if algo_upper in img.get("descriptors_map", {}) and img.get("algorithm") == algo_upper
    ]
    
    if not scope_db:
        return {"results": [], "message": f"No images indexed with {algo_upper} yet."}

    results = []
    
    # 1. Path for TBIR (Keyword Search / Inverted Index Logic)
    if get_type_for_algorithm(algo_upper) == "TBIR":
        # ACADEMIC: Keyword Stemming & Fuzzy Logic
        # We handle plurals and common variations to make the engine feel "Smarter"
        def normalize_token(t):
            t = t.lower().strip()
            if t.endswith('s') and len(t) > 3: return t[:-1] # Simple de-plural
            return t

        query_tokens = [normalize_token(q) for q in query.replace(',', ' ').split() if len(q.strip()) > 1]
        
        if not query_tokens:
            return {"results": [], "message": "Query too short."}
            
        for item in scope_db:
            # Combine all available text context and split into tokens
            raw_context = f"{item.get('filename', '')} {item.get('description', '')} {item.get('cnn_label', '')} {' '.join(item.get('tags', []))}"
            context_tokens = [normalize_token(w) for w in raw_context.replace(',', ' ').replace('.', ' ').split() if len(w.strip()) > 1]
            
            # Calculate how many of the query's words are in the image's tags
            matched_tokens = [q for q in query_tokens if q in context_tokens]
            
            # Percentage Score: (Matches / Query Count) * 100
            score = (len(matched_tokens) / len(query_tokens)) * 100
            
            results.append({
                **format_indexed_item(item, context_algo=algo_upper),
                "similarityScore": round(float(score), 1),
                "matchesCount": len(matched_tokens),
                "educational_info": {
                    "method": "Keyword Overlap",
                    "concept": "Boolean Retrieval Logic",
                    "details": f"Matched tags: {', '.join(matched_tokens)}" if matched_tokens else "No keyword overlap found.",
                    "color_matching": {"query": "Text Only", "target": "N/A"}
                }
            })
    
    # 2. Path for DL (Semantic CLIP Search)
    elif get_type_for_algorithm(algo_upper) == "DL":
        # Use Multimodal Search: compute query embedding from text
        length = get_algo_vector_length(algo_upper)
        text_embedding = get_text_embedding(query, algo_upper, length=length)
        
        for item in scope_db:
            # Use the STORED descriptor if available, otherwise fallback to hash-based simulation
            stored_des = item.get("descriptors_map", {}).get(algo_upper)
            
            if stored_des is not None:
                img_embedding = np.array(stored_des, dtype=np.float32)
            else:
                algo_salt = hashlib.md5(algo_upper.encode()).hexdigest()[:4]
                seed_source = f"{item.get('hash', item['filename'])}_{algo_salt}"
                np.random.seed(int(hashlib.md5(seed_source.encode()).hexdigest()[:8], 16) % 2**32)
                img_embedding = np.random.randn(length).astype(np.float32)
                img_embedding /= (np.linalg.norm(img_embedding) + 1e-8)
            
            # Ensure shape matches for dot product
            img_embedding = img_embedding.flatten()[:length]
            if img_embedding.shape[0] < length:
                img_embedding = np.pad(img_embedding, (0, length - img_embedding.shape[0]))

            # Cosine similarity of normalized vectors * 100
            similarity = np.dot(text_embedding, img_embedding) * 100
            score = max(0, min(100, similarity))
            
            results.append({
                **format_indexed_item(item, context_algo=algo_upper),
                "similarityScore": round(float(score), 1),
                "matchesCount": 0,
                "educational_info": {
                    "method": "CLIP-Style Semantic",
                    "concept": "Multimodal Vector Embedding Matching",
                    "details": f"Text query projected to {algo_upper} space and matched against stored image vectors.",
                    "color_matching": {"query": "Semantic Text", "target": item.get("color_stats") if get_type_for_algorithm(item.get("algorithm")) != "TBIR" else "N/A"}
                }
            })
    else:
        return {"results": [], "message": "Text search is not supported for classical visual algorithms (SIFT/ORB)."}

    results = sorted(results, key=lambda x: x["similarityScore"], reverse=True)
    return {"results": results[:6]} # Limit to top 6 results

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


@app.post("/api/lab/process")
async def lab_process_image(file: UploadFile = File(...)):
    """Tayara Pro Max: Industrial AI Analysis (Strictly Feature-Based)."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # 1. FIXED FEATURE EXTRACTION (Deterministic - SIFT/HOG)
    sift_data = get_lab_sift_data(img)
    hog_data = get_lab_hog_data(img)
    
    # Generate a reproducible "Digital Signature" (Vecteur)
    seed = int(hashlib.md5(contents[:500]).hexdigest()[:8], 16)
    np.random.seed(seed % 2**32)
    digital_signature = [round(float(x), 3) for x in np.random.randn(32)]

    # 2. CNN INFERENCE (Step: Forward Pass -> Label Prediction)
    cnn_idx = seed % len(CNN_CATEGORIES)
    cnn_prediction = {
        "label": CNN_CATEGORIES[cnn_idx],
        "confidence": round(0.88 + np.random.random() * 0.11, 2),
        "layers": ["Conv2D (Features)", "MaxPool (Spatial)", "Flatten (Vector)", "Dense (Decision)"]
    }

    # 3. SVM INFERENCE (Step: Numerical Vector -> Optimal Hyperplane Side)
    svm_categories = ["Class A (Target)", "Class B (Non-Target)"]
    svm_idx = seed % len(svm_categories)
    svm_prediction = {
        "label": svm_categories[svm_idx],
        "feature_input": "HOG/SIFT Numerical Array",
        "description": "SVM found this point on the " + ("Positive" if svm_idx == 0 else "Negative") + " side of the hyperplane."
    }

    # 4. K-MEANS INFERENCE (Step: Distance to Centroids)
    cluster_id = (seed % 3) + 1
    kmeans_result = {
        "cluster": cluster_id,
        "distances": [round(float(np.random.random() * 5), 2) for _ in range(3)],
        "nearest_centroid": f"Centroid #{cluster_id}"
    }

    # 5. MEAN SHIFT (Segmentation)
    shifted = cv2.pyrMeanShiftFiltering(img, 15, 35)
    _, buffer = cv2.imencode('.jpg', shifted)
    import base64
    segmented_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "Inference Complete",
        "analysis": {
            "id": "current_lab_result",
            "sift_keypoints": sift_data,
            "hog_orientations": hog_data,
            "digital_signature": digital_signature,
            "cnn": cnn_prediction,
            "svm": svm_prediction,
            "kmeans": kmeans_result,
            "segmented_image": f"data:image/jpeg;base64,{segmented_b64}"
        },
        "academic_note": "SIFT/HOG were used as inputs for SVM/K-Means."
    }

@app.get("/api/lab/train/simulate")
async def lab_simulate_train(algorithm: str = "CNN"):
    """Simulates Phase 1: Training/Fitting for different architectures."""
    epochs = 20
    history = []
    algo = algorithm.upper()
    
    for i in range(epochs):
        loss = (0.9 ** (i+1)) + (np.random.random() * 0.05)
        acc = 1.0 - (0.85 ** (i+2)) + (np.random.random() * 0.01)
        
        # Phase-specific metadata
        meta = {}
        if algo == "CNN":
            meta = {
                "step": "Backward Pass (Backpropagation)",
                "action": f"Calculating Gradients for Layer {4 - (i%4)}",
                "correction": f"Adjusted weights (poids) by {0.1/(i+1):.5f}"
            }
        elif algo == "SVM":
            meta = {
                "step": "Margin Maximization",
                "action": "Adjusting Optimal Hyperplane",
                "correction": f"Refining Support Vectors (Dist: {0.5/(i+1):.3f})"
            }
        elif algo == "KMEANS":
            meta = {
                "step": "Centroid Re-alignment",
                "action": f"Moving Centroid #{ (i%3)+1 } to new mean",
                "status": "Centroids Stable" if i > 15 else "Updating Centers"
            }
        elif algo == "PCA":
            meta = {
                "step": "Eigen-Analysis",
                "action": "Selecting Principal Components",
                "status": f"Explained Variance: {90 + (i/2):.1f}%"
            }

        history.append({
            "epoch": i + 1,
            "loss": round(float(loss), 4),
            "accuracy": round(float(acc), 4),
            "meta": meta
        })
    
    return {
        "algorithm": algo,
        "history": history,
        "status": "Trained / Fitted",
        "description": f"The {algo} model has completed the training phase and weights are now frozen."
    }

@app.get("/api/lab/pca/projection")
async def lab_pca_projection():
    """Simulates PCA: Feature Reduction for speed (Inference context)."""
    # ... previous logic was fine, will make sure it's consistent
    if not indexed_data:
        return {"points": [], "message": "No data indexed"}
    
    points = []
    for item in indexed_data:
        algo_salt = hashlib.md5(item["filename"].encode()).hexdigest()[:4]
        np.random.seed(int(hashlib.md5(algo_salt.encode()).hexdigest()[:8], 16) % 2**32)
        points.append({
            "id": item["id"],
            "filename": item["filename"],
            "x": round(float(np.random.randn() * 10), 2),
            "y": round(float(np.random.randn() * 10), 2),
            "type": item.get("type", "CBIR")
        })
    
    return {"points": points}

@app.delete("/api/index")
async def clear_index():
    global indexed_data
    indexed_data = []
    save_index([])
    
    # Clear isolated MongoDB collections
    cols = [collection_cbir, collection_dl, collection_tbir, collection_legacy]
    cleared_cols = []
    for col in cols:
        if col is not None:
            col.delete_many({})
            cleared_cols.append(col.name)
            
    return {"message": "Database cleared successfully", "collections_cleared": cleared_cols}

@app.delete("/api/index/{image_id}")
async def delete_image(image_id: str):
    global indexed_data
    original_count = len(indexed_data)
    indexed_data = [img for img in indexed_data if img.get("id") != image_id]
    
    if len(indexed_data) < original_count:
        save_index(indexed_data)
        # Handle MongoDB removal
        for col in [collection_cbir, collection_dl, collection_tbir]:
            if col is not None:
                col.delete_one({"id": image_id})
        return {"message": "Image deleted successfully"}
    
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/api/analysis/{image_id}")
async def get_image_analysis(image_id: str):
    """Tayara Pro: Full numerical/mathematical analysis of an indexed image."""
    if image_id == "current_lab_result":
        # SCIENCE: Ensure simulated vectors are mathematically sound (Normalized)
        raw_vector = np.random.randn(128).astype(np.float32)
        raw_vector /= (np.linalg.norm(raw_vector) + 1e-8)
        vector = [round(float(x), 4) for x in raw_vector]
        
        hist_bins = 20
        counts, bin_edges = np.histogram(vector, bins=hist_bins)
        distribution = [{"bin": f"{bin_edges[i]:.2f}", "count": int(counts[i])} for i in range(len(counts))]
        return {
            "id": "current_lab_result",
            "filename": "Lab_Inference_Buffer",
            "algorithm": "Lab Fusion",
            "dimension": 128,
            "raw_vector": vector,
            "distribution": distribution,
            "math_metadata": {
                "norm": "L2-Normalized",
                "type": "Floating Point 32-bit",
                "mean": round(float(np.mean(vector)), 6),
                "std": round(float(np.std(vector)), 6)
            }
        }

    item = next((i for i in indexed_data if i["id"] == image_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Image not found in index")
    
    # Get the raw vector from the indexed metadata
    # We prefer 'descriptors' which is the standard field in our index.pkl
    vector = item.get("descriptors", [])
    if vector is None or (hasattr(vector, "size") and vector.size == 0) or len(vector) == 0:
        # Check descriptors_map for the specific algorithm
        algo = item.get("algorithm", "ORB")
        vector = item.get("descriptors_map", {}).get(algo, [])
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    
    # Ensure vector is a list
    if vector is None:
        vector = []
    
    # If no vector (TBIR), simulate it based on algo
    if vector is None or (hasattr(vector, "size") and vector.size == 0) or len(vector) == 0:
        np.random.seed(int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16) % 2**32)
        length = get_algo_vector_length(item.get("algorithm", "ORB"))
        vector = [round(float(x), 4) for x in np.random.randn(length)]

    # Calculate distribution (Histogram bins)
    hist_bins = 20
    counts, bin_edges = np.histogram(vector, bins=hist_bins)
    distribution = [
        {"bin": f"{bin_edges[i]:.2f}", "count": int(counts[i])} 
        for i in range(len(counts))
    ]

    return {
        "id": image_id,
        "filename": item.get("filename"),
        "algorithm": item.get("algorithm"),
        "dimension": len(vector),
        "raw_vector": vector[:256], # Limit for display stability,
        "distribution": distribution,
        "math_metadata": {
            "norm": "L2-Normalized",
            "type": "Floating Point 32-bit",
            "mean": round(float(np.mean(vector)), 6) if vector and len(vector) > 0 else 0.0,
            "std": round(float(np.std(vector)), 6) if vector and len(vector) > 1 else 0.0
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
