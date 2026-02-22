import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import pickle

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

def get_detector(algorithm: str):
    algo = algorithm.upper() if algorithm else "ORB"
    if algo == "SIFT":
        return cv2.SIFT_create(nfeatures=500)
    elif algo == "AKAZE":
        return cv2.AKAZE_create()
    else:  # Default: ORB
        return cv2.ORB_create(nfeatures=1000)

def get_matcher(algorithm: str):
    algo = algorithm.upper() if algorithm else "ORB"
    if algo == "SIFT":
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif algo == "AKAZE":
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:  # ORB
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def process_image_content(contents, algorithm="ORB"):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    detector = get_detector(algorithm)
    kp, des = detector.detectAndCompute(img, None)
    # Return keypoint coordinates for visualization
    kp_coords = [(int(k.pt[0]), int(k.pt[1])) for k in kp] if kp else []
    return kp_coords, des

@app.get("/")
async def root():
    return {"message": "Visual Search API — Multi-Algorithm Edition"}

@app.get("/api/stats")
async def stats():
    return {"indexed_count": len(indexed_data)}

@app.post("/api/index")
async def index_image(
    file: UploadFile = File(None),
    url: str = Form(None),
    filename: str = Form(None),
    algorithm: str = Form("ORB")
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

    result = process_image_content(contents, algorithm)
    if result is None:
        return {"status": "error", "message": "Could not decode image"}

    kp_coords, des = result

    if des is not None and len(des) > 0:
        indexed_data.append({
            "filename": target_filename,
            "url": url if url else None,
            "descriptors": des,
            "algorithm": algorithm.upper(),
            "keypoint_count": len(kp_coords),
        })
        save_index(indexed_data)
        return {
            "status": "success",
            "filename": target_filename,
            "keypoints": len(kp_coords),
            "algorithm": algorithm.upper(),
            "keypoint_coords": kp_coords[:50],  # Return first 50 for visualization
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
        return {"results": [], "message": "No images indexed yet. Please upload images first."}

    matcher = get_matcher(algorithm)
    results = []

    for item in indexed_data:
        des_indexed = item["descriptors"]
        if des_indexed is None or len(des_indexed) == 0:
            continue
        try:
            matches = matcher.match(des_query, des_indexed)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 60]
            score = (len(good_matches) / max(len(kp_query), 1)) * 100
            score = min(score * 2.5, 100)

            results.append({
                "filename": item["filename"],
                "url": item.get("url"),
                "similarityScore": round(score, 1),
                "matchesCount": len(good_matches),
                "algorithm": item.get("algorithm", "ORB"),
            })
        except Exception:
            continue

    results = sorted(results, key=lambda x: x["similarityScore"], reverse=True)
    return {"results": results[:6], "query_keypoints": len(kp_query)}


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
