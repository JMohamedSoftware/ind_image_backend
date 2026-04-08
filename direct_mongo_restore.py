"""
DIRECT MONGO RESTORE: Reads index_backup.pkl and writes directly to MongoDB Atlas.
Run this locally - it bypasses Render entirely.
"""
import pickle, os, hashlib
import numpy as np

# --- Read the MONGO_URL from the .env file ---
from dotenv import load_dotenv
load_dotenv()

MONGO_URL = os.environ.get("MONGO_URL")
if not MONGO_URL:
    print("ERROR: No MONGO_URL in .env file!")
    exit(1)

import pymongo
client = pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=10000)
client.admin.command("ping")
print("Connected to MongoDB Atlas!")

db = client["test"]
collection_cbir = db["image_index_content"]
collection_dl   = db["image_index_deep"]
collection_tbir = db["image_index_text"]

CBIR_TYPE = ["SIFT", "HOG", "ORB", "AKAZE", "CANNY"]
ML_TYPE   = ["SVM", "K-MEANS", "PCA", "PFCM", "MEANSHIFT"]
DL_TYPE   = ["CNN", "CLIP", "BACKPROPAGATION"]

def get_type(algo):
    a = algo.upper()
    if a in DL_TYPE: return "DL"
    if a in ML_TYPE: return "ML"
    if a.startswith("TBIR"): return "TBIR"
    return "CBIR"

def get_collection(tp):
    if tp == "DL" or tp == "ML": return collection_dl
    if tp == "TBIR": return collection_tbir
    return collection_cbir

# Read backup
with open("index_backup.pkl", "rb") as f:
    data = pickle.load(f)

print("Backup items to restore: %d" % len(data))

ok = 0
fail = 0
for i, item in enumerate(data):
    try:
        # Serialize numpy arrays to lists
        new_item = dict(item)
        if "descriptors_map" in new_item:
            dm = {}
            for k, v in new_item["descriptors_map"].items():
                if isinstance(v, np.ndarray):
                    dm[k] = v.tolist()
                elif isinstance(v, list):
                    dm[k] = v
                else:
                    dm[k] = v
            new_item["descriptors_map"] = dm

        # Remove _id if exists
        new_item.pop("_id", None)

        # Route to correct collection
        tp = get_type(new_item.get("algorithm", "ORB"))
        col = get_collection(tp)

        img_hash = new_item.get("hash", new_item.get("filename", ""))
        col.replace_one({"hash": img_hash}, new_item, upsert=True)

        fn = new_item.get("filename", "?")[:40]
        print("[%d] OK  %s | type=%s | algos=%s" % (i, fn, tp, ",".join(new_item.get("descriptors_map", {}).keys())))
        ok += 1
    except Exception as e:
        print("[%d] FAIL: %s" % (i, e))
        fail += 1

print("\nRestore complete: %d OK, %d FAIL" % (ok, fail))

# Verify counts
for name, col in [("CBIR", collection_cbir), ("DL/ML", collection_dl), ("TBIR", collection_tbir)]:
    cnt = col.count_documents({})
    print("Collection %s -> %d documents" % (name, cnt))
