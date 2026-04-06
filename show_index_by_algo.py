"""
show_index_by_algo.py
=====================
يعرض كل الصور المفهرسة مقسّمة حسب الخوارزمية
من MongoDB (إذا موجود) أو من index.pkl المحلي
"""
import pickle
import os
import sys

# ─── Try MongoDB First ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    import pymongo
    load_dotenv()
    MONGO_URL = os.environ.get("MONGO_URL")
    data = []
    if MONGO_URL:
        client = pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client["test"]
        
        # Query all three isolated collections
        collections = [
            db["image_index_content"],
            db["image_index_deep"],
            db["image_index_text"]
        ]
        
        data = []
        for col in collections:
            docs = list(col.find({}))
            for doc in docs:
                doc.pop("_id", None)
            data.extend(docs)
            
        print(f"✅ Connected to MongoDB Atlas — {len(data)} items loaded from isolated collections\n")
    else:
        raise Exception("No MONGO_URL")
except Exception as e:
    print(f"⚠️  MongoDB not available ({e}), falling back to local index.pkl\n")
    data = []

# ─── Fallback to Local index.pkl ───────────────────────────────────────────
if not data:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_FILE = os.path.join(BASE_DIR, "index.pkl")
    if not os.path.exists(INDEX_FILE):
        print("❌ index.pkl not found and MongoDB unavailable. No data to show.")
        sys.exit(1)
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"📂 Loaded {len(data)} items from local index.pkl\n")

if not data:
    print("📭 No indexed images found.")
    sys.exit(0)

# ─── Group by Algorithm ────────────────────────────────────────────────────
groups = {}
for item in data:
    algo = item.get("algorithm", "UNKNOWN")
    groups.setdefault(algo, []).append(item)

# ─── Display ───────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  📊 INDEXED IMAGES BY ALGORITHM  (Total: {len(data)})")
print("=" * 60)

for algo in sorted(groups.keys()):
    items = groups[algo]
    img_type = items[0].get("type", "?")
    print(f"\n┌─ 🔷 {algo}  [{img_type}]  — {len(items)} image(s)")
    print(f"│")
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        prefix = "└──" if is_last else "├──"
        filename   = item.get("filename", "?")
        desc       = item.get("description") or "—"
        kp         = item.get("keypoint_count", 0)
        color      = item.get("color_stats", {})

        # Descriptor info
        d_map      = item.get("descriptors_map", {})
        algo_key   = list(d_map.keys())[0] if d_map else "?"
        d_val      = d_map.get(algo_key)
        if d_val is not None:
            try:
                import numpy as np
                arr = np.array(d_val)
                desc_info = f"shape={arr.shape}"
            except Exception:
                desc_info = f"len={len(d_val)}"
        else:
            desc_info = "None"

        print(f"│ {prefix} 🖼️  {filename}")
        print(f"│      Description : {desc[:60]}")
        print(f"│      Keypoints   : {kp}")
        if color:
            r = color.get('r', '?')
            g = color.get('g', '?')
            b = color.get('b', '?')
            print(f"│      Color (RGB%) : R={r}  G={g}  B={b}")
        print(f"│      Descriptor  : [{algo_key}] {desc_info}")
        if not is_last:
            print(f"│")

print("\n" + "=" * 60)
print("✅ Done.")
