"""
show_index_from_api.py
======================
يجيب قائمة كل الصور المفهرسة من الـ API ويعرضها حسب الخوارزمية
"""
import urllib.request
import json

# ────────────────────────────────────────────────────────
# 🔧 غيّر هذا إذا عندك URL مختلف
API_URL = "https://ind-image-backend.onrender.com"
# ────────────────────────────────────────────────────────

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())

print(f"🔗 Fetching from: {API_URL}\n")

# ─── Health Check ─────────────────────────────────────
try:
    health = fetch(f"{API_URL}/health")
    total  = health.get("indexed_count", "?")
    mongo  = health.get("mongo_connected", False)
    print(f"✅ Server OK  |  Total indexed: {total}  |  MongoDB: {'✅' if mongo else '❌'}\n")
except Exception as e:
    print(f"❌ Can't reach server: {e}")
    exit(1)

# ─── Fetch All Images ─────────────────────────────────
try:
    all_items = fetch(f"{API_URL}/api/index/list")
except Exception as e:
    print(f"❌ Failed to fetch /api/index/list: {e}")
    exit(1)

if not all_items:
    print("📭 No indexed images found.")
    exit(0)

# ─── Group by Algorithm ───────────────────────────────
groups = {}
for item in all_items:
    algo = item.get("algorithm", "UNKNOWN")
    groups.setdefault(algo, []).append(item)

# ─── Display ──────────────────────────────────────────
print("=" * 65)
print(f"  📊 INDEXED IMAGES BY ALGORITHM  (Total: {len(all_items)})")
print("=" * 65)

for algo in sorted(groups.keys()):
    items = groups[algo]
    img_type = items[0].get("type", "?")
    print(f"\n┌─ 🔷 {algo}  [{img_type}]  — {len(items)} image(s)")
    print(f"│")
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        prefix  = "└──" if is_last else "├──"
        filename = item.get("filename", "?")
        desc     = item.get("description") or "—"
        kp       = item.get("keypoint_count", 0)
        color    = item.get("color_stats") or {}
        url      = item.get("url") or "—"

        print(f"│ {prefix} 🖼️  {filename}")
        print(f"│      📝 Description : {str(desc)[:60]}")
        print(f"│      🔑 Keypoints   : {kp}")
        if color:
            r = color.get('r', '?')
            g = color.get('g', '?')
            b = color.get('b', '?')
            print(f"│      🎨 Color (RGB%) : R={r}  G={g}  B={b}")
        print(f"│      🔗 URL         : {str(url)[:55]}")
        if not is_last:
            print(f"│")

print("\n" + "=" * 65)
print(f"✅ Done. {len(all_items)} images across {len(groups)} algorithm(s).")
