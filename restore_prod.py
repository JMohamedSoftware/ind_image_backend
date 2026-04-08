import pickle
import requests
import time

API_BASE = "https://ind-image-backend.onrender.com"
backup_file = 'index_backup.pkl'

with open(backup_file, 'rb') as f:
    data = pickle.load(f)

print(f"Starting restoration of {len(data)} items to {API_BASE}...")

success = 0
failed = 0

for i, item in enumerate(data):
    url = item.get('url')
    algo = item.get('algorithm', 'ORB')
    desc = item.get('description', '')
    
    if not url:
        print(f"[{i}] Skipping {item.get('filename')} (No URL)")
        continue
    
    print(f"[{i}] Restoring {item.get('filename')} via {algo}...")
    try:
        payload = {
            "url": url,
            "algorithm": algo,
            "description": desc,
            "multi_extract": True # Re-extract everything for completeness
        }
        r = requests.post(f"{API_BASE}/api/index", data=payload, timeout=45)
        if r.status_code == 200:
            print(f"  Result: Success ✅")
            success += 1
        else:
            print(f"  Result: Failed ❌ ({r.status_code}): {r.text[:50]}")
            failed += 1
    except Exception as e:
        print(f"  Result: Error ⚠️: {e}")
        failed += 1
    
    # Small sleep to be nice to Render
    time.sleep(1)

print("-" * 30)
print(f"Restoration complete: {success} Success, {failed} Failed.")
