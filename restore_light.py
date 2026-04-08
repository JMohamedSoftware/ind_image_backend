import pickle
import requests
import time

API_BASE = "https://ind-image-backend.onrender.com"
backup_file = 'index_backup.pkl'

with open(backup_file, 'rb') as f:
    data = pickle.load(f)

print(f"Starting LIGHT restoration of {len(data)} items to {API_BASE}...")

success = 0
failed = 0

for i, item in enumerate(data):
    url = item.get('url')
    algo = item.get('algorithm', 'ORB')
    desc = item.get('description', '')
    
    if not url:
        print(f"[{i}] Skipping {item.get('filename')} (No URL)")
        continue
    
    print(f"[{i}] Restoring {item.get('filename')} via {algo} (Lightweight)...")
    try:
        payload = {
            "url": url,
            "algorithm": algo,
            "description": desc,
            "multi_extract": False # FASTER restore
        }
        r = requests.post(f"{API_BASE}/api/index", data=payload, timeout=20)
        if r.status_code == 200:
            print(f"  Result: Success ✅")
            success += 1
        else:
            print(f"  Result: Failed ❌ ({r.status_code})")
            failed += 1
    except Exception as e:
        print(f"  Result: Error ⚠️: {e}")
        failed += 1
    
    time.sleep(0.5)

print("-" * 30)
print(f"Light Restoration complete: {success} Success, {failed} Failed.")
