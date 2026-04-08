"""
RESTORE SCRIPT: Copies index_backup.pkl -> index.pkl 
then pushes all data back to MongoDB via the local API.
"""
import pickle, shutil, os, requests, time

BASE = "http://127.0.0.1:8010"
BACKUP = "index_backup.pkl"
INDEX  = "index.pkl"

# 1. Read backup
with open(BACKUP, "rb") as f:
    data = pickle.load(f)
print("Backup contains %d items." % len(data))

# 2. Copy backup -> index.pkl  (so the server sees it on next startup)
shutil.copy2(BACKUP, INDEX)
print("Copied backup -> index.pkl")

# 3. Re-index each item via the running API (pushes back to MongoDB)
print("\nRe-indexing all items to MongoDB...")
ok = 0
fail = 0
for i, item in enumerate(data):
    url = item.get("url") or item.get("filename")
    algo = item.get("algorithm", "ORB")
    desc = item.get("description") or item.get("cnn_label") or ""
    
    if not url:
        print("[%d] SKIP - no url" % i)
        continue
    
    try:
        r = requests.post(
            BASE + "/api/index",
            data={"url": url, "algorithm": algo, "description": desc},
            timeout=30
        )
        if r.status_code == 200:
            ok += 1
            fn = item.get("filename","?")[:40]
            print("[%d] OK  %s | algo=%s" % (i, fn, algo))
        else:
            fail += 1
            print("[%d] FAIL %s | status=%d" % (i, url[:40], r.status_code))
    except Exception as e:
        fail += 1
        print("[%d] ERR %s" % (i, e))
    
    time.sleep(0.3)

print("\nDone. Restored: %d OK, %d FAIL" % (ok, fail))
