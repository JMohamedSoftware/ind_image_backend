import os, pickle

INDEX_FILE = "index.pkl"
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    print("LOCAL BACKUP: %d items found" % len(data))
    for i, d in enumerate(data[:5]):
        fn = d.get("filename", "?")
        algo = d.get("algorithm", "?")
        algos = list(d.get("descriptors_map", {}).keys())
        print("  [%d] %s | algo=%s | has_algos=%s" % (i, fn, algo, ",".join(algos)))
else:
    print("NO LOCAL BACKUP FOUND at", INDEX_FILE)
