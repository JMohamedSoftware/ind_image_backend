import os, pickle, glob

search_root = r"C:\Users\User\Desktop\image search and indexing"
pkl_files = glob.glob(os.path.join(search_root, "**", "*.pkl"), recursive=True)

for p in pkl_files:
    size_kb = os.path.getsize(p) / 1024
    import datetime
    dt = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        count = len(data) if isinstance(data, list) else "not-list"
    except Exception as e:
        count = "ERR"
        data = []
    
    short = p.split("image search and indexing")[-1]
    print(short, "| %.0fKB | %s | items=%s" % (size_kb, dt, count))
    if isinstance(data, list):
        for item in data[:3]:
            fn = item.get("filename", "?")
            al = list(item.get("descriptors_map", {}).keys())
            print("   ->", fn[:40], "|", ",".join(al))
