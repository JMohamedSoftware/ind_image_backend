import pickle
import os

backup_file = 'index_backup.pkl'
if not os.path.exists(backup_file):
    print("Backup not found!")
    exit(1)

with open(backup_file, 'rb') as f:
    data = pickle.load(f)

print(f"Total items in backup: {len(data)}")
for i, item in enumerate(data):
    fn = item.get('filename', 'unknown')
    url = item.get('url', 'no-url')
    algo = item.get('algorithm', 'unknown')
    desc = item.get('description', '')
    print(f"[{i}] {fn} | {url[:50]} | {algo} | '{desc}'")
