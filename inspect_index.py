import pickle
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index.pkl")

if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as f:
        try:
            data = pickle.load(f)
            print(f"Total items: {len(data)}")
            
            types_count = {}
            for item in data:
                t = item.get('type', 'CBIR (missing field)')
                types_count[t] = types_count.get(t, 0) + 1
            
            print("\nCounts by Type:")
            for t, count in types_count.items():
                print(f"  {t}: {count}")
                
            if len(data) > 0:
                print("\nExample Item Keys:", data[0].keys())
                print("Example 'type':", data[0].get('type'))
                print("Example 'algorithm':", data[0].get('algorithm'))
                print("Example 'description':", data[0].get('description'))
                # Check for descriptors_map
                if 'descriptors_map' in data[0]:
                    print("Example 'descriptors_map' keys:", list(data[0]['descriptors_map'].keys()))
                else:
                    print("Warning: 'descriptors_map' missing in first item.")
        except Exception as e:
            print(f"Error loading index.pkl: {e}")
else:
    print("index.pkl not found.")
