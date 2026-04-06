import os
import pymongo
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.environ.get("MONGO_URL")

def migrate():
    if not MONGO_URL:
        print("❌ MONGO_URL not found in environment.")
        return

    try:
        client = pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client["test"]
        
        legacy_col = db["image_index"]
        col_cbir = db["image_index_content"]
        col_dl = db["image_index_deep"]
        col_tbir = db["image_index_text"]
        
        print(f"✅ Connected to MongoDB. Starting migration...")
        
        # 1. Fetch all from legacy
        docs = list(legacy_col.find({}))
        if not docs:
            print("📭 No documents found in legacy 'image_index' collection.")
            return

        print(f"📦 Found {len(docs)} documents in legacy collection.")
        
        count_cbir = 0
        count_dl = 0
        count_tbir = 0
        
        for doc in docs:
            # Determine target type
            target_type = doc.get("type", "CBIR")
            
            if target_type == "DL":
                target_col = col_dl
                count_dl += 1
            elif target_type == "TBIR":
                target_col = col_tbir
                count_tbir += 1
            else:
                target_col = col_cbir
                count_cbir += 1
            
            # Upsert into new collection
            # Use hash as identifier
            target_col.replace_one({"hash": doc["hash"]}, doc, upsert=True)
        
        print("-" * 30)
        print(f"✅ Migration Complete!")
        print(f"🔹 CBIR (Content): {count_cbir}")
        print(f"🔹 DL (Deep)     : {count_dl}")
        print(f"🔹 TBIR (Text)   : {count_tbir}")
        print("-" * 30)
        print("💡 Legacy collection 'image_index' was left intact as a backup.")
        
    except Exception as e:
        print(f"❌ DATABASE ERROR: {e}")

if __name__ == "__main__":
    migrate()
