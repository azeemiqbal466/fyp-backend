from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
import os

load_dotenv()

DB_NAME = "mydatabase"
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in environment variables")

# Singleton objects
_client = None
_db = None

def get_db_connection():
    global _client, _db
    try:
        if _db is None:
            _client = MongoClient(MONGO_URI)
            _db = _client[DB_NAME]
        return _db
    except PyMongoError as e:
        raise RuntimeError(f"Database connection failed: {e}")
