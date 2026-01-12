import os
import json
from datetime import datetime, timezone
from bson import ObjectId
from pymongo import MongoClient
from dotenv import load_dotenv
import json


# -------------------------------------------------------
# ENV SETUP
# -------------------------------------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
FILE_DETAILS = os.getenv("FILE_DETAILS")  # collection name

if not all([MONGO_URI, DB_NAME, FILE_DETAILS]):
    raise ValueError("Mongo environment variables not set properly")

# -------------------------------------------------------
# MONGO CLIENT (REUSED)
# -------------------------------------------------------
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[FILE_DETAILS]


# -------------------------------------------------------
# NORMALIZER
# Ensures nested structures stored as REAL objects
# -------------------------------------------------------


def normalize_structured_data(summarized_data):
    """
    Accepts final LLM output and returns it AS-IS
    so it can be directly stored in MongoDB.

    Supported input:
    - dict (preferred)
    - JSON string
    """

    # If LLM returned JSON string → convert once
    if isinstance(summarized_data, str):
        summarized_data = json.loads(summarized_data)

    # Final safety check
    if not isinstance(summarized_data, dict):
        raise ValueError("summarized_data must be a dict or valid JSON string")

    # ✅ RETURN AS-IS (NO NORMALIZATION)
    return summarized_data

# -------------------------------------------------------
# MAIN STORE FUNCTION
# -------------------------------------------------------
def store_trade_finance_result(
    extracted_results: dict,
    object_url: str,
    filename: str,
    original_s3_file: str,
    email_text: str = ""
):
    """
    Stores final trade finance extracted results into MongoDB

    Parameters:
    - extracted_results : dict (final LLM output / summarized result)
    - object_url        : merged PDF S3 URL
    - filename          : merged PDF filename
    - original_s3_file  : original uploaded S3 path
    - email_text        : optional email body text
    """

    normalized_data = normalize_structured_data(
        extracted_results
    )

    document = {
        "_id": ObjectId(),

        # 🔐 System metadata
        "clusterId": ObjectId("6964a25d9576df2e993331b5"),
        "userId": ObjectId("6964a0a09576df2e9933315d"),
        "status": "1",
        "processingStatus": "Completed",

        # 📄 File metadata
        "fileName": filename,
        "originalFile": object_url,
        "originalS3File": original_s3_file,

        # 🧾 Extracted data
        "extractedValues": normalized_data,
        "updatedExtractedValues": normalized_data,

        # 💳 Credits (future)
        "credits": None,

        # ⏱ Audit
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    }

    collection.insert_one(document)

    return str(document["_id"])
