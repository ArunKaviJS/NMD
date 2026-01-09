import os
import time
import traceback
from typing import Dict, Any
from trp import Document  # TRP library
import boto3
from dotenv import load_dotenv
from document_type_classifier import DocumentTypeClassifier
from invoice_llm_extractor import InvoiceLLMExtractor
from courier_dispatch_advice import CourierDispatchAdviceLLMExtractor
from airway_bill_llm_extractor import AirWaybillLLMExtractor
from letter_of_credit_llm_extractor import LetterOfCreditLLMExtractor
from email_attachment_fetcher import fetch_unread_mbd_emirates_attachments




load_dotenv()

# Load AWS credentials from env
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
REGION = os.getenv("REGION", "ap-south-1")

# Initialize Textract client
textract_client = boto3.client(
    "textract",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)

# ===============================
# Normalize Textract JSON
# ===============================

def normalize_textract_response(textract_output: Dict[str, Any]) -> Dict[str, Any]:
    print("🔄 Normalizing Textract JSON using TRP (tables + lines)...")

    doc = Document(textract_output)
    normalized = {
        "tables": [],
        "lines": []
    }

    for page in doc.pages:
        # ---- TABLE TEXT ----
        for table in page.tables:
            for row in table.rows:
                row_text = " ".join(
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text
                )
                if row_text:
                    normalized["lines"].append(row_text)

        # ---- NON-TABLE LINES ----
        for line in page.lines:
            if line.text and line.text.strip():
                normalized["lines"].append(line.text.strip())

    print(
        f"✅ Normalization complete: "
        f"{len(normalized['tables'])} tables, "
        f"{len(normalized['lines'])} lines"
    )

    return normalized


# ===============================
# Run Textract on local file
# ===============================
def run_textract_local(file_path: str) -> Dict[str, Any]:
    """
    Run AWS Textract directly on a local file (PDF or image).
    Returns normalized data + raw textract JSON.
    """
    try:
        print(f"📄 Reading local file: {file_path}")
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Call Textract analyze_document (synchronous)
        print("📄 Calling AWS Textract analyze_document...")
        response = textract_client.analyze_document(
            Document={"Bytes": file_bytes},
            FeatureTypes=["TABLES"]
        )

        # Normalize
        normalized_data = normalize_textract_response(response)

        page_count = response.get("DocumentMetadata", {}).get("Pages", 0)
        final_output = {
            "page_count": page_count,
            "normalized_data": normalized_data,
            "raw_textract": response
        }

        print("✅ Textract local job succeeded")
        return normalized_data

    except Exception as e:
        print(f"❌ Textract error: {e}")
        traceback.print_exc()
        return {}

# ===============================
# Example usage
# ===============================
def main():
    
    
    

    # Step 2: Run Textract on attachment
   
    mail_data = fetch_unread_mbd_emirates_attachments()

    for file_path in mail_data["files"]:
        normalized_doc = run_textract_local(file_path)

    print(normalized_doc)
    
    if not normalized_doc:
        print("❌ Textract failed or returned empty result")
        return

    # Step 2: Initialize classifier
    classifier = DocumentTypeClassifier()

    # Step 3: Classify document
    doc_type = classifier.classify(normalized_doc)

    print("📄 DOCUMENT TYPE IDENTIFIED:", doc_type)
    
     # Step 2: If invoice → extract fields
    if doc_type == "INVOICE":
        invoice_extractor = InvoiceLLMExtractor()
        invoice_data = invoice_extractor.extract(normalized_doc)

        print("🧾 Extracted Invoice Fields:")
        print(invoice_data)
        
    elif doc_type == "COURIER_DISPATCH_ADVICE":
        extractor = CourierDispatchAdviceLLMExtractor()
        courier_data = extractor.extract(normalized_doc)

        print("📦 COURIER DISPATCH ADVICE DATA:")
        print(courier_data)
        
    elif doc_type == "AIR_WAYBILL":
        extractor = AirWaybillLLMExtractor()
        awb_data = extractor.extract(normalized_doc)

        print("✈️ AIR WAYBILL DATA:")
        print(awb_data)
        
    elif doc_type == "LETTER_OF_CREDIT":
        extractor = LetterOfCreditLLMExtractor()
        lc_data = extractor.extract(normalized_doc)

        print("🏦 LETTER OF CREDIT DATA:")
        print(lc_data)

        
        
    else:
        print("ℹ️ No extractor configured for this document type yet")


if __name__ == "__main__":
    main()