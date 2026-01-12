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
from summarize_llm import SummarizeLLM
from certificate_of_origin_llm_extractor import CertificateOfOriginLLMExtractor
from email_pdf_merger_uploader import merge_pdfs_unique_and_upload
from mongo_trade_finance_store import store_trade_finance_result


load_dotenv()


bucket_name = "yc-retails-invoice"
s3_folder = "uploads_trade_finance/"
local_working_folder = "merged_output/"


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
    # --------------------------------
    # Step 1: Fetch unread email attachments
    # --------------------------------
    mail_data = fetch_unread_mbd_emirates_attachments()
    attachment_files = mail_data.get("files", [])

    print(f"📂 Processing {len(attachment_files)} attachment(s)")

    if not attachment_files:
        print("⚠️ No attachments found. Exiting.")
        return {}

    # --------------------------------
    # Step 2: Initialize classifier
    # --------------------------------
    classifier = DocumentTypeClassifier()

    # --------------------------------
    # Step 3: FINAL OUTPUT CONTAINER
    # --------------------------------
    final_llm_results = []

    # --------------------------------
    # Step 4: Process each attachment
    # --------------------------------
    for file_path in attachment_files:
        print(f"\n📄 Processing file: {file_path}")

        normalized_doc = run_textract_local(file_path)

        if not normalized_doc:
            print("⚠️ Skipping empty Textract result")
            continue

        doc_type = classifier.classify(normalized_doc)
        print("📌 Document Type:", doc_type)

        extracted_data = None

        if doc_type == "INVOICE":
            extracted_data = InvoiceLLMExtractor().extract(normalized_doc)

        elif doc_type == "COURIER_DISPATCH_ADVICE":
            extracted_data = CourierDispatchAdviceLLMExtractor().extract(normalized_doc)

        elif doc_type == "AIR_WAYBILL":
            extracted_data = AirWaybillLLMExtractor().extract(normalized_doc)

        elif doc_type == "LETTER_OF_CREDIT":
            extracted_data = LetterOfCreditLLMExtractor().extract(normalized_doc)

        elif doc_type == "CERTIFICATE_OF_ORIGIN":
            extracted_data = CertificateOfOriginLLMExtractor().extract(normalized_doc)

        else:
            print("ℹ️ No extractor configured for this document type")

        final_llm_results.append({
            "file_name": os.path.basename(file_path),
            "doc_type": doc_type,
            "extracted_data": extracted_data
        })

    # --------------------------------
    # Step 5: Summarize (LC vs Docs)
    # --------------------------------
    print("\n🧾 Running Trade Finance Summary LLM...")
    summarized_data = SummarizeLLM().extract(final_llm_results)

    # # Convert summary dict → readable paragraph text for PDF
    # summary_text = (
    #     f"Overall Status: {summarized_data.get('overall_status')}\n\n"
    #     f"Summary:\n{summarized_data.get('summary')}\n\n"
    #     f"Missing Documents:\n"
    #     f"{', '.join(summarized_data.get('missing_documents', [])) or 'None'}\n\n"
    #     f"Detailed Findings:\n"
    # )

    # for item in summarized_data.get("detailed_findings", []):
    #     summary_text += f"- {item.get('issue_type')}: {item.get('description')}\n"

    # --------------------------------
    # Step 6: Merge Email Summary + PDFs → Upload S3
    # --------------------------------
    print("\n📦 Creating merged PDF & uploading to S3...")

    merge_result = merge_pdfs_unique_and_upload(
        attachments=attachment_files,
        folder_path=local_working_folder,
        bucket_name=bucket_name,
        s3_folder=s3_folder,
        aws_access_key=AWS_ACCESS_KEY,
        aws_secret_key=AWS_SECRET_KEY,
        aws_region=REGION
    )

    print("\n✅ MERGED PDF RESULT")
    print(merge_result)
    
    mongo_id = store_trade_finance_result(
    extracted_results=summarized_data,
    object_url=merge_result["object_url"],
    filename=merge_result["filename"],
    original_s3_file=merge_result["s3_key"],
    email_text="Email subject: MBD Emirates"
)
    print("✅ Mongo Document ID:", mongo_id)

    # --------------------------------
    # Step 7: Final return object
    # --------------------------------
    return {
        "documents_extracted": final_llm_results,
        "summary": summarized_data,
        "merged_pdf": merge_result
    }


final_results = main()
print("\n🎯 FINAL PIPELINE OUTPUT")
print(final_results)
