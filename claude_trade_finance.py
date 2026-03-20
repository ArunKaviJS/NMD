import os
import time
import traceback
import boto3
import json
import base64
from typing import Dict, Any
from dotenv import load_dotenv
from agent_and_subagents.document_type_classifier import DocumentTypeClassifier
from agent_and_subagents.invoice_llm_extractor import InvoiceLLMExtractor
from agent_and_subagents.BillOfExchangeLLMExtractor import BillOfExchangeLLMExtractor
from agent_and_subagents.Billoflading import BillOfLadingLLMExtractor
from agent_and_subagents.PackingListllmextractor import PackingListLLMExtractor
from agent_and_subagents.inspection_certificate import InspectionCertificateLLMExtractor
from agent_and_subagents.InsuranceCertificateLLMExtractor import InsuranceCertificateLLMExtractor
from agent_and_subagents.letter_of_credit_llm_extractor import LetterOfCreditLLMExtractor
from agent_and_subagents.summarize_llm import SummarizeLLM
from agent_and_subagents.certificate_of_origin_llm_extractor import CertificateOfOriginLLMExtractor
from email_and_mongo.email_pdf_merger_uploader import merge_pdfs_unique_and_upload
from email_and_mongo.mongo_trade_finance_store import store_trade_finance_result
from email_and_mongo.email_attachment_fetcher import fetch_unread_mbd_emirates_attachments
load_dotenv()




# Load AWS credentials from env
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
REGION = os.getenv("REGION", "ap-south-1")


# Initialize Bedrock client for Claude Sonnet OCR
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)


EXPECTED_DOCUMENT_TYPES = {
    "BILL_OF_EXCHANGE":       "Bill of Exchange",
    "BILL_OF_LADING":         "Bill of Lading",
    "CERTIFICATE_OF_ORIGIN":  "Certificate of Origin",
    "INVOICE":                "Commercial Invoice",
    "INSPECTION_CERTIFICATE": "Inspection Certificate",
    "INSURANCE_CERTIFICATE":  "Insurance Certificate",
    "LETTER_OF_CREDIT":       "Letter of Credit",
    "PACKING_LIST":           "Packing List",
}


bucket_name = "yc-retails-invoice"
s3_folder = "uploads_trade_finance/"
local_working_folder = "merged_output/"


def run_azure_ocr_local(file_path: str) -> str:
    """
    Run OCR using Claude Sonnet via AWS Bedrock.
    Sends the PDF/image as base64 and extracts full text content.
    """
    try:
        print(f"📄 Reading local file: {file_path}")

        # Read and encode file as base64
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        base64_data = base64.standard_b64encode(file_bytes).decode("utf-8")

        # Determine media type based on file extension
        ext = os.path.splitext(file_path)[1].lower()
        media_type_map = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(ext, "application/pdf")

        # Build the source block based on media type
        if media_type == "application/pdf":
            source_block = {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data
            }
            content_block = {
                "type": "document",
                "source": source_block
            }
        else:
            source_block = {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data
            }
            content_block = {
                "type": "image",
                "source": source_block
            }

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        content_block,
                        {
                            "type": "text",
                            "text": (
                                "You are a document OCR engine. "
                                "Extract and return ALL text content from this document exactly as it appears. "
                                "Preserve the layout, tables, labels, values, line breaks, and structure as closely as possible. "
                                "Do not summarize, interpret, or omit any content. "
                                "Output only the raw extracted text with no additional commentary."
                            )
                        }
                    ]
                }
            ]
        }

        print(f"🤖 Sending to Claude Sonnet via Bedrock...")

        response = bedrock_client.invoke_model(
            modelId="global.anthropic.claude-sonnet-4-6",
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())
        raw_content = result["content"][0]["text"]

        print(f"✅ Claude OCR complete. Extracted {len(raw_content)} characters")
        print("raw_content", raw_content)

        return raw_content

    except Exception as e:
        print(f"❌ Claude Bedrock OCR error: {e}")
        traceback.print_exc()
        return ""
#============
# Example usage
# ===============================
def main():
    # --------------------------------
    # Step 1: Fetch unread email attachments
    # --------------------------------
    mail_data = fetch_unread_mbd_emirates_attachments()
    #attachment_files = mail_data.get("files", [])
    attachment_files = []

    # ✅ It's a single dict now, not a list
    attachment_files = mail_data.get("files", [])
    email_subject = mail_data.get("email_subject", "NMD Emirates")

    print("Total attachments:", attachment_files)

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
    uploaded_doc_types = set()
    
    # --------------------------------
    # Step 5: collect unrecognised file names here
    # --------------------------------
    unexpected_files = []   


    for file_path in attachment_files:
        print(f"\n📄 Processing file: {file_path}")

        normalized_doc = run_azure_ocr_local(file_path)

        if not normalized_doc:
            print("⚠️ Skipping empty Textract result")
            continue

         # Classify — catch anything the classifier doesn't recognise
        try:
            doc_type = classifier.classify(normalized_doc)
        except ValueError as e:
            file_name = os.path.basename(file_path)
            print(f"⚠️ Unclassified document skipped: {file_name} — {e}")
            unexpected_files.append(file_name)
            continue  # skip extraction entirely for this file
        print("📌 Document Type:", doc_type)
        
        

        if doc_type in EXPECTED_DOCUMENT_TYPES:
            uploaded_doc_types.add(doc_type)

        extracted_data = None

        if doc_type == "INVOICE":
            extracted_data = InvoiceLLMExtractor().extract(normalized_doc)

        elif doc_type == "LETTER_OF_CREDIT":
            extracted_data = LetterOfCreditLLMExtractor().extract(normalized_doc)

        elif doc_type == "CERTIFICATE_OF_ORIGIN":
            extracted_data = CertificateOfOriginLLMExtractor().extract(normalized_doc)

        elif doc_type == "BILL_OF_EXCHANGE":
            extracted_data = BillOfExchangeLLMExtractor().extract(normalized_doc)

        elif doc_type == "BILL_OF_LADING":
            extracted_data = BillOfLadingLLMExtractor().extract(normalized_doc)

        elif doc_type == "INSPECTION_CERTIFICATE":
            extracted_data = InspectionCertificateLLMExtractor().extract(normalized_doc)

        elif doc_type == "INSURANCE_CERTIFICATE":
            extracted_data = InsuranceCertificateLLMExtractor().extract(normalized_doc)

        elif doc_type == "PACKING_LIST":
            extracted_data = PackingListLLMExtractor().extract(normalized_doc)

        else:
            print("ℹ️ No extractor configured for this document type")

        
        final_llm_results.append({
            "file_name": os.path.basename(file_path),
            "doc_type": doc_type,
            "extracted_data": extracted_data
        })
        
    missing_documents = [
    EXPECTED_DOCUMENT_TYPES[doc]
    for doc in EXPECTED_DOCUMENT_TYPES
    if doc not in uploaded_doc_types
    ]

    print('extracted_text',extracted_data)
    print('doctype*****',doc_type)
    print('final llm results',final_llm_results)
    # --------------------------------
    # Step 5: Summarize (LC vs Docs)
    # --------------------------------
    print("\n🧾 Running Trade Finance Summary LLM...")
    summarized_data = SummarizeLLM().extract({
            "documents": final_llm_results,
            "missing_documents": missing_documents
        })

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
    email_text="Email subject: NMD Emirates"
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


def run_live():
    print("🚀 Starting LIVE email processing service (poll every 5 seconds)...")

    try:
        while True:
            try:
                print("\n⏳ Checking for new emails...")
                result = main()
                print('Final Results',result)

                if not result or not result.get("documents_extracted"):
                    print("📭 No new attachments found")
                else:
                    print("📨 New documents processed successfully")

            except Exception as e:
                print("❌ Error during pipeline execution")
                traceback.print_exc()

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n🛑 Live service stopped by user (Ctrl+C)")


run_live()
