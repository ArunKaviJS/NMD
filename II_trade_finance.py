import os
import time
import traceback
import boto3
from typing import Dict, Any
from dotenv import load_dotenv
from agent_and_subagents.document_type_classifier import DocumentTypeClassifier
from agent_and_subagents.invoice_llm_extractor import InvoiceLLMExtractor
from agent_and_subagents.airway_bill_llm_extractor import AirWaybillLLMExtractor
from agent_and_subagents.letter_of_credit_llm_extractor import LetterOfCreditLLMExtractor
from email_and_mongo.email_attachment_fetcher import fetch_unread_mbd_emirates_attachments
from agent_and_subagents.summarize_llm import SummarizeLLM
from agent_and_subagents.certificate_of_origin_llm_extractor import CertificateOfOriginLLMExtractor
from email_and_mongo.email_pdf_merger_uploader import merge_pdfs_unique_and_upload
from email_and_mongo.mongo_trade_finance_store import store_trade_finance_result
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
load_dotenv()


AZURE_ENDPOINT = os.getenv("AZURE_AI_SERVICES_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_AI_SERVICES_API_KEY")

client = DocumentIntelligenceClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_KEY)
)



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



EXPECTED_DOCUMENT_TYPES = {
    "AIR_WAYBILL": "Air Waybill",
    "CERTIFICATE_OF_ORIGIN": "Certificate of Origin",
    "LETTER_OF_CREDIT": "Letter of Credit",
    "INVOICE": "Commercial Invoice"
}


bucket_name = "yc-retails-invoice"
s3_folder = "uploads_trade_finance/"
local_working_folder = "merged_output/"




def run_azure_ocr_local(file_path: str) -> str:
    """
    Run Azure Document Intelligence (prebuilt-layout)
    and return FULL TEXT only.
    """

    try:
        print(f"📄 Reading local file: {file_path}")

        with open(file_path, "rb") as f:
            poller = client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=f
            )

        result = poller.result()
        raw_content=result.content
        print('raw_content',raw_content)
        
        text_lines = []

        # Extract text from pages
        for page in result.pages:
            for line in page.lines:
                if line.content.strip():
                    text_lines.append(line.content.strip())

        full_text = "\n".join(text_lines)

        print(f"✅ Azure OCR complete. Extracted {len(text_lines)} lines")

        return full_text

    except Exception as e:
        print(f"❌ Azure OCR error: {e}")
        traceback.print_exc()
        return ""
# ===============================
# Example usage
# ===============================
def main():
    # --------------------------------
    # Step 1: Fetch unread email attachments
    # --------------------------------
    mail_data = fetch_unread_mbd_emirates_attachments()
    #attachment_files = mail_data.get("files", [])
    attachment_files = []

    for email_data in mail_data:
        files = email_data.get("files", [])
        attachment_files.extend(files)

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


    for file_path in attachment_files:
        print(f"\n📄 Processing file: {file_path}")

        normalized_doc = run_azure_ocr_local(file_path)

        if not normalized_doc:
            print("⚠️ Skipping empty Textract result")
            continue

        doc_type = classifier.classify(normalized_doc)
        print("📌 Document Type:", doc_type)
        
        

        if doc_type in EXPECTED_DOCUMENT_TYPES:
            uploaded_doc_types.add(doc_type)

        extracted_data = None

        if doc_type == "INVOICE":
            extracted_data = InvoiceLLMExtractor().extract(normalized_doc)


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
        
    missing_documents = [
    EXPECTED_DOCUMENT_TYPES[doc]
    for doc in EXPECTED_DOCUMENT_TYPES
    if doc not in uploaded_doc_types
    ]


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
