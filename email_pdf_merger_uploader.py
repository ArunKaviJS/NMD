import os
import boto3
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from PyPDF2 import PdfMerger


def merge_pdfs_unique_and_upload(
    attachments: list,
    folder_path: str = "",
    bucket_name: str = "",
    s3_folder: str = "",
    aws_access_key: str = "",
    aws_secret_key: str = "",
    aws_region: str = "ap-south-1"
):
    """
    Merge PDF attachments into a uniquely named PDF
    and upload it to AWS S3.
    """

    import os
    from datetime import datetime
    from PyPDF2 import PdfMerger
    import boto3

    os.makedirs(folder_path, exist_ok=True)

    # -------------------------------
    # Generate Unique Filename
    # -------------------------------
    unique_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_filename = f"Merged_{unique_id}"

    merged_pdf_path = os.path.join(folder_path, f"{base_filename}.pdf")

    # -------------------------------
    # Merge Attachments
    # -------------------------------
    merger = PdfMerger()

    for file_path in attachments:
        if file_path.lower().endswith(".pdf") and os.path.exists(file_path):
            merger.append(file_path)

    merger.write(merged_pdf_path)
    merger.close()

    # -------------------------------
    # Upload to S3
    # -------------------------------
    s3_key = f"{s3_folder}{base_filename}.pdf"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

    s3.upload_file(merged_pdf_path, bucket_name, s3_key)

    object_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_key}"

    return {
        "local_pdf_path": merged_pdf_path,
        "s3_bucket": bucket_name,
        "s3_key": s3_key,
        "object_url": object_url,
        "filename": f"{base_filename}.pdf",
        "message": "PDFs merged and uploaded successfully"
    }
