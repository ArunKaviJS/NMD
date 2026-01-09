import os
import imaplib
import email
import uuid
from email.header import decode_header
from datetime import datetime


def fetch_unread_mbd_emirates_attachments():
    """
    Fetch attachments from the latest UNREAD email
    whose subject contains 'MBD emirates' (case-insensitive).

    Downloads files into a unique folder.

    Returns:
        {
            "folder_path": str,
            "files": [file_path1, file_path2, ...]
        }
    """

    IMAP_SERVER = os.getenv("IMAP_SERVER")
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")

    if not all([IMAP_SERVER, EMAIL_USER, EMAIL_PASS]):
        raise ValueError("Missing IMAP environment variables")

    print("📧 Connecting to IMAP server...")
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")

    # Search only UNSEEN emails
    status, messages = mail.search(None, '(UNSEEN)')
    email_ids = messages[0].split()

    if not email_ids:
        raise Exception("No unread emails found")

    # Process newest first
    for email_id in reversed(email_ids):
        status, msg_data = mail.fetch(email_id, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        # Decode subject safely
        subject = ""
        for part, enc in decode_header(msg.get("Subject", "")):
            if isinstance(part, bytes):
                subject += part.decode(enc or "utf-8", errors="ignore")
            else:
                subject += part

        print(f"🔍 Checking subject: {subject}")

        if "mbd emirates" not in subject.lower():
            continue

        print(f"✅ Matched unread email: {subject}")

        # Create unique folder
        unique_folder = f"mail_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        os.makedirs(unique_folder, exist_ok=True)

        saved_files = []

        for part in msg.walk():
            content_disposition = part.get("Content-Disposition", "")
            if "attachment" in content_disposition.lower():
                filename = part.get_filename()
                if not filename:
                    continue

                decoded_name, _ = decode_header(filename)[0]
                if isinstance(decoded_name, bytes):
                    decoded_name = decoded_name.decode(errors="ignore")

                file_path = os.path.join(unique_folder, decoded_name)

                with open(file_path, "wb") as f:
                    f.write(part.get_payload(decode=True))

                saved_files.append(file_path)
                print(f"📎 Saved attachment: {file_path}")

        if not saved_files:
            raise Exception("Unread matched email found but no attachments")

        # Mark email as SEEN
        mail.store(email_id, '+FLAGS', '\\Seen')

        return {
            "folder_path": unique_folder,
            "files": saved_files
        }

    raise Exception("No unread email found with subject 'MBD emirates'")


# ===============================
# 🔎 RUN THIS FILE ALONE TO TEST
# ===============================
if __name__ == "__main__":
    result = fetch_unread_mbd_emirates_attachments()

    print("\n📂 DOWNLOAD SUMMARY")
    print("Folder:", result["folder_path"])
    for f in result["files"]:
        print(" -", f)
