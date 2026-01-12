import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


class DocumentTypeClassifier:
    """
    Classifies Trade Finance documents into one of:
    - INVOICE
    - AIR_WAYBILL
    - COURIER_DISPATCH_ADVICE
    - LETTER_OF_CREDIT
    - CERTIFICATE_OF_ORIGIN
    """

    ALLOWED_TYPES = [
        "INVOICE",
        "AIR_WAYBILL",
        "COURIER_DISPATCH_ADVICE",
        "LETTER_OF_CREDIT",
        "CERTIFICATE_OF_ORIGIN",
    ]

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        if not self.deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT is not set")

    def classify(self, document: dict) -> str:
        """
        document format:
        {
            "tables": [...],
            "lines": [...]
        }

        returns:
        INVOICE | AIR_WAYBILL | COURIER_DISPATCH_ADVICE |
        LETTER_OF_CREDIT | CERTIFICATE_OF_ORIGIN
        """

        system_prompt = (
            "You are a Trade Finance document classifier used by a bank.\n\n"
            "Classify the document into EXACTLY ONE of:\n"
            "INVOICE, AIR_WAYBILL, COURIER_DISPATCH_ADVICE, "
            "LETTER_OF_CREDIT, CERTIFICATE_OF_ORIGIN.\n\n"

            "STRICT IDENTIFICATION RULES:\n\n"

            "1. CERTIFICATE_OF_ORIGIN:\n"
            "- Explicitly contains the title 'CERTIFICATE OF ORIGIN'\n"
            "- Mentions Chamber of Commerce or issuing authority\n"
            "- Contains declarations by Chamber and Exporter\n"
            "- Mentions country of origin\n"
            "- Often includes LC number and invoice reference\n"
            "- NOT a transport contract and NOT a billing document\n\n"

            "2. AIR_WAYBILL:\n"
            "- Issued by an airline or air cargo carrier\n"
            "- Mentions Air Waybill or AWB\n"
            "- Contains flight number and airport routing\n"
            "- Serves as a transport contract\n\n"

            "3. COURIER_DISPATCH_ADVICE:\n"
            "- Issued by courier or express companies\n"
            "- Mentions courier, express, pickup, delivery\n"
            "- Contains HAWB or courier tracking number\n"
            "- Used for document or parcel dispatch\n\n"

            "4. INVOICE:\n"
            "- Contains pricing, total amount, currency\n"
            "- Seller and buyer commercial transaction\n"
            "- Commercial or Proforma Invoice\n\n"

            "5. LETTER_OF_CREDIT:\n"
            "- Issued by a bank\n"
            "- Contains LC terms, conditions, availability\n"
            "- References UCP, issuing bank, advising bank\n\n"

            "PRIORITY RULES:\n"
            "- If 'CERTIFICATE OF ORIGIN' appears → CERTIFICATE_OF_ORIGIN\n"
            "- Do NOT confuse Certificate of Origin with Air Waybill\n"
            "- Output ONLY the document type string\n"
        )

        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(document, indent=2)},
            ],
        )

        result = response.choices[0].message.content.strip()

        # Hard safety guard
        if result not in self.ALLOWED_TYPES:
            raise ValueError(f"Unexpected classification result: {result}")

        return result
