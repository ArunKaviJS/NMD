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
    """

    ALLOWED_TYPES = [
        "INVOICE",
        "AIR_WAYBILL",
        "COURIER_DISPATCH_ADVICE",
        "LETTER_OF_CREDIT",
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
        INVOICE | AIR_WAYBILL | COURIER_DISPATCH_ADVICE | LETTER_OF_CREDIT
        """

        system_prompt = (
                "You are a Trade Finance logistics document classifier.\n\n"
                "Classify the document into EXACTLY ONE of:\n"
                "INVOICE, AIR_WAYBILL, COURIER_DISPATCH_ADVICE, LETTER_OF_CREDIT.\n\n"
                "IMPORTANT DISTINCTION RULES:\n"
                "1. AIR_WAYBILL:\n"
                "- Issued by an airline or air cargo carrier\n"
                "- Contains flight number and airport routing\n"
                "- Mentions Air Waybill as a transport contract\n"
                "- Includes weight or valuation charges\n\n"
                "2. COURIER_DISPATCH_ADVICE:\n"
                "- Issued by courier or express companies\n"
                "- Contains HAWB number\n"
                "- Mentions Express, Economy, Pickup, Charges\n"
                "- Used for document or small parcel dispatch\n\n"
                "Rules:\n"
                "- If courier indicators exist → COURIER_DISPATCH_ADVICE\n"
                "- If airline transport indicators exist → AIR_WAYBILL\n"
                "- Output ONLY the document type string"
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

        # Safety check (hard guard)
        if result not in self.ALLOWED_TYPES:
            raise ValueError(f"Unexpected classification result: {result}")

        return result
