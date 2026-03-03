import os
import json
import re
from openai import AzureOpenAI


class AirWaybillLLMExtractor:
    """
    Extract mandatory fields from AIR WAYBILL (AWB)
    Used by banks to control cargo release
    """

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    def _safe_json_parse(self, text: str) -> dict:
        """
        Safely parse JSON from LLM output
        """
        if not text:
            raise ValueError("Empty LLM response")

        text = text.strip()
        text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found:\n{text}")

        return json.loads(match.group(0))

    def extract(self, normalized_doc):
        """
        Extract AIR WAYBILL mandatory fields
        """

        system_prompt = """
        You are a Trade Finance Air Waybill (AWB) Extraction Engine.

        Document Type: AIR WAYBILL (AWB)

        Rules:
        - Extract ONLY information explicitly stated in the document
        - DO NOT infer, calculate, or assume values
        - Preserve wording exactly as shown
        - If a field is not clearly mentioned, return null
        - Output MUST be valid JSON only
        - Do NOT add extra fields
        - Do NOT rename fields
        - Do NOT explain anything

        Field Mapping Rules:
        - AWB Number must be the 11-digit Air Waybill number (e.g., 176-XXXXXXXX or 176 XXXXXXXX)
        - AWB Date = Execution date / Issued date (NOT flight date unless explicitly stated as AWB date)
        - Shipment Date = Flight Date / Date of Departure
        - Beneficiary = Shipper (if no separate beneficiary mentioned)
        - Applicant/Consignee = Consignee
        - If multiple shipment dates appear, extract only the main flight date
        - Goods Description must be taken from “Nature and Quantity of Goods” or similar section

        Required JSON Schema:

        {
        "awb_number": null,
        "awb_date": null,
        "shipper": null,
        "consignee": null,
        "shipment_date": null,
        "beneficiary": null,
        "applicant_consignee": null,
        "goods_description": null
        }
"""

        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(normalized_doc)},
            ],
        )

        raw_output = response.choices[0].message.content

        # Debug (disable in production)
        print("\n✈️ RAW AWB LLM OUTPUT:\n", raw_output)

        try:
            return self._safe_json_parse(raw_output)
        except Exception as e:
            print("❌ AWB extraction failed:", str(e))
            return {
                "error": "AWB_EXTRACTION_FAILED",
                "raw_llm_output": raw_output
            }
