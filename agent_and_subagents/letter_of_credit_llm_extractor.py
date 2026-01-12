import os
import json
import re
from openai import AzureOpenAI


class LetterOfCreditLLMExtractor:
    """
    Extract mandatory fields from LETTER OF CREDIT (LC)
    Banks reject documents if LC data mismatches
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

    def extract(self, normalized_doc: dict) -> dict:
        """
        Extract LETTER OF CREDIT mandatory fields
        """

        system_prompt = """
You are a Trade Finance Letter of Credit (LC) Extraction Engine.

Document Type: LETTER OF CREDIT

Rules:
- Extract ONLY what is explicitly stated
- NO assumptions or interpretation
- Preserve original wording (bank-grade)
- Missing fields must be null
- Output MUST be valid JSON only
- No explanations or commentary

Required JSON Schema:
{
  "lc_number": null,
  "date_of_issue": null,
  "issuing_bank": null,
  "advising_bank": null,
  "applicant_name": null,
  "applicant_address": null,
  "beneficiary_name": null,
  "beneficiary_address": null,
  "currency": null,
  "lc_amount": null,
  "tolerance": null,
  "incoterms": null,
  "port_of_loading": null,
  "port_of_discharge": null,
  "latest_shipment_date": null,
  "documents_required": {
    "commercial_invoice": false,
    "packing_list": false,
    "air_waybill": false,
    "certificate_of_origin": false,
    "special_certificates": []
  },
  "payment_terms": null,
  "availability": null,
  "lc_expiry_date": null,
  "place_of_expiry": null
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

        # Debug – disable in production
        print("\n🏦 RAW LC LLM OUTPUT:\n", raw_output)

        try:
            return self._safe_json_parse(raw_output)
        except Exception as e:
            print("❌ LC extraction failed:", str(e))
            return {
                "error": "LC_EXTRACTION_FAILED",
                "raw_llm_output": raw_output
            }
