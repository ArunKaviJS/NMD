import os
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

class CertificateOfOriginLLMExtractor:
    """
    Extract mandatory fields from CERTIFICATE OF ORIGIN (CO)

    Used for trade finance document checking and LC compliance
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
        Extract CERTIFICATE OF ORIGIN fields
        """

        system_prompt = """
You are a Trade Finance Document Extraction Engine.

Document Type: CERTIFICATE OF ORIGIN

Extraction Rules:
- Extract ONLY information explicitly stated in the document
- DO NOT infer or assume values
- Preserve original wording as it appears
- If a field is not present, set it to null
- Output MUST be valid JSON only
- No explanations, summaries, or commentary

Required JSON Schema:
{
  "certificate_type": null,
  "certificate_number": null,
  "issuing_authority": null,
  "place_of_issue": null,
  "date_of_issue": null,

  "exporter_name": null,
  "exporter_address": null,

  "consignee_name": null,
  "consignee_address": null,

  "buyer_name": null,
  "buyer_address": null,

  "country_of_origin": null,
  "country_of_destination": null,

  "method_of_dispatch": null,
  "type_of_shipment": null,

  "vessel_or_aircraft": null,
  "voyage_or_flight_number": null,

  "port_of_loading": null,
  "port_of_discharge": null,
  "final_destination": null,
  "date_of_departure": null,

  "goods_description": null,
  "hs_code": null,
  "number_of_packages": null,
  "package_type": null,
  "gross_weight": null,

  "invoice_number": null,
  "invoice_date": null,
  "lc_number": null,

  "chamber_declaration_present": false,
  "exporter_declaration_present": false,

  "authorized_signatory_name": null,
  "signatory_company": null
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
        print("\n📄 RAW CERTIFICATE OF ORIGIN LLM OUTPUT:\n", raw_output)

        try:
            return self._safe_json_parse(raw_output)
        except Exception as e:
            print("❌ CO extraction failed:", str(e))
            return {
                "error": "CERTIFICATE_OF_ORIGIN_EXTRACTION_FAILED",
                "raw_llm_output": raw_output
            }
