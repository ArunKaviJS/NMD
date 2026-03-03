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

    def extract(self, normalized_doc):
        """
        Extract CERTIFICATE OF ORIGIN fields
        """

        system_prompt = """
        You are a Trade Finance Document Extraction Engine.

        Document Type: CERTIFICATE OF ORIGIN

        Rules:
        - Extract ONLY information explicitly stated in the document
        - DO NOT infer, calculate, or assume values
        - Preserve original wording exactly as it appears
        - If a field is not clearly present, return null
        - Output MUST be valid JSON only
        - Do NOT add extra fields
        - Do NOT rename fields
        - Do NOT explain anything

        Field Mapping Rules:
        - Importer = Consignee
        - Beneficiary = Exporter (if no separate beneficiary mentioned)
        - Shipper = Exporter (if no separate shipper mentioned)
        - Country of Origin must be explicitly stated (e.g., “People's Republic of China”)
        - Certificate Number must appear explicitly as Certificate No / Certificate Number (Invoice or LC number must NOT be used as certificate number)

        Required JSON Schema:

        {
        "certificate_number": null,
        "importer": null,
        "exporter": null,
        "goods_description": null,
        "country_of_origin": null,
        "beneficiary": null,
        "shipper": null
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
