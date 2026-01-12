import os
import json
import re
from openai import AzureOpenAI


class CourierDispatchAdviceLLMExtractor:
    """
    Extract mandatory fields from COURIER DISPATCH ADVICE
    Used by banks to track document movement
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
        Safely extract JSON from LLM output
        """
        if not text:
            raise ValueError("Empty LLM response")

        text = text.strip()
        text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in output:\n{text}")

        return json.loads(match.group(0))

    def extract(self, normalized_doc: dict) -> dict:
        """
        Extract Courier Dispatch Advice mandatory fields
        """

        system_prompt = """
You are a Trade Finance Document Extraction Engine.

Document Type: COURIER DISPATCH ADVICE

Purpose:
- Track dispatch of DOCUMENTS (not goods)
- Used by banks and exporters

Extraction Rules:
- Extract ONLY if explicitly present
- Do NOT guess or infer
- Missing fields must be null
- Output MUST be valid JSON only
- No explanations

Required JSON Schema:
{
  "exporter_name": null,
  "exporter_address": null,
  "importer_or_bank_name": null,
  "importer_or_bank_address": null,
  "courier_company": null,
  "courier_awb_number": null,
  "dispatch_date": null,
  "contract_number": null,
  "invoice_number": null,
  "documents_sent": [],
  "authorized_signatory": null,
  "exporter_stamp_present": null
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

        # Debug (comment in prod)
        print("\n🔎 RAW COURIER DISPATCH LLM OUTPUT:\n", raw_output)

        try:
            return self._safe_json_parse(raw_output)
        except Exception as e:
            print("❌ Courier Dispatch Advice extraction failed:", str(e))
            return {
                "error": "COURIER_DISPATCH_ADVICE_EXTRACTION_FAILED",
                "raw_llm_output": raw_output
            }
