import os
import json
import re
from openai import AzureOpenAI


class InvoiceLLMExtractor:

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    def _safe_json_parse(self, text: str) -> dict:
        """
        Safely extract JSON object from LLM output
        """

        if not text:
            raise ValueError("LLM returned empty response")

        # Remove markdown ```json ``` wrappers
        text = text.strip()
        text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

        # Extract first JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in LLM output:\n{text}")

        return json.loads(match.group(0))

    def extract(self, normalized_doc: dict) -> dict:

        system_prompt = """
You are a Trade Finance Invoice Extraction Engine.

Document Type: COMMERCIAL INVOICE or PROFORMA INVOICE

Rules:
- Extract values ONLY if explicitly present
- DO NOT guess
- DO NOT explain
- Missing fields must be null
- Output MUST be valid JSON only

Required JSON Schema:
{
  "invoice_type": null,
  "invoice_number": null,
  "invoice_date": null,
  "seller_name": null,
  "seller_address": null,
  "buyer_name": null,
  "buyer_address": null,
  "goods_description": null,
  "hs_code": null,
  "quantity": null,
  "unit_price": null,
  "total_amount": null,
  "currency": null,
  "incoterms": null,
  "port_of_loading": null,
  "port_of_discharge": null,
  "country_of_origin": null,
  "payment_terms": null
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

        # 🔍 OPTIONAL DEBUG (comment out in prod)
        print("\n🔎 RAW LLM OUTPUT:\n", raw_output)

        try:
            return self._safe_json_parse(raw_output)
        except Exception as e:
            print("❌ Invoice LLM parsing failed:", str(e))
            return {
                "error": "INVOICE_EXTRACTION_FAILED",
                "raw_llm_output": raw_output
            }
