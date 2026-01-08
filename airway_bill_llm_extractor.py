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

    def extract(self, normalized_doc: dict) -> dict:
        """
        Extract AIR WAYBILL mandatory fields
        """

        system_prompt = """
You are a Trade Finance Air Waybill (AWB) Extraction Engine.

Document Type: AIR WAYBILL (AWB)

Purpose:
- Controls cargo release
- Used by airlines, banks, and customs

Extraction Rules:
- Extract ONLY what is explicitly present
- Do NOT guess or infer
- Missing fields must be null
- Output MUST be valid JSON only
- No explanations

Required JSON Schema:
{
  "awb_number": null,
  "awb_type": null,
  "shipper_name": null,
  "shipper_address": null,
  "consignee_name": null,
  "consignee_address": null,
  "airport_of_origin": null,
  "airport_of_destination": null,
  "flight_number": null,
  "flight_date": null,
  "goods_description": null,
  "number_of_packages": null,
  "gross_weight": null,
  "freight_terms": null,
  "carrier_or_courier_stamp_present": null
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
