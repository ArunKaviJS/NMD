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

    def extract(self, normalized_doc):

        system_prompt = """
        You are a Trade Finance Commercial Invoice Extraction Engine.

        Document Type: COMMERCIAL INVOICE or PROFORMA INVOICE

        Rules:
        - Extract values ONLY if explicitly present in the document
        - DO NOT guess or infer
        - DO NOT explain or add commentary
        - If a field is not clearly mentioned, return null
        - Output MUST be valid JSON only — no markdown, no extra text
        - Do NOT add extra fields
        - Do NOT rename fields

        Field Mapping Rules:
        - Exporter = Seller / Shipper (include full name + address as one string)
        - Importer = Buyer / Consignee (include full name + address as one string)
        - lc_number = any L/C No or LC reference mentioned in the document
        - goods_description = full commodity description as written in the document
        - hs_code = Harmonized System / HS Code of the goods
        - quantity = number of packages/bags/units with unit type (e.g. "5,000 Bags / 125 MT")
        - unit_price = price per unit with basis (e.g. "USD 24.00 per bag")
        - total_fob = FOB subtotal value
        - freight = ocean/air freight charges added
        - insurance = marine/air insurance premium added
        - total_cif = final CIF / CFR / C&F total value
        - incoterm = trade term used (e.g. CIF, FOB, CFR)
        - port_of_loading = port where goods are loaded
        - port_of_discharge = destination port
        - vessel_voyage = vessel name and/or voyage number
        - bank_details = full bank name, branch, account number, IFSC, SWIFT, correspondent bank

        Required JSON Schema:

        {
        "exporter": null,
        "importer_consignee": null,
        "invoice_number": null,
        "invoice_date": null,
        "lc_number": null,
        "goods_description": null,
        "hs_code": null,
        "quantity": null,
        "unit_price": null,
        "total_fob": null,
        "freight": null,
        "insurance": null,
        "total_cif": null,
        "currency": null,
        "incoterm": null,
        "port_of_loading": null,
        "port_of_discharge": null,
        "vessel_voyage": null,
        "bank_details": null
        }"""
        
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
