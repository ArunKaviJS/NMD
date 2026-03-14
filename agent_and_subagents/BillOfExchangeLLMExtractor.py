import os
import json
import re
from openai import AzureOpenAI


class BillOfExchangeLLMExtractor:
    """
    Extract mandatory fields from BillOfExchangeLLMExtractor
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

    def extract(self, normalized_doc):
        """
        Extract BillOfExchangeLLMExtractor mandatory fields
        """

        system_prompt = """
        You are a Trade Finance Bill of Exchange Extraction Engine.

        Document Type: BILL OF EXCHANGE (drawn under Letter of Credit, UCP 600)

        Rules:
        - Extract values ONLY if explicitly present in the document
        - DO NOT guess or infer
        - DO NOT explain or add commentary
        - If a field is not clearly mentioned, return null
        - Output MUST be valid JSON only — no markdown, no extra text
        - Do NOT add extra fields
        - Do NOT rename fields

        Field Mapping Rules:
        - boe_number = Bill of Exchange number / reference at the top of the document
        - boe_date = Date of the Bill of Exchange
        - boe_place = Place of drawing / issuance
        - drawer = Beneficiary / Exporter who draws the bill
                    (full name + address + IEC/PAN + bank details if present, as one string)
        - drawee = Issuing Bank / party on whom the bill is drawn
                    (full name + address + SWIFT if present, as one string)
        - pay_to_order_of = The bank or party to whom payment is ordered (the presenting/negotiating bank)
        - amount_figures = Amount in numeric figures (e.g. "USD 1,25,000.00")
        - amount_words = Amount in words exactly as written in the document
        - currency = Currency code (e.g. "USD")
        - tenor = Payment term — "AT SIGHT" for sight bills, or usance period (e.g. "90 days after sight")
        - lc_number = Letter of Credit number this bill is drawn under
        - lc_date = Date of the Letter of Credit referenced
        - lc_type = Type of LC (e.g. "Irrevocable, Sight")
        - confirming_bank = Confirming bank if mentioned
        - invoice_number = Commercial Invoice number referenced
        - invoice_date = Commercial Invoice date referenced
        - bl_number = Bill of Lading number referenced
        - vessel = Vessel name referenced
        - port_of_loading = Port goods shipped from
        - port_of_discharge = Port goods shipped to
        - goods_description = Brief goods description as stated in the bill
        - incoterm = Trade term stated (e.g. "CIF Jebel Ali", "FOB Chennai")
        - account_of = Party for whose account the bill is drawn (usually the Applicant/Buyer)

        Required JSON Schema:

        {
        "drawer": null,
        "drawee": null,
        "amount_figures": null,
        "amount_words": null,
        "currency": null,
        "tenor": null,
        "lc_number": null,
        "lc_date": null,
        "lc_type": null,
        "confirming_bank": null,
        "invoice_number": null,
        "invoice_date": null,
        "bl_number": null,
        "incoterm": null,
        "account_of": null
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
            print("❌ BillOfExchangeLLMExtractor extraction failed:", str(e))
            return {
                "error": "BillOfExchangeLLMExtractor_EXTRACTION_FAILED",
                "raw_llm_output": raw_output
            }
