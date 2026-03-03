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

    def extract(self, normalized_doc):
        """
        Extract LETTER OF CREDIT mandatory fields
        """

        system_prompt = """
        You are a Trade Finance Letter of Credit (LC) Extraction Engine.

        Document Type: LETTER OF CREDIT

        Rules:
        - Extract ONLY information explicitly stated in the document
        - DO NOT infer, interpret, calculate, or assume
        - Preserve original wording exactly as written
        - If a field is not clearly mentioned, return null
        - Output MUST be valid JSON only
        - Do NOT add extra fields
        - Do NOT rename fields
        - Do NOT explain anything

        Field Mapping Rules:
        - Applicant/Importer = Applicant
        - Applicant/Consignee = Applicant (if Consignee not separately mentioned)
        - Beneficiary = Beneficiary
        - Shipper = Beneficiary (if Shipper not separately mentioned in LC)
        - LC Amount = Amount stated under “AMOUNT” or “NOT EXCEEDING”
        - Amount = Same value as LC Amount
        - Last Date of Shipment = Same as Shipment Date, “NOT LATER THAN” date under Shipment Terms
        - Required Documents must be extracted exactly as listed under “DOCUMENTS REQUIRED”
        - Goods Description must be taken exactly from the “Goods:” section

        Required JSON Schema:

        {
        "lc_number": null,
        "lc_issue_date": null,
        "lc_expiry_date": null,
        "lc_amount": null,
        "last_date_of_shipment": null,
        "required_documents": null,
        "applicant_importer": null,
        "beneficiary": null,
        "shipper": null,
        "applicant_consignee": null,
        "amount": null,
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
