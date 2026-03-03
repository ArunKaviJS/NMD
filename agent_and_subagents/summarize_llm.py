import os
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()


system_prompt = """
  You are a trade finance operations assistant for a bank.

  You will be given structured data extracted from:

  Letter of Credit (LC)
  Commercial Invoice
  Certificate of Origin (COO)
  Air Waybill (AWB)

  Your task is to:

  Extract and structure required key fields into separate tables.
  Perform cross-document comparison checks.
  Identify only factual, document-based observations.
  Clearly indicate matches or mismatches.
  Provide concise operational remarks suitable for banking review.
  Return STRICT JSON output only.
  
  ---------------------
  STRICT BEHAVIOR RULES
 ----------------------
 
Do NOT assume intent.
Do NOT make commercial judgments.
Treat LC terms as strict.
Do NOT invent clauses or requirements not present in data.
Do NOT add explanations outside JSON.
Do NOT repeat generic review phrases.
Keep comparison remarks concise (1–2 lines maximum per comparison).
Mention specific document name when reporting mismatch.
If all documents match → state “Match”.
If mismatch → clearly state which document differs and how.

DOCUMENT TABLE STRUCTURE REQUIREMENTS

You MUST create the following EXACTLY with these names:

Fields:
lcNumber
lcIssueDate
lcExpiryDate
lcAmount
lastDateOfShipment
requiredDocuments
applicantImporter

invoiceNumber
invoiceDate
invoiceAmount
goodsDescription
importer
exporter



certificateNumber
importer
exporter
goodsDescription
countryOfOrigin



awbNumber
awbDate
shipper
consignee
shipmentDate

COMPARISON SECTION REQUIREMENTS

You MUST include the following comparison checks:

beneficiaryShipperComparison
Comparison on LC, Invoice, COO, AWB – Beneficiary/Shipper
Output format examples:

"Match"

"Mismatch – AWB consignee differs from LC beneficiary."

"Mismatch – COO exporter name differs from Invoice exporter."

applicantConsigneeComparison
Comparison on LC, Invoice, AWB – Applicant/Consignee
Output examples:

"Match"

"Mismatch – AWB consignee differs from LC applicant."

invoiceAmountComparison
Comparison on Invoice & LC – Invoice Amount
Output examples:

"Invoice amount is within LC limit."

"Invoice amount exceeds LC limit."

shipmentDateComparison
Comparison on LC & AWB – Shipment Date
Output examples:

"Within LC shipment validity."

"Shipment date exceeds LC last date of shipment."

goodsDescriptionComparison
Comparison on LC, Invoice, COO, AWB – Goods Description
Output examples:

"Description matches."

"Minor deviation – AWB includes packing details."

"Material mismatch – Invoice description differs from LC."
OUTPUT FORMAT – STRICT JSON ONLY

You MUST return a SINGLE valid JSON object.

Do NOT:

Add markdown

Add explanations

Add headings outside JSON

Wrap JSON in code blocks

The JSON structure MUST follow this format:

{
"lcNumber": "...",
"lcIssueDate": "...",
"lcExpiryDate": "...",
"lcAmount": "...",
"lastDateOfShipment": "...",
"requiredDocuments": "...",
"applicantImporter": "...",

"invoiceNumber": "...",
"invoiceDate": "...",
"invoiceAmount": "...",
"invoiceGoodsDescription": "...",
"invoiceImporter": "...",
"invoiceExporter": "...",

"certificateNumber": "...",
"cooImporter": "...",
"cooExporter": "...",
"cooGoodsDescription": "...",
"countryOfOrigin": "...",

"awbNumber": "...",
"awbDate": "...",
"awbShipper": "...",
"awbConsignee": "...",
"shipmentDate": "...",

"beneficiaryShipperComparison": "...",
"applicantConsigneeComparison": "...",
"invoiceAmountComparison": "...",
"shipmentDateComparison": "...",
"goodsDescriptionComparison": "..."
}

Return ONLY this JSON structure. No additional text.
"""
class SummarizeLLM:

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

    def extract(self, payload: dict) -> dict:
        documents = payload.get("documents", [])
        missing_documents = payload.get("missing_documents", [])
        """
        Send normalized document data to LLM and get structured JSON output
        ready for MongoDB storage.
        """

        response = self.client.chat.completions.create(
        model=self.deployment,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps({
                    "documents": documents,
                    "missing_documents": missing_documents
                })
            }
        ],
    )


        raw_output = response.choices[0].message.content.strip()

        # Debug / audit log
        print("\n🏦 TRADE FINANCE COMPLIANCE SUMMARY:\n")
        print(raw_output)

        # Parse into dict (safe)
        parsed_output = self._safe_json_parse(raw_output)
        parsed_output["MissingDocuments"] = (
        missing_documents if missing_documents else ["No missing documents noted"]
    )


        # Ensure keys exist and are correct types
        required_keys = [
            "overallStatus",
            "summary",
            "lcValidationSummary",
            "detailedFindings",
            "missingDocuments",
        ]
        for key in required_keys:
            if key not in parsed_output:
                parsed_output[key] = {} if key == "detailedFindings" else []

        return parsed_output