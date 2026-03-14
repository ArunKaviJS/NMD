import os
import json
import re
from openai import AzureOpenAI


class InspectionCertificateLLMExtractor:
    """
    Extract mandatory fields from InspectionCertificate
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

    def extract(self, normalized_doc):
        """
        Extract InspectionCertificate mandatory fields
        """

        system_prompt = """
        You are a Trade Finance Inspection Certificate Extraction Engine.

        Document Type: INSPECTION CERTIFICATE / QUALITY & QUANTITY CERTIFICATE

        Rules:
        - Extract values ONLY if explicitly present in the document
        - DO NOT guess or infer
        - DO NOT explain or add commentary
        - If a field is not clearly mentioned, return null
        - Output MUST be valid JSON only — no markdown, no extra text
        - Do NOT add extra fields
        - Do NOT rename fields

        Field Mapping Rules:
        - certificate_number = Cert No / Certificate Number at the top of the document
        - inspection_date = Date the physical inspection was carried out
        - report_date = Date the certificate / report was issued (may differ from inspection date)
        - issuing_body = Full name of the inspection/certification company issuing the certificate
                        (e.g. "SGS India Pvt Ltd")
        - inspector_name = Name of the inspector who conducted the inspection (if mentioned)
        - inspector_id = Inspector ID / accreditation number (if mentioned)
        - client_exporter = Client / Exporter full name + address as one string
        - consignee = Consignee full name + address as one string
        - inspection_location = Warehouse / location where inspection was physically conducted
        - commodity = Full commodity / goods description including variety, grade, processing type
        - hs_code = HS Code of the inspected goods
        - quantity_stated = Quantity as per invoice/contract (number + unit)
        - quantity_inspected = Actual quantity physically inspected (number + unit)
        - quantity_variance = Variance between stated and inspected quantity
        - quantity_status = Outcome of quantity verification (e.g. "VERIFIED & CORRECT")
        - net_weight = Net weight of inspected goods with unit
        - gross_weight = Gross weight of inspected goods with unit
        - packing_details = Packing type, weight per pack, palletisation details as one string
        - container_number = Container number if mentioned in packing/cargo details
        - test_results = Array of objects — one entry per inspection parameter tested.
                        Each object must have exactly these keys:
                        "parameter"    : name of the test/parameter
                        "specification": required specification / acceptable limit
                        "result_found" : actual result found during inspection
                        "conformity"   : "CONFORMS" / "DOES NOT CONFORM" / "N/A"
        - overall_conclusion = Overall conclusion statement as written in the certificate
                            (e.g. "GOODS FOUND TO BE IN CONFORMITY WITH SPECIFICATIONS")
        - invoice_reference = Invoice number referenced in the certificate
        - lc_number = LC number referenced in the certificate
        - vessel = Vessel name referenced
        - destination = Destination port / country mentioned

        Required JSON Schema:

        {
        "certificate_number": null,
        "inspection_date": null,
        "report_date": null,
        "issuing_body": null,
        "inspector_name": null,
        "inspector_id": null,
        "client_exporter": null,
        "consignee": null,
        "inspection_location": null,
        "commodity": null,
        "hs_code": null,
        "quantity_stated": null,
        "quantity_inspected": null,
        "quantity_variance": null,
        "quantity_status": null,
        "net_weight": null,
        "gross_weight": null,
        "packing_details": null,
        "container_number": null,
        "test_results": [],
        "overall_conclusion": null,
        "invoice_reference": null,
        "lc_number": null,
        "vessel": null,
        "destination": null
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

        # Debug (disable in production)
        print("\n✈️ RAW InspectionCertificateLLMExtractor LLM OUTPUT:\n", raw_output)

        try:
            return self._safe_json_parse(raw_output)
        except Exception as e:
            print("❌ InspectionCertificateLLMExtractor extraction failed:", str(e))
            return {
                "error": "InspectionCertificateLLMExtractorFailed",
                "raw_llm_output": raw_output
            }
