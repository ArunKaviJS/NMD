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
        You are a Trade Finance Certificate of Origin Extraction Engine.

        Document Type: CERTIFICATE OF ORIGIN (NON-PREFERENTIAL or PREFERENTIAL)

        Rules:
        - Extract values ONLY if explicitly present in the document
        - DO NOT guess or infer
        - DO NOT explain or add commentary
        - If a field is not clearly mentioned, return null
        - Output MUST be valid JSON only — no markdown, no extra text
        - Do NOT add extra fields
        - Do NOT rename fields

        Field Mapping Rules:
        - exporter = Exporter name + full address as one string (from field 1 or Exporter section)
        - consignee = Consignee name + full address as one string (from field 2 or Consignee section)
        - port_of_loading = departure port / from port (found in Means of Transport & Route section)
        - port_of_discharge = destination port / to port (found in Means of Transport & Route section)
        - country_of_origin = country explicitly stated as origin (field 4)
        - country_of_destination = destination country (field 5)
        - hs_code = Harmonized System / HS Code from goods description table (field 8)
        - commodity_description = full goods description as written (field 7 — Marks, Numbers & Description)
        - quantity = number of units with unit type (e.g. "5,000 Bags") from field 9
        - net_weight = net weight with unit (e.g. "125.000 MT") from field 10
        - gross_weight = gross weight with unit (e.g. "126.250 MT") from field 11
        - invoice_reference = Invoice number and date referenced in the certificate (field 12)
        - certificate_number = Certificate / Cert No at the top of the document
        - certificate_date = Date of issue of the certificate
        - issuing_authority = Name of the issuing body / chamber (e.g. Chennai Chamber of Commerce)
        - issuing_authority_date = Date stamped/signed by the issuing authority (field 14)
        - number_of_packages = Extract the packing marks, numbers, and package count/type 
                            from the MARKS, NUMBERS & DESCRIPTION column (field 7).
                            Look for bag type, weight per bag, marks/stencil codes.
                            Return as a descriptive string (e.g. "5,000 PP Woven Bags x 25 kg | Marks: AAEPL/2026")

        Required JSON Schema:

        {
        "certificate_number": null,
        "certificate_date": null,
        "exporter": null,
        "consignee": null,
        "port_of_loading": null,
        "port_of_discharge": null,
        "country_of_origin": null,
        "country_of_destination": null,
        "hs_code": null,
        "commodity_description": null,
        "quantity": null,
        "net_weight": null,
        "gross_weight": null,
        "number_of_packages": null,
        "invoice_reference": null,
        "issuing_authority": null,
        "issuing_authority_date": null
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
        print("\n📄 RAW CERTIFICATE OF ORIGIN LLM OUTPUT:\n", raw_output)

        try:
            return self._safe_json_parse(raw_output)
        except Exception as e:
            print("❌ CO extraction failed:", str(e))
            return {
                "error": "CERTIFICATE_OF_ORIGIN_EXTRACTION_FAILED",
                "raw_llm_output": raw_output
            }
