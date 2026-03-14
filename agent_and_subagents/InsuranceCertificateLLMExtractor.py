import os
import json
import re
from openai import AzureOpenAI


class InsuranceCertificateLLMExtractor:
    """
    Extract mandatory fields from InsuranceCertificate
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
        Extract Insurance Certificate mandatory fields
        """

        system_prompt = """
        You are a Trade Finance Marine Cargo Insurance Certificate Extraction Engine.

        Document Type: MARINE CARGO INSURANCE CERTIFICATE / POLICY

        Rules:
        - Extract values ONLY if explicitly present in the document
        - DO NOT guess or infer
        - DO NOT explain or add commentary
        - If a field is not clearly mentioned, return null
        - Output MUST be valid JSON only — no markdown, no extra text
        - Do NOT add extra fields
        - Do NOT rename fields

        Field Mapping Rules:
        - policy_number = Insurance Policy Number / Certificate Number at the top
        - certificate_date = Date the insurance certificate was issued
        - insured_assured = Name + full address of the Insured / Assured party (the exporter/seller)
                            as one string
        - beneficiary = Full beneficiary name as written — may include "and/or Order of [Bank]".
                        Capture the EXACT wording including any bank order clause.
                        This field appears twice in your required fields — extract once, used for both.
        - sum_insured = Total insured amount with currency (e.g. "USD 1,37,500.00")
        - cif_value = Base CIF / Invoice value the insurance is calculated on (e.g. "USD 1,25,000.00")
        - coverage_factor = Coverage percentage over CIF (e.g. "110% of CIF" or "CIF + 10%")
        - coverage_type = Institute Cargo Clauses type — extract as:
                        "ICC (A) — All Risks"  if Institute Cargo Clauses (A) is mentioned
                        "ICC (B)"              if Institute Cargo Clauses (B) is mentioned
                        "ICC (C)"              if Institute Cargo Clauses (C) is mentioned
                        Include any additional clauses on the same line
                        (e.g. "ICC (A) — All Risks + War Clauses + Strikes Clauses")
        - war_clause = "INCLUDED" or "NOT INCLUDED" based on Institute War Clauses mention
        - strikes_clause = "INCLUDED" or "NOT INCLUDED" based on Institute Strikes Clauses mention
        - vessel = Vessel name for the insured voyage
        - port_of_loading = Voyage FROM port (origin)
        - port_of_discharge = Voyage TO port (destination)
        - on_board_date = On Board Date / date goods were loaded onto vessel
                        (DIFFERENT from certificate date — look for "On Board Date" field specifically)
        - commodity = Full commodity description including variety/grade as one string
        - hs_code = HS Code of the insured goods
        - quantity_packing = Quantity and packing details as one string
        - marks_numbers = Shipping marks and numbers as written
        - conditions_of_cover = Array of strings — each coverage condition as a separate item
        - exclusions = Array of strings — each exclusion or special condition as a separate item
        - claims_payable_at = Name + address of claims settlement office
        - survey_agent = Name + address of survey agent at destination
        - invoice_reference = Invoice number referenced in the certificate
        - issuing_company = Full name of the insurance company issuing the certificate

        Required JSON Schema:

        {
        "policy_number": null,
        "certificate_date": null,
        "insured_assured": null,
        "beneficiary": null,
        "sum_insured": null,
        "cif_value": null,
        "coverage_factor": null,
        "coverage_type": null,
        "war_clause": null,
        "strikes_clause": null,
        "vessel": null,
        "port_of_loading": null,
        "port_of_discharge": null,
        "on_board_date": null,
        "commodity": null,
        "hs_code": null,
        "quantity_packing": null,
        "marks_numbers": null,
        "conditions_of_cover": [],
        "exclusions": [],
        "claims_payable_at": null,
        "survey_agent": null,
        "invoice_reference": null,
        "issuing_company": null
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
        print("\n✈️ RAW InsuranceCertificate LLM OUTPUT:\n", raw_output)

        try:
            return self._safe_json_parse(raw_output)
        except Exception as e:
            print("❌ InsuranceCertificate extraction failed:", str(e))
            return {
                "error": "InsuranceCertificatellm_FAILED",
                "raw_llm_output": raw_output
            }
