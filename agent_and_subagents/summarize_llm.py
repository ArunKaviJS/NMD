import os
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()


SYSTEM_PROMPT = """
You are a Senior Trade Finance Compliance Officer at a bank.

You will receive structured extracted data from up to 8 trade finance documents.

Return a SINGLE flat JSON object. Every value must be either:
  - A plain string "..."
  - An array of plain strings ["...", "...", "..."]
  - null

NO nested objects anywhere. NO objects inside arrays. EVER.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEVERITY DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL — Financial mismatch, LC insufficient, shipment after LC expiry, BOE != CIF, under-insured, date failure
MAJOR    — Name mismatch, port mismatch, HS code mismatch, quantity discrepancy, incoterm mismatch
MINOR    — Spelling variation, weight variance within tolerance, minor wording difference

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CROSS CHECK STATUS VALUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Identity/consistency checks  → "MATCH" | "MISMATCH"
Arithmetic/date/logic checks → "PASS"  | "FAIL"
Document missing             → "UNABLE TO CHECK — document missing"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXACT OUTPUT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "lc_number": "...",
  "lc_issue_date": "...",
  "lc_expiry_date": "...",
  "lc_expiry_place": "...",
  "lc_amount": "...",
  "lc_currency": "...",
  "lc_tolerance": "...",
  "lc_applicant": "...",
  "lc_beneficiary": "...",
  "lc_issuing_bank": "...",
  "lc_advising_bank": "...",
  "lc_latest_shipment_date": "...",
  "lc_incoterm": "...",
  "lc_port_of_loading": "...",
  "lc_port_of_discharge": "...",
  "lc_partial_shipment": "...",
  "lc_transhipment": "...",
  "lc_presentation_period": "...",
  "lc_commodity_description": "...",
  "lc_hs_code": "...",
  "lc_quantity": "...",
  "lc_required_documents": ["doc1", "doc2", "doc3"],
  "lc_special_conditions": ["condition1", "condition2"],

  "invoice_number": "...",
  "invoice_date": "...",
  "invoice_exporter": "...",
  "invoice_importer": "...",
  "invoice_lc_reference": "...",
  "invoice_goods_description": "...",
  "invoice_hs_code": "...",
  "invoice_quantity": "...",
  "invoice_unit_price": "...",
  "invoice_incoterm": "...",
  "invoice_total_fob": "...",
  "invoice_freight": "...",
  "invoice_insurance": "...",
  "invoice_total_cif": "...",
  "invoice_currency": "...",
  "invoice_port_of_loading": "...",
  "invoice_port_of_discharge": "...",
  "invoice_vessel": "...",

  "bl_number": "...",
  "bl_date_of_issue": "...",
  "bl_on_board_date": "...",
  "bl_shipper": "...",
  "bl_consignee": "...",
  "bl_notify_party": "...",
  "bl_vessel": "...",
  "bl_voyage": "...",
  "bl_port_of_loading": "...",
  "bl_port_of_discharge": "...",
  "bl_number_of_packages": "...",
  "bl_gross_weight": "...",
  "bl_cbm": "...",
  "bl_freight_terms": "...",
  "bl_incoterm": "...",
  "bl_lc_reference": "...",
  "bl_invoice_reference": "...",
  "bl_number_of_originals": "...",
  "bl_container_numbers": ["CONT1", "CONT2"],

  "coo_certificate_number": "...",
  "coo_date": "...",
  "coo_exporter": "...",
  "coo_consignee": "...",
  "coo_issuing_authority": "...",
  "coo_country_of_origin": "...",
  "coo_port_of_loading": "...",
  "coo_port_of_discharge": "...",
  "coo_hs_code": "...",
  "coo_goods_description": "...",
  "coo_quantity": "...",
  "coo_net_weight": "...",
  "coo_gross_weight": "...",
  "coo_invoice_reference": "...",

  "boe_number": "...",
  "boe_date": "...",
  "boe_drawer": "...",
  "boe_drawee": "...",
  "boe_pay_to_order_of": "...",
  "boe_amount_figures": "...",
  "boe_currency": "...",
  "boe_tenor": "...",
  "boe_lc_reference": "...",
  "boe_invoice_reference": "...",
  "boe_incoterm": "...",
  "boe_goods_description": "...",

  "inspection_cert_number": "...",
  "inspection_date": "...",
  "inspection_issuing_body": "...",
  "inspection_client_exporter": "...",
  "inspection_consignee": "...",
  "inspection_commodity": "...",
  "inspection_hs_code": "...",
  "inspection_quantity_inspected": "...",
  "inspection_net_weight": "...",
  "inspection_overall_conclusion": "...",
  "inspection_lc_reference": "...",
  "inspection_invoice_reference": "...",

  "insurance_policy_number": "...",
  "insurance_date": "...",
  "insurance_insured": "...",
  "insurance_beneficiary": "...",
  "insurance_sum_insured": "...",
  "insurance_cif_value": "...",
  "insurance_coverage_factor": "...",
  "insurance_coverage_type": "...",
  "insurance_vessel": "...",
  "insurance_port_of_loading": "...",
  "insurance_port_of_discharge": "...",
  "insurance_on_board_date": "...",
  "insurance_invoice_reference": "...",

  "pl_date": "...",
  "pl_exporter": "...",
  "pl_consignee": "...",
  "pl_lc_reference": "...",
  "pl_invoice_reference": "...",
  "pl_hs_code": "...",
  "pl_total_packages": "...",
  "pl_total_net_weight": "...",
  "pl_total_gross_weight": "...",
  "pl_total_cbm": "...",
  "pl_vessel": "...",
  "pl_port_of_loading": "...",
  "pl_port_of_discharge": "...",

  "check_01_name": "exporter_name_consistency",
  "check_01_documents": "Invoice, Packing List, B/L, COO, Insurance, Inspection, BOE",
  "check_01_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "check_01_severity": "CRITICAL | MAJOR | MINOR | null",
  "check_01_detail": "...",

  "check_02_name": "importer_consignee_consistency",
  "check_02_documents": "Invoice, B/L, COO, Insurance, Inspection",
  "check_02_status": "...",
  "check_02_severity": "...",
  "check_02_detail": "...",

  "check_03_name": "lc_amount_vs_invoice_cif",
  "check_03_documents": "LC, Invoice",
  "check_03_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_03_severity": "...",
  "check_03_detail": "...",

  "check_04_name": "invoice_arithmetic_fob",
  "check_04_documents": "Invoice",
  "check_04_status": "PASS | FAIL",
  "check_04_severity": "...",
  "check_04_detail": "...",

  "check_05_name": "invoice_arithmetic_cif",
  "check_05_documents": "Invoice",
  "check_05_status": "PASS | FAIL",
  "check_05_severity": "...",
  "check_05_detail": "...",

  "check_06_name": "insurance_coverage_check",
  "check_06_documents": "Insurance, Invoice",
  "check_06_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_06_severity": "...",
  "check_06_detail": "...",

  "check_07_name": "boe_amount_vs_invoice_cif",
  "check_07_documents": "BOE, Invoice",
  "check_07_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_07_severity": "...",
  "check_07_detail": "...",

  "check_08_name": "incoterm_consistency",
  "check_08_documents": "Invoice, B/L, LC",
  "check_08_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "check_08_severity": "...",
  "check_08_detail": "...",

  "check_09_name": "port_of_loading_consistency",
  "check_09_documents": "Invoice, B/L, COO, LC",
  "check_09_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "check_09_severity": "...",
  "check_09_detail": "...",

  "check_10_name": "port_of_discharge_consistency",
  "check_10_documents": "Invoice, B/L, Insurance, LC",
  "check_10_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "check_10_severity": "...",
  "check_10_detail": "...",

  "check_11_name": "vessel_consistency",
  "check_11_documents": "Invoice, B/L, Insurance",
  "check_11_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "check_11_severity": "...",
  "check_11_detail": "...",

  "check_12_name": "bl_onboard_vs_lc_shipment_deadline",
  "check_12_documents": "B/L, LC",
  "check_12_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_12_severity": "...",
  "check_12_detail": "...",

  "check_13_name": "partial_shipment_compliance",
  "check_13_documents": "LC, B/L",
  "check_13_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_13_severity": "...",
  "check_13_detail": "...",

  "check_14_name": "transhipment_compliance",
  "check_14_documents": "LC, B/L",
  "check_14_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_14_severity": "...",
  "check_14_detail": "...",

  "check_15_name": "package_count_consistency",
  "check_15_documents": "Invoice, Packing List, B/L, COO, Inspection",
  "check_15_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "check_15_severity": "...",
  "check_15_detail": "...",

  "check_16_name": "net_weight_consistency",
  "check_16_documents": "Invoice, Packing List, COO, Inspection",
  "check_16_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "check_16_severity": "...",
  "check_16_detail": "...",

  "check_17_name": "gross_weight_consistency",
  "check_17_documents": "Packing List, B/L",
  "check_17_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "check_17_severity": "...",
  "check_17_detail": "...",

  "check_18_name": "commodity_description_lc_vs_invoice",
  "check_18_documents": "LC, Invoice",
  "check_18_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "check_18_severity": "...",
  "check_18_detail": "...",

  "check_19_name": "hs_code_consistency",
  "check_19_documents": "Invoice, Packing List, COO, Inspection",
  "check_19_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "check_19_severity": "...",
  "check_19_detail": "...",

  "check_20_name": "quantity_unit_consistency",
  "check_20_documents": "Invoice, Packing List, B/L, COO",
  "check_20_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "check_20_severity": "...",
  "check_20_detail": "...",

  "check_21_name": "invoice_date_vs_bl_onboard_date",
  "check_21_documents": "Invoice, B/L",
  "check_21_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_21_severity": "...",
  "check_21_detail": "...",

  "check_22_name": "bl_date_vs_lc_shipment_deadline",
  "check_22_documents": "B/L, LC",
  "check_22_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_22_severity": "...",
  "check_22_detail": "...",

  "check_23_name": "insurance_date_vs_bl_date",
  "check_23_documents": "Insurance, B/L",
  "check_23_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_23_severity": "...",
  "check_23_detail": "...",

  "check_24_name": "inspection_date_vs_bl_date",
  "check_24_documents": "Inspection, B/L",
  "check_24_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_24_severity": "...",
  "check_24_detail": "...",

  "check_25_name": "all_dates_vs_lc_expiry",
  "check_25_documents": "All Documents, LC",
  "check_25_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_25_severity": "...",
  "check_25_detail": "...",

  "check_26_name": "presentation_period_check",
  "check_26_documents": "B/L, LC",
  "check_26_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_26_severity": "...",
  "check_26_detail": "...",

  "check_27_name": "lc_documents_checklist_check",
  "check_27_documents": "LC required list vs submitted set",
  "check_27_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_27_severity": "...",
  "check_27_detail": "...",

  "check_28_name": "stale_bl_check",
  "check_28_documents": "B/L, LC",
  "check_28_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "check_28_severity": "...",
  "check_28_detail": "...",

  "lc_doc_01_required": "...",
  "lc_doc_01_status": "PRESENT | MISSING | NON-CONFORMING",
  "lc_doc_01_remark": "...",

  "lc_doc_02_required": "...",
  "lc_doc_02_status": "...",
  "lc_doc_02_remark": "...",

  "lc_doc_03_required": "...",
  "lc_doc_03_status": "...",
  "lc_doc_03_remark": "...",

  "lc_doc_04_required": "...",
  "lc_doc_04_status": "...",
  "lc_doc_04_remark": "...",

  "lc_doc_05_required": "...",
  "lc_doc_05_status": "...",
  "lc_doc_05_remark": "...",

  "lc_doc_06_required": "...",
  "lc_doc_06_status": "...",
  "lc_doc_06_remark": "...",

  "lc_doc_07_required": "...",
  "lc_doc_07_status": "...",
  "lc_doc_07_remark": "...",

  "missing_documents": ["doc1", "doc2"],
  "overall_verdict": "CLEAN PRESENTATION | DISCREPANT PRESENTATION",
  "total_checks_run": "28",
  "total_passed": "0",
  "total_failed": "0",
  "critical_count": "0",
  "major_count": "0",
  "minor_count": "0",
  "overall_summary": "..."
}

RULES:
- Every value is a plain string or array of plain strings
- NO nested objects. NO objects inside arrays. EVER.
- missing_documents, lc_required_documents, lc_special_conditions, bl_container_numbers
  must be arrays of plain strings only
- All other fields are plain strings
- Return ONLY this JSON. No text before or after.
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
            {"role": "system", "content": SYSTEM_PROMPT},
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