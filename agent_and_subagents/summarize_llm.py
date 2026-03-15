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

  "exporter_name_consistency_name": "exporter_name_consistency",
  "exporter_name_documents": "Invoice, Packing List, B/L, COO, Insurance, Inspection, BOE",
  "exporter_name_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "exporter_name_severity": "CRITICAL | MAJOR | MINOR | null",
  "check_01_detail": "...",

  "importer_consignee_consistency_name": "importer_consignee_consistency",
  "importer_consignee_documents": "Invoice, B/L, COO, Insurance, Inspection",
  "importer_consignee_status": "...",
  "importer_consignee_severity": "...",
  "check_02_detail": "...",

  "lc_amount_vs_invoice_cif_name": "lc_amount_vs_invoice_cif",
  "documents_for_lc_amount_vs_invoice_cif": "LC, Invoice",
  "lc_amount_vs_invoice_cif_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "lc_amount_vs_invoice_cif_severity": "...",
  "check_03_detail": "...",

  "invoice_arithmetic_fob_name": "invoice_arithmetic_fob",
  "invoice_arithmetic_fob_documents": "Invoice",
  "invoice_arithmetic_fob_status": "PASS | FAIL",
  "invoice_arithmetic_fob_severity": "...",
  "check_04_detail": "...",

  "invoice_arithmetic_cif_name": "invoice_arithmetic_cif",
  "invoice_arithmetic_cif_documents": "Invoice",
  "invoice_arithmetic_cif5_status": "PASS | FAIL",
  "invoice_arithmetic_cif_severity": "...",
  "check_05_detail": "...",

  "insurance_coverage_check_name": "insurance_coverage_check",
  "insurance_coverage_check_documents": "Insurance, Invoice",
  "insurance_coverage_check_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "insurance_coverage_check_severity": "...",
  "insurance_coverage_check_detail": "...",

  "boe_amount_vs_invoice_cif_name": "boe_amount_vs_invoice_cif",
  "boe_amount_vs_invoice_cif_documents": "BOE, Invoice",
  "boe_amount_vs_invoice_cif_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "boe_amount_vs_invoice_cif_severity": "...",
  "boe_amount_vs_invoice_cif_detail": "...",

  "incoterm_consistency_name": "incoterm_consistency",
  "incoterm_consistency_documents": "Invoice, B/L, LC",
  "incoterm_consistency_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "incoterm_consistency_severity": "...",
  "incoterm_consistency_detail": "...",

  "port_of_loading_consistency_name": "port_of_loading_consistency",
  "port_of_loading_consistency_documents": "Invoice, B/L, COO, LC",
  "port_of_loading_consistency_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "port_of_loading_consistency_severity": "...",
  "port_of_loading_consistency_detail": "...",

  "port_of_discharge_consistency_name": "port_of_discharge_consistency",
  "port_of_discharge_consistency_documents": "Invoice, B/L, Insurance, LC",
  "port_of_discharge_consistency_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "port_of_discharge_consistency_severity": "...",
  "port_of_discharge_consistency_detail": "...",

  "vessel_consistency_name": "vessel_consistency",
  "vessel_consistency_documents": "Invoice, B/L, Insurance",
  "vessel_consistency_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "vessel_consistency_severity": "...",
  "vessel_consistency_detail": "...",

  "bl_onboard_vs_lc_shipment_deadline_name": "bl_onboard_vs_lc_shipment_deadline",
  "bl_onboard_vs_lc_shipment_deadline_documents": "B/L, LC",
  "bl_onboard_vs_lc_shipment_deadline_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "bl_onboard_vs_lc_shipment_deadline_severity": "...",
  "bl_onboard_vs_lc_shipment_deadline_detail": "...",

  "partial_shipment_compliance_name": "partial_shipment_compliance",
  "partial_shipment_compliance_documents": "LC, B/L",
  "partial_shipment_compliance_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "partial_shipment_compliance_severity": "...",
  "partial_shipment_compliance_detail": "...",

  "transhipment_compliance_name": "transhipment_compliance",
  "transhipment_compliance_documents": "LC, B/L",
  "transhipment_compliance_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "transhipment_compliance_severity": "...",
  "transhipment_compliance_detail": "...",

  "package_count_consistency_name": "package_count_consistency",
  "package_count_consistency_documents": "Invoice, Packing List, B/L, COO, Inspection",
  "package_count_consistency_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "package_count_consistency_severity": "...",
  "package_count_consistency_detail": "...",

  "net_weight_consistency_name": "net_weight_consistency",
  "net_weight_consistency_documents": "Invoice, Packing List, COO, Inspection",
  "net_weight_consistency_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "net_weight_consistency_severity": "...",
  "net_weight_consistency_detail": "...",

  "gross_weight_consistency_name": "gross_weight_consistency",
  "gross_weight_consistency_documents": "Packing List, B/L",
  "gross_weight_consistency_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "gross_weight_consistency_severity": "...",
  "gross_weight_consistency_detail": "...",

  "commodity_description_lc_vs_invoice_name": "commodity_description_lc_vs_invoice",
  "commodity_description_lc_vs_invoice_documents": "LC, Invoice",
  "commodity_description_lc_vs_invoice_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "commodity_description_lc_vs_invoice_severity": "...",
  "commodity_description_lc_vs_invoice_detail": "...",

  "hs_code_consistency_name": "hs_code_consistency",
  "hs_code_consistency_documents": "Invoice, Packing List, COO, Inspection",
  "hs_code_consistency_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "hs_code_consistency_severity": "...",
  "hs_code_consistency_detail": "...",

  "quantity_unit_consistency_name": "quantity_unit_consistency",
  "quantity_unit_consistency_documents": "Invoice, Packing List, B/L, COO",
  "quantity_unit_consistency_status": "MATCH | MISMATCH | UNABLE TO CHECK — document missing",
  "quantity_unit_consistency_severity": "...",
  "quantity_unit_consistency_detail": "...",

  "invoice_date_vs_bl_onboard_date_name": "invoice_date_vs_bl_onboard_date",
  "invoice_date_vs_bl_onboard_date_documents": "Invoice, B/L",
  "invoice_date_vs_bl_onboard_date_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "invoice_date_vs_bl_onboard_date_severity": "...",
  "invoice_date_vs_bl_onboard_date_detail": "...",

  "bl_date_vs_lc_shipment_deadline_name": "bl_date_vs_lc_shipment_deadline",
  "bl_date_vs_lc_shipment_deadline_documents": "B/L, LC",
  "bl_date_vs_lc_shipment_deadline_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "bl_date_vs_lc_shipment_deadline_severity": "...",
  "bl_date_vs_lc_shipment_deadline_detail": "...",

  "insurance_date_vs_bl_date_name": "insurance_date_vs_bl_date",
  "insurance_date_vs_bl_date_documents": "Insurance, B/L",
  "insurance_date_vs_bl_date_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "insurance_date_vs_bl_date_severity": "...",
  "insurance_date_vs_bl_date_detail": "...",

  "inspection_date_vs_bl_date_name": "inspection_date_vs_bl_date",
  "inspection_date_vs_bl_date_documents": "Inspection, B/L",
  "inspection_date_vs_bl_date_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "inspection_date_vs_bl_date_severity": "...",
  "inspection_date_vs_bl_date_detail": "...",

  "all_dates_vs_lc_expiry_name": "all_dates_vs_lc_expiry",
  "all_dates_vs_lc_expiry_documents": "All Documents, LC",
  "all_dates_vs_lc_expiry_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "all_dates_vs_lc_expiry_severity": "...",
  "all_dates_vs_lc_expiry_detail": "...",

  "presentation_period_check_name": "presentation_period_check",
  "presentation_period_check_documents": "B/L, LC",
  "presentation_period_check_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "presentation_period_check_severity": "...",
  "presentation_period_check_detail": "...",

  "lc_documents_checklist_check_name": "lc_documents_checklist_check",
  "lc_documents_checklist_check_documents": "LC required list vs submitted set",
  "lc_documents_checklist_check_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "lc_documents_checklist_check_severity": "...",
  "lc_documents_checklist_check_detail": "...",

  "stale_bl_check_name": "stale_bl_check",
  "stale_bl_check_documents": "B/L, LC",
  "stale_bl_check_status": "PASS | FAIL | UNABLE TO CHECK — document missing",
  "stale_bl_check_severity": "...",
  "stale_bl_check_detail": "...",

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