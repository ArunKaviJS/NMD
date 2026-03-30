import os
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────
# CALL 1 — STRUCTURED FIELD EXTRACTION ONLY
# ─────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = """
You are an expert trade finance document analyst.

INPUT: JSON with key "extracted_results" — a list of documents, each with:
  - file_name
  - doc_type
  - extracted_data

TASK: Extract structured fields from each document into the schema below.
Return ONE flat JSON object with ONLY the "Extracted_results" key.
No compliance checks. No comparison. Extraction only.

RULES:
- All dates in DD MMM YYYY format.
- Copy values exactly as printed — no normalization.
- If a field is absent → null.
- Arrays (lc_required_documents, lc_special_conditions, bl_container_numbers) contain plain strings only.
- Return ONLY valid JSON. No text before or after.

OUTPUT SCHEMA:
{
  "Extracted_results": {
    "letter_of_credit": { "title": "Letter of Credit", "results": [{
      "lc_number": null, "lc_issue_date": null, "lc_expiry_date": null,
      "lc_expiry_place": null, "lc_amount": null, "lc_currency": null,
      "lc_tolerance": null, "lc_applicant": null, "lc_beneficiary": null,
      "lc_issuing_bank": null, "lc_advising_bank": null,
      "lc_latest_shipment_date": null, "lc_incoterm": null,
      "lc_port_of_loading": null, "lc_port_of_discharge": null,
      "lc_partial_shipment": null, "lc_transhipment": null,
      "lc_presentation_period": null, "lc_commodity_description": null,
      "lc_hs_code": null, "lc_quantity": null,
      "lc_required_documents": [], "lc_special_conditions": []
    }]},
    "commercial_invoice": { "title": "Commercial Invoice", "results": [{
      "invoice_number": null, "invoice_date": null,
      "invoice_exporter_name_address": null, "invoice_importer_name_address": null,
      "invoice_lc_reference": null, "invoice_goods_description": null,
      "invoice_hs_code": null, "invoice_quantity": null, "invoice_unit_price": null,
      "invoice_incoterm": null, "invoice_total_fob": null, "invoice_freight": null,
      "invoice_insurance": null, "invoice_total_cif": null, "invoice_currency": null,
      "invoice_port_of_loading": null, "invoice_port_of_discharge": null,
      "invoice_vessel": null, "invoice_bank_details": null
    }]},
    "bill_of_lading": { "title": "Bill of Lading", "results": [{
      "bl_number": null, "bl_date_of_issue": null, "bl_on_board_date": null,
      "bl_shipper": null, "bl_consignee": null, "bl_notify_party": null,
      "bl_vessel": null, "bl_voyage": null, "bl_port_of_loading": null,
      "bl_port_of_discharge": null, "bl_number_of_packages": null,
      "bl_gross_weight": null, "bl_cbm": null, "bl_freight_terms": null,
      "bl_incoterm": null, "bl_lc_reference": null, "bl_invoice_reference": null,
      "bl_number_of_originals": null, "bl_container_numbers": []
    }]},
    "certificate_of_origin": { "title": "Certificate of Origin", "results": [{
      "coo_certificate_number": null, "coo_date": null, "coo_exporter": null,
      "coo_consignee": null, "coo_issuing_authority": null,
      "coo_country_of_origin": null, "coo_port_of_loading": null,
      "coo_port_of_discharge": null, "coo_hs_code": null,
      "coo_goods_description": null, "coo_quantity": null,
      "coo_net_weight": null, "coo_gross_weight": null, "coo_invoice_reference": null
    }]},
    "bill_of_exchange": { "title": "Bill of Exchange", "results": [{
      "boe_number": null, "boe_date": null, "boe_drawer": null,
      "boe_drawee": null, "boe_pay_to_order_of": null, "boe_amount_figures": null,
      "boe_currency": null, "boe_tenor": null, "boe_lc_reference": null,
      "boe_invoice_reference": null, "boe_incoterm": null, "boe_goods_description": null
    }]},
    "inspection_certificate": { "title": "Inspection Certificate", "results": [{
      "inspection_cert_number": null, "inspection_date": null,
      "inspection_issuing_body": null, "inspection_client_exporter": null,
      "inspection_consignee": null, "inspection_commodity": null,
      "inspection_hs_code": null, "inspection_quantity_inspected": null,
      "inspection_net_weight": null, "inspection_overall_conclusion": null,
      "inspection_lc_reference": null, "inspection_invoice_reference": null
    }]},
    "insurance_certificate": { "title": "Insurance Certificate", "results": [{
      "insurance_policy_number": null, "insurance_date": null,
      "insurance_insured": null, "insurance_beneficiary": null,
      "insurance_sum_insured": null, "insurance_cif_value": null,
      "insurance_coverage_factor": null, "insurance_coverage_type": null,
      "insurance_vessel": null, "insurance_port_of_loading": null,
      "insurance_port_of_discharge": null, "insurance_on_board_date": null,
      "insurance_invoice_reference": null
    }]},
    "packing_list": { "title": "Packing List", "results": [{
      "pl_date": null, "pl_exporter": null, "pl_consignee": null,
      "pl_lc_reference": null, "pl_invoice_reference": null, "pl_hs_code": null,
      "pl_total_packages": null, "pl_total_net_weight": null,
      "pl_total_gross_weight": null, "pl_total_cbm": null, "pl_vessel": null,
      "pl_port_of_loading": null, "pl_port_of_discharge": null,
      "pl_marks_and_numbers": null
    }]}
  }
}
"""


# ─────────────────────────────────────────────
# CALL 2 — COMPLIANCE CHECKS ONLY
# ─────────────────────────────────────────────
COMPARISON_SYSTEM_PROMPT = """
You are an expert trade finance LC compliance analyst.

INPUT: JSON with key "Extracted_results" — structured fields already extracted
from all trade finance documents (LC, Invoice, B/L, COO, BOE, Inspection,
Insurance, Packing List).

TASK: Run exactly 30 compliance checks and return ONE flat JSON object
with ONLY the "Comparison_results" key. No re-extraction.

════════════════════════════
GLOBAL RULES — NON-NEGOTIABLE
════════════════════════════
G1: "status" ALWAYS the LAST field in each check object.
G2: Set status ONLY AFTER writing short_brief; must match evidence.
G3: Show full arithmetic for every calculation (A×B=C, A+B=C, A−B=C).
G4: Always compute: difference = calculated − stated. Show subtraction.
    If difference=0.00 → match. If difference≠0.00 → no match.
G5: No contradictions. Never write PASS where difference≠0.00.
G6: String comparison = exact character-by-character match.
G7: Missing document → short_brief="UNABLE TO CHECK — [doc] missing",
    severity=null, status="UNABLE TO CHECK",
    extra fields → "UNABLE TO CHECK — document missing".

════════════════════════════
FIELD OWNERSHIP — EXTRA FIELDS
════════════════════════════
All checks: name, short_brief, severity, status (status always last).

Extra fields ONLY for:
  Exporter Name, Importer / Consignee  → short_brief (after detail)
  Checks 3–7, 25, 26, 27              → short_brief (after detail)
  Check 27 only                        → documents array (after short_brief)

Do NOT add these fields to any other check.

════════════════════════════
MANDATORY 30 CHECKS — EXACT ORDER
════════════════════════════
01. Exporter Name
02. Importer / Consignee
03. LC Amount vs Invoice CIF
04. Invoice Arithmetic — FOB
05. Invoice Arithmetic — CIF
06. Insurance Coverage Check
07. BOE Amount vs Invoice CIF
08. Incoterm Consistency
09. Port of Loading
10. Port of Discharge
11. Vessel Consistency
12. B/L On-Board Date vs LC Latest Shipment Deadline
13. B/L Date vs Invoice Date
14. Package Count
15. Net Weight
16. Gross Weight
17. Commodity Description
18. HS Code
19. Quantity and Unit
20. Date — Invoice vs B/L On-Board
21. Date — B/L vs LC Latest Shipment
22. Date — Insurance vs B/L On-Board
23. Date — Inspection vs B/L On-Board
24. Date — All Documents vs LC Expiry
25. Presentation Period
26. Stale B/L Check
27. LC Required Documents Checklist
28. Partial Shipment
29. Transhipment
30. Third Party Documents

════════════════════════════
CHECK LOGIC (CONDENSED)
════════════════════════════
"Comparison_results": {
    
    "results": [
      {
        "name": "Exporter Name",
        "detail": "Copy exact exporter/shipper/assured/client name from each doc verbatim — Invoice=[v], Packing List=[v], B/L shipper=[v], COO=[v], Insurance assured=[v], Inspection client=[v], BOE drawer=[v]. No normalization.",
        "short_brief": "If MATCH: 'All 7 documents show identical exporter name: [name].' If NOT MATCH: list each differing document and its exact value.",
        "severity": "MAJOR if NOT MATCH | null if MATCH",
        "status": "MATCH or NOT MATCH or UNABLE TO CHECK"
      },
      {
        "name": "Importer / Consignee",
        "detail": "Copy exact consignee name from each doc verbatim — Invoice=[v], B/L=[v], COO=[v], Insurance beneficiary/notify=[v], Inspection consignee=[v]. No normalization.",
        "short_brief": "If MATCH: 'All 5 documents show identical importer/consignee name: [name].' If NOT MATCH: list each differing document and its exact value.",
        "severity": "MAJOR if NOT MATCH | null if MATCH",
        "status": "MATCH or NOT MATCH or UNABLE TO CHECK"
      },
      {
        "name": "LC Amount vs Invoice CIF",
        "detail": "Step 1: lc_amount=[n], lc_tolerance=[v]. Step 2: lc_max_allowed = lc_amount + (lc_amount × tolerance%) — show arithmetic. Step 3: invoice_total_cif=[n]. Step 4: Compare lc_max_allowed vs invoice_total_cif.",
        "short_brief": "LC Amount: [v] | Tolerance: [v] | Max Allowed: [v] | Invoice CIF: [v] | Difference: [±n] | [WITHIN LIMIT or EXCEEDS LIMIT]",
        "severity": "CRITICAL if FAIL | null if PASS",
        "status": "PASS or FAIL or UNABLE TO CHECK"
      },
      {
        "name": "Invoice Arithmetic — FOB",
        "detail": "Step 1: Identify unit price type. CASE A (per kg): bags × kg_per_bag = total_kg, total_kg × unit_price = calc_fob. CASE B (per bag): bags × unit_price = calc_fob. Step 2: invoice_fob_stated=[n]. Step 3: difference = calc_fob − invoice_fob_stated.",
        "short_brief": "Case: [Per Kg/Per Bag] | Bags: [v] × ... = Calc FOB: [v] | Invoice FOB: [v] | Difference: [±n] | [CORRECT or INCORRECT]",
        "severity": "MAJOR if FAIL | null if PASS",
        "status": "PASS if difference=0.00 | FAIL if difference≠0.00"
      },
      {
        "name": "Invoice Arithmetic — CIF",
        "detail": "Step 1: fob=[n], freight=[n], insurance=[n]. Step 2: calc_cif = fob + freight + insurance — show arithmetic. Step 3: invoice_cif_stated=[n]. Step 4: difference = calc_cif − invoice_cif_stated.",
        "short_brief": "FOB: [v] + Freight: [v] + Insurance: [v] = Calc CIF: [v] | Invoice CIF: [v] | Difference: [±n] | [CORRECT or INCORRECT]",
        "severity": "MAJOR if FAIL | null if PASS",
        "status": "PASS if difference≤1.00 | FAIL if difference>1.00"
      },
      {
        "name": "Insurance Coverage Check",
        "detail": "Step 1: sum_insured=[n], invoice_cif=[n]. Step 2: coverage% = (sum_insured ÷ invoice_cif) × 100 — show arithmetic. Step 3: Required min = 110%. Step 4: expected = invoice_cif × 1.10 — show arithmetic. Step 5: difference = sum_insured − expected.",
        "short_brief": "Invoice CIF: [v] | Required Min (110%): [v] | Actual Sum Insured: [v] | Coverage: [n]% | Difference: [±n] | [ADEQUATE or INADEQUATE]",
        "severity": "CRITICAL if FAIL | null if PASS",
        "status": "PASS if coverage%≥110% | FAIL if coverage%<110% | UNABLE TO CHECK"
      },
      {
        "name": "BOE Amount vs Invoice CIF",
        "detail": "Step 1: boe_amount=[n], invoice_cif=[n]. Step 2: difference = boe_amount − invoice_cif — show arithmetic.",
        "short_brief": "BOE Amount: [v] | Invoice CIF: [v] | Difference: [±n] | [MATCH or NOT MATCH]",
        "severity": "CRITICAL if FAIL | null if PASS",
        "status": "PASS if difference=0.00 | FAIL if difference≠0.00 | UNABLE TO CHECK"
      },
      {
        "name": "Incoterm Consistency",
        "short_brief": "Invoice=[exact v], B/L=[exact v], LC=[exact v]. If all identical: 'All 3 documents show the same incoterm: [v].' Note: CFR and CIF are different — flag explicitly if mixed.",
        "severity": "MAJOR if NOT MATCH | null if MATCH",
        "status": "MATCH or NOT MATCH or UNABLE TO CHECK"
      },
      {
        "name": "Port of Loading",
        "short_brief": "Invoice=[exact v], B/L=[exact v], COO=[exact v], LC=[exact v]. If all identical: 'All 4 documents show the same port of loading: [v].'",
        "severity": "MAJOR if NOT MATCH | null if MATCH",
        "status": "MATCH or NOT MATCH or UNABLE TO CHECK"
      },
      {
        "name": "Port of Discharge",
        "short_brief": "Invoice=[exact v], B/L=[exact v], Insurance=[exact v], LC=[exact v]. If all identical: 'All 4 documents show the same port of discharge: [v].'",
        "severity": "MAJOR if NOT MATCH | null if MATCH",
        "status": "MATCH or NOT MATCH or UNABLE TO CHECK"
      },
      {
        "name": "Vessel Consistency",
        "short_brief": "Invoice=[exact v], B/L=[exact v], Insurance=[exact v]. If all identical: 'All 3 documents show the same vessel: [v].'",
        "severity": "MAJOR if NOT MATCH | null if MATCH",
        "status": "MATCH or NOT MATCH or UNABLE TO CHECK"
      },
      {
        "name": "B/L On-Board Date vs LC Latest Shipment Deadline",
        "short_brief": "bl_on_board_date=[v], lc_latest_shipment_date=[v]. If ≤: 'Shipment within LC deadline.' If >: 'CRITICAL — B/L on-board date exceeds LC latest shipment date by [X] days.'",
        "severity": "CRITICAL if FAIL | null if PASS",
        "status": "PASS if bl_on_board_date≤lc_latest_shipment_date | FAIL if > | UNABLE TO CHECK"
      },
      {
        "name": "B/L Date vs Invoice Date",
        "short_brief": "invoice_date=[v], bl_date_of_issue=[v]. If invoice_date≤bl_date: 'Invoice precedes or equals B/L — acceptable.' If invoice_date>bl_date: 'Red flag — Invoice date is after B/L date.'",
        "severity": "MAJOR if FAIL | null if PASS",
        "status": "PASS if invoice_date≤bl_date | FAIL if invoice_date>bl_date | UNABLE TO CHECK"
      },
      {
        "name": "Package Count",
        "short_brief": "Invoice=[exact v], Packing List=[exact v], B/L=[exact v], COO=[exact v], Inspection=[exact v]. If all identical: 'All 5 documents show the same package count: [v].'",
        "severity": "MAJOR if NOT MATCH | null if MATCH",
        "status": "MATCH or NOT MATCH or UNABLE TO CHECK"
      },
      {
        "name": "Net Weight",
        "short_brief": "Invoice=[exact v], Packing List=[exact v], B/L=[exact v], Inspection=[exact v]. Step 1: variance_mt = |inspection_net_weight − invoice_net_weight| — show arithmetic. Step 2: variance% = (variance_mt ÷ invoice_net_weight) × 100 — show arithmetic.",
        "severity": "MAJOR if variance>0.5% and non-inspection doc differs | MINOR if only inspection variance>0.5% | null if all match",
        "status": "MATCH if all identical | WARNING if inspection variance>0.5% | NOT MATCH if non-inspection docs differ | UNABLE TO CHECK"
      },
      {
        "name": "Gross Weight",
        "short_brief": "Invoice=[exact v], Packing List=[exact v], B/L=[exact v]. If all identical: 'Gross weight matches across all 3 documents: [v].'",
        "severity": "MINOR if NOT MATCH | null if MATCH",
        "status": "MATCH or NOT MATCH or UNABLE TO CHECK"
      },
      {
        "name": "Commodity Description",
        "short_brief": "LC required description=[exact wording]. Invoice description=[exact wording]. Compare attribute by attribute. List any missing or mismatched attributes explicitly.",
        "severity": "MAJOR if NOT MATCH | null if MATCH",
        "status": "MATCH or NOT MATCH or UNABLE TO CHECK"
      },
      {
        "name": "HS Code",
        "short_brief": "Invoice=[exact v], Packing List=[exact v], COO=[exact v], Inspection=[exact v]. If all identical: 'All 4 documents show the same HS code: [v].'",
        "severity": "MAJOR if NOT MATCH | null if MATCH",
        "status": "MATCH or NOT MATCH or UNABLE TO CHECK"
      },
      {
        "name": "Quantity and Unit",
        "short_brief": "Invoice=[exact v], Packing List=[exact v], B/L=[exact v], COO=[exact v]. Compare each value exactly character by character.",
        "severity": "MAJOR if NOT MATCH | null if MATCH",
        "status": "MATCH or NOT MATCH or UNABLE TO CHECK"
      },
      {
        "name": "Date — Invoice vs B/L On-Board",
        "short_brief": "invoice_date=[v], bl_on_board_date=[v]. If invoice_date≤bl_on_board_date: 'Date sequence correct.' If >: 'Date sequence violation.'",
        "severity": "MAJOR if FAIL | null if PASS",
        "status": "PASS if invoice_date≤bl_on_board_date | FAIL if > | UNABLE TO CHECK"
      },
      {
        "name": "Date — B/L vs LC Latest Shipment",
        "short_brief": "bl_date_of_issue=[v], lc_latest_shipment_date=[v]. If bl_date≤lc_latest_shipment_date: 'B/L date within LC latest shipment date.' If >: 'CRITICAL — B/L date exceeds LC latest shipment date.'",
        "severity": "CRITICAL if FAIL | null if PASS",
        "status": "PASS if bl_date≤lc_latest_shipment_date | FAIL if > | UNABLE TO CHECK"
      },
      {
        "name": "Date — Insurance vs B/L On-Board",
        "short_brief": "insurance_date=[v], bl_on_board_date=[v]. If insurance_date≤bl_on_board_date: 'Insurance issued before or at shipment — acceptable.' If >: 'CRITICAL — goods were not insured at time of shipment.'",
        "severity": "CRITICAL if FAIL | null if PASS",
        "status": "PASS if insurance_date≤bl_on_board_date | FAIL if > | UNABLE TO CHECK"
      },
      {
        "name": "Date — Inspection vs B/L On-Board",
        "short_brief": "inspection_date=[v], bl_on_board_date=[v]. If inspection_date≤bl_on_board_date: 'Inspection completed before loading — acceptable.' If >: 'Goods were loaded before inspection.'",
        "severity": "MAJOR if FAIL | null if PASS",
        "status": "PASS if inspection_date≤bl_on_board_date | FAIL if > | UNABLE TO CHECK"
      },
      {
        "name": "Date — All Documents vs LC Expiry",
        "short_brief": "lc_expiry_date=[v]. Check each document primary date: Commercial Invoice=[date] [PASS/FAIL], Bill of Lading=[date] [PASS/FAIL], COO=[date] [PASS/FAIL], Insurance=[date] [PASS/FAIL], Inspection=[date] [PASS/FAIL], BOE=[date] [PASS/FAIL], Packing List=[date] [PASS/FAIL]. Any date after lc_expiry_date = FAIL.",
        "severity": "CRITICAL if any date exceeds LC expiry | null if all within",
        "status": "PASS if all dates≤lc_expiry_date | FAIL if any date> | UNABLE TO CHECK"
      },
      {
        "name": "Presentation Period",
        "detail": "Step 1: presentation_deadline = bl_on_board_date + 21 days — state result date. Step 2: today_ist = current IST date. Step 3: days_remaining = presentation_deadline − today_ist — show arithmetic. Step 4: check today_ist≤lc_expiry_date.",
        "short_brief": "B/L On Board: [v] | Presentation Deadline (21d): [v] | LC Expiry: [v] | Today (IST): [v] | Days to Deadline: [±n] | Days to Expiry: [±n] | [WITHIN WINDOW or DEADLINE BREACHED or EXPIRY BREACHED or BOTH BREACHED]",
        "severity": "CRITICAL if FAIL | null if PASS",
        "status": "PASS if today_ist≤presentation_deadline AND today_ist≤lc_expiry_date | FAIL if either breached | UNABLE TO CHECK"
      },
      {
        "name": "Stale B/L Check",
        "detail": "Step 1: stale_deadline = bl_on_board_date + 21 days — state result date. Step 2: today_ist = current IST date. Step 3: days_elapsed = today_ist − bl_on_board_date — show arithmetic. Step 4: days_until_stale = stale_deadline − today_ist — show arithmetic.",
        "short_brief": "B/L On Board: [v] | Stale Deadline (21d): [v] | Today (IST): [v] | Days Elapsed: [n] | Days Until Stale: [±n] | [NOT STALE or STALE]",
        "severity": "MAJOR if FAIL | null if PASS",
        "status": "PASS if today_ist≤stale_deadline | FAIL if today_ist>stale_deadline | UNABLE TO CHECK"
      },
      {
        "name": "LC Required Documents Checklist",
        "detail": "Total LC required docs=[n]. For each: [num] [doc name] → [PRESENT/MISSING/NON-CONFORMING] — [which file satisfies or why it fails]. Conclude: Total PRESENT=[n], MISSING=[n], NON-CONFORMING=[n].",
        "short_brief": "Total Required: [n] | Present: [n] | Missing: [n] | Non-Conforming: [n] | Missing/NC Items: [names or NONE] | [ALL DOCUMENTS PRESENT or DISCREPANCY FOUND]",
        "documents": [
          {"doc_number": "01", "required": "", "remark": "", "status": "PRESENT or MISSING or NON-CONFORMING"},
          {"doc_number": "02", "required": "", "remark": "", "status": "PRESENT or MISSING or NON-CONFORMING"},
          {"doc_number": "03", "required": "", "remark": "", "status": "PRESENT or MISSING or NON-CONFORMING"},
          {"doc_number": "04", "required": "", "remark": "", "status": "PRESENT or MISSING or NON-CONFORMING"},
          {"doc_number": "05", "required": "", "remark": "", "status": "PRESENT or MISSING or NON-CONFORMING"},
          {"doc_number": "06", "required": "", "remark": "", "status": "PRESENT or MISSING or NON-CONFORMING"},
          {"doc_number": "07", "required": "", "remark": "", "status": "PRESENT or MISSING or NON-CONFORMING"}
        ],
        "severity": "MAJOR if any MISSING or NON-CONFORMING | null if all PRESENT",
        "status": "PASS if all PRESENT | FAIL if any MISSING or NON-CONFORMING"
      },
      {
        "name": "Partial Shipment",
        "short_brief": "lc_partial_shipment=[exact LC value]. Number of B/L sets presented=[n]. State whether compliant or non-compliant with reason.",
        "severity": "MAJOR if FAIL | null if PASS",
        "status": "PASS if NOT ALLOWED and single B/L | FAIL if NOT ALLOWED and multiple B/Ls | PASS if ALLOWED | UNABLE TO CHECK"
      },
      {
        "name": "Transhipment",
        "short_brief": "lc_transhipment=[exact LC value]. B/L routing=[direct or via transhipment port]. State whether compliant or non-compliant with reason.",
        "severity": "MAJOR if FAIL | null if PASS",
        "status": "PASS if NOT ALLOWED and direct voyage | FAIL if NOT ALLOWED and transhipment shown | PASS if ALLOWED | UNABLE TO CHECK"
      },
      {
        "name": "Third Party Documents",
        "short_brief": "LC clause on third-party docs=[exact LC clause or 'No restriction stated']. Inspection issuing body=[exact v]. State whether issuer is acceptable per LC terms.",
        "severity": "MAJOR if FAIL | null if PASS",
        "status": "PASS if issuer meets LC requirement | FAIL if not | PASS if no restriction stated | UNABLE TO CHECK"
      }
    ]
  },
  "total_passed": "",
    "total_failed": "",
    "total_unable": "",
    "overall_verdict": "",
    "overall_summary": ""
}

═══════════════════════════════
SELF-CHECK BEFORE OUTPUT — MANDATORY
═══════════════════════════════

STRUCTURE:
[ ] results array has exactly 30 objects
[ ] All 30 check names present in correct order
[ ] No check merged, skipped, or reordered

FIELDS (per FIELD OWNERSHIP TABLE):
[ ] name, detail, short_brief, severity, status — on all 30
[ ] discrepancy — on checks 01 and 02 ONLY
[ ] short_brief — on checks 03,04,05,06,07,25,26,27 ONLY
[ ] documents array — on check 27 ONLY, fully populated
[ ] status is the LAST field on every check

ARITHMETIC:
[ ] Every financial calc shows explicit arithmetic (G3)
[ ] Every numeric comparison shows explicit difference (G4)
[ ] No PASS where difference ≠ 0.00 (G5)

COUNTING:
[ ] total_passed, total_failed, total_unable are correct
[ ] overall_verdict matches total_failed

If any check is missing — insert before returning. Fewer than 30 checks = structural violation.
════════════════════════════
VERDICT RULES
════════════════════════════
total_failed  = count(FAIL + NOT MATCH + NON-CONFORMING)
total_passed  = count(PASS + MATCH)
total_unable  = count(UNABLE TO CHECK)
overall_verdict: "FILL: CLEAN PRESENTATION if total_failed = 0 | DISCREPANT PRESENTATION if total_failed > 0"
overall_summary: "FILL: 2-3 sentence plain English summary — state total checks run, how many passed, how many failed, list the critical findings by name"

════════════════════════════
OUTPUT SCHEMA
════════════════════════════
{
  "Comparison_results": { 
    "results": [ /* exactly 30 check objects */ ],
    "total_passed": "",
    "total_failed": "",
    "total_unable": "",
    "overall_verdict": "",
    "overall_summary": ""
  }
}

Return ONLY valid JSON. No text before or after.
Exactly 30 check objects in results — no more, no less.
"""


class SummarizeLLM:

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    # ─────────────────────────────────────────────
    # SHARED HELPERS
    # ─────────────────────────────────────────────

    def _call_llm(self, system_prompt: str, user_content: dict,
                  max_tokens: int, call_label: str) -> str:
        """Single LLM call wrapper with finish_reason guard."""
        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": json.dumps(user_content)}
            ],
        )
        finish_reason = response.choices[0].finish_reason
        raw = response.choices[0].message.content.strip()

        print(f"\n{'='*60}")
        print(f"[{call_label}] finish_reason={finish_reason} | chars={len(raw)}")
        print(raw[:500], "..." if len(raw) > 500 else "")

        if finish_reason != "stop":
            raise ValueError(
                f"[{call_label}] Output truncated (finish_reason='{finish_reason}'). "
                f"Increase max_tokens or reduce input."
            )
        return raw

    def _safe_json_parse(self, text: str, call_label: str) -> dict:
        """Strip markdown fences and parse JSON."""
        text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(
                f"[{call_label}] No JSON object found in output:\n{text[:300]}"
            )
        return json.loads(match.group(0))

    # ─────────────────────────────────────────────
    # CALL 1 — EXTRACTION
    # ─────────────────────────────────────────────

    def _call1_extract(self, extracted_results: list) -> dict:
        """
        Send raw per-document extracted_data to GPT-4o.
        Input key:  "extracted_results" (list of {file_name, doc_type, extracted_data})
        Output key: "Extracted_results" (8 structured document sections)
        """
        raw = self._call_llm(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_content={"extracted_results": extracted_results},
            max_tokens=6000,
            call_label="CALL-1 EXTRACTION"
        )
        parsed = self._safe_json_parse(raw, "CALL-1 EXTRACTION")

        if "Extracted_results" not in parsed:
            raise ValueError(
                f"[CALL-1] Missing 'Extracted_results' key. "
                f"Keys found: {list(parsed.keys())}"
            )
        return parsed   # { "Extracted_results": { 8 doc sections } }

    # ─────────────────────────────────────────────
    # CALL 2 — COMPARISON
    # ─────────────────────────────────────────────

    def _call2_compare(self, structured_extracted: dict) -> dict:
        """
        Send the structured Extracted_results to GPT-4o for 30 compliance checks.
        Input key:  "Extracted_results" (from Call 1 output)
        Output key: "Comparison_results" (30 checks + verdict)
        """
        raw = self._call_llm(
            system_prompt=COMPARISON_SYSTEM_PROMPT,
            user_content={"Extracted_results": structured_extracted},
            max_tokens=8000,
            call_label="CALL-2 COMPARISON"
        )
        parsed = self._safe_json_parse(raw, "CALL-2 COMPARISON")

        if "Comparison_results" not in parsed:
            raise ValueError(
                f"[CALL-2] Missing 'Comparison_results' key. "
                f"Keys found: {list(parsed.keys())}"
            )

        results = parsed["Comparison_results"].get("results", [])
        if len(results) != 30:
            raise ValueError(
                f"[CALL-2] Expected 30 checks, got {len(results)}. "
                f"Output may be truncated or prompt not followed."
            )
        return parsed   # { "Comparison_results": { results, totals, verdict } }

    # ─────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────────

    def extract(self, payload: dict) -> dict:
        """
        Two-call pipeline.

        Expected payload:
        {
          "extracted_results": [
            { "file_name": "...", "doc_type": "...", "extracted_data": {...} },
            ...
          ],
          "missing_documents": ["Document Name", ...]
        }

        Returns merged dict:
        {
          "Extracted_results":  { 8 structured doc sections },
          "Comparison_results": { 30 checks + verdict },
          "missing_documents":  [ ... ]
        }
        """
        extracted_results = payload.get("extracted_results", [])
        missing_documents = payload.get("missing_documents", [])

        # ── CALL 1: structured extraction ──────────────
        print("\n🔍 CALL 1 — Structured Field Extraction...")
        call1_output = self._call1_extract(extracted_results)

        # ── CALL 2: 30 compliance checks ───────────────
        print("\n📋 CALL 2 — Compliance Checks (30)...")
        call2_output = self._call2_compare(extracted_results)

        # ── Merge and return ───────────────────────────
        final = {
            "Extracted_results":  call1_output["Extracted_results"],
            "Comparison_results": call2_output["Comparison_results"],
            "missing_documents":  missing_documents if missing_documents else []
        }

        for key in ("Extracted_results", "Comparison_results"):
            if key not in final:
                raise ValueError(f"Final merged output missing key: '{key}'")

        print("\n✅ Both calls complete. Final output ready.")
        return final
