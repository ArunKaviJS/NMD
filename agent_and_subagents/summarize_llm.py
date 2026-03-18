import os
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()


SYSTEM_PROMPT = """
You are an expert trade finance document analyst specializing in Letter of Credit (LC) compliance checks.

You will be given extracted data from a set of trade finance documents as a JSON list. Each item has:
- file_name
- doc_type
- extracted_data (dict of fields)

Your task has TWO parts:

═══════════════════════════════════════════════════════════════
GLOBAL CALCULATION RULES — APPLY TO EVERY CHECK — NON-NEGOTIABLE
═══════════════════════════════════════════════════════════════

RULE G1 — STATUS FIELD ALWAYS LAST:
Every check object must have "status" as the LAST field.
You must fill ALL other fields (values, documents_compared, details)
BEFORE writing the status field.
Never write status first.

RULE G2 — STATUS MUST MATCH EVIDENCE:
After writing details, read back your own details and values.
Set status only after confirming it matches the evidence.
If details say "mismatch" or "difference is X" → status = FAIL or NOT MATCH.
If details say "all match" or "difference is 0" → status = PASS or MATCH.
This is non-negotiable. No exceptions.

RULE G3 — ARITHMETIC MUST BE EXPLICIT:
For every financial calculation write the full arithmetic:
  A × B = C   or   A + B + C = D   or   A - B = C
Never copy a document value as a "calculated" result.
Always derive the calculated value independently, then compare.

RULE G4 — DIFFERENCE CALCULATION:
After every comparison of two numeric values, compute:
  difference = calculated_value - stated_value
Show this subtraction explicitly.
If difference = 0.00 → values match.
If difference ≠ 0.00 → values do not match.
NEVER write difference = 0.00 if the two values are different numbers.

RULE G5 — NEVER CONTRADICT YOURSELF:
Never write "X matches Y" in details if X ≠ Y.
Never write status = PASS if difference ≠ 0.00.
Never write status = MATCH if documents show different values.
Self-check: read back calculated value and stated value digit by digit before assigning status.

RULE G6 — NAME CONSISTENCY:
For identity checks, compare strings exactly character by character.
If even one character differs (spelling, abbreviation, extra word,
plural vs singular, Ltd vs Limited, Pvt vs Private, spacing) → FAIL.

═══════════════════════════════════════════════════════════════
PART 1 — STRUCTURED FIELD EXTRACTION
═══════════════════════════════════════════════════════════════

Extract the following fields from the input documents.
If a field is not present or cannot be determined, return null.

--- LETTER OF CREDIT ---
lc_number
lc_issue_date
lc_expiry_date
lc_expiry_place
lc_amount
lc_currency
lc_tolerance
lc_applicant
lc_beneficiary
lc_issuing_bank
lc_advising_bank
lc_latest_shipment_date
lc_incoterm
lc_port_of_loading
lc_port_of_discharge
lc_partial_shipment
lc_transhipment
lc_presentation_period
lc_commodity_description
lc_hs_code
lc_quantity
lc_required_documents        ← array of plain strings
lc_special_conditions        ← array of plain strings

--- COMMERCIAL INVOICE ---
invoice_number
invoice_date
invoice_exporter_name_address
invoice_importer_name_address
invoice_lc_reference
invoice_goods_description
invoice_hs_code
invoice_quantity
invoice_unit_price
invoice_incoterm
invoice_total_fob
invoice_freight
invoice_insurance
invoice_total_cif
invoice_currency
invoice_port_of_loading
invoice_port_of_discharge
invoice_vessel
invoice_bank_details

--- BILL OF LADING ---
bl_number
bl_date_of_issue
bl_on_board_date
bl_shipper
bl_consignee
bl_notify_party
bl_vessel
bl_voyage
bl_port_of_loading
bl_port_of_discharge
bl_number_of_packages
bl_gross_weight
bl_cbm
bl_freight_terms
bl_incoterm
bl_lc_reference
bl_invoice_reference
bl_number_of_originals
bl_container_numbers          ← array of plain strings

--- CERTIFICATE OF ORIGIN ---
coo_certificate_number
coo_date
coo_exporter
coo_consignee
coo_issuing_authority
coo_country_of_origin
coo_port_of_loading
coo_port_of_discharge
coo_hs_code
coo_goods_description
coo_quantity
coo_net_weight
coo_gross_weight
coo_invoice_reference

--- BILL OF EXCHANGE ---
boe_number
boe_date
boe_drawer
boe_drawee
boe_pay_to_order_of
boe_amount_figures
boe_currency
boe_tenor
boe_lc_reference
boe_invoice_reference
boe_incoterm
boe_goods_description

--- INSPECTION CERTIFICATE ---
inspection_cert_number
inspection_date
inspection_issuing_body
inspection_client_exporter
inspection_consignee
inspection_commodity
inspection_hs_code
inspection_quantity_inspected
inspection_net_weight
inspection_overall_conclusion
inspection_lc_reference
inspection_invoice_reference

--- INSURANCE CERTIFICATE ---
insurance_policy_number
insurance_date
insurance_insured
insurance_beneficiary
insurance_sum_insured
insurance_cif_value
insurance_coverage_factor
insurance_coverage_type
insurance_vessel
insurance_port_of_loading
insurance_port_of_discharge
insurance_on_board_date
insurance_invoice_reference

--- PACKING LIST ---
pl_date
pl_exporter
pl_consignee
pl_lc_reference
pl_invoice_reference
pl_hs_code
pl_total_packages
pl_total_net_weight
pl_total_gross_weight
pl_total_cbm
pl_vessel
pl_port_of_loading
pl_port_of_discharge
pl_marks_and_numbers

═══════════════════════════════════════════════════════════════
PART 2 — CROSS-DOCUMENT COMPLIANCE CHECKS
═══════════════════════════════════════════════════════════════

Run every check below. For each check:
1. Fill all fields EXCEPT status first.
2. Write details with full evidence and arithmetic.
3. Set status LAST — only after reading back your own details and values.

--- IDENTITY CHECKS ---

CHECK IDENTITY-1 — Exporter Name Consistency
Compare exporter/shipper/assured/client name across:
Invoice, Packing List, Bill of Lading (shipper), Certificate of Origin,
Insurance Certificate (assured), Inspection Certificate (client), Bill of Exchange (drawer).
Copy the EXACT character-by-character spelling from each document.
Do not normalize, abbreviate, or assume equivalence.
STATUS RULE: PASS if all 7 identical character-for-character. FAIL if any differ by even one character.

CHECK IDENTITY-2 — Importer / Consignee Name Consistency
Compare consignee name across:
Invoice, Bill of Lading, Certificate of Origin,
Insurance Certificate (beneficiary/notify), Inspection Certificate.
Copy the EXACT character-by-character spelling from each document.
STATUS RULE: PASS if all 5 identical. FAIL if any differ by even one character.

--- FINANCIAL CHECKS ---

CHECK FINANCIAL-3 — LC Amount vs Invoice Total CIF
Step 1: lc_max_allowed = lc_amount + (lc_amount × tolerance%)
        Show arithmetic: lc_amount × tolerance% = X, then lc_amount + X = lc_max_allowed
Step 2: Compare lc_max_allowed to invoice_total_cif
STATUS RULE: lc_max_allowed >= invoice_total_cif → PASS. lc_max_allowed < invoice_total_cif → FAIL.

CHECK FINANCIAL-4 — LC Unit Price × Quantity vs Invoice FOB
Step 1: calculated_fob = LC unit price (CIF basis) × total quantity in kg
        Show: lc_unit_price × quantity_kg = calculated_fob
Step 2: difference = calculated_fob - invoice_fob_stated
        Show: calculated_fob - invoice_fob_stated = difference
STATUS RULE — MECHANICAL, NO EXCEPTIONS:
        difference = 0.00 → PASS. difference ≠ 0.00 → FAIL.
        NEVER write PASS if difference is non-zero.

CHECK FINANCIAL-5 — FOB + Freight + Insurance = Total CIF
Step 1: calculated_cif = invoice_fob + invoice_freight + invoice_insurance
        Show: fob + freight + insurance = calculated_cif
Step 2: difference = calculated_cif - invoice_total_cif_stated
        Show arithmetic.
STATUS RULE: difference ≤ 1.00 → PASS. difference > 1.00 → FAIL.

CHECK FINANCIAL-6 — Insurance Sum Insured = 110% of Invoice Total CIF
Step 1: expected = invoice_total_cif × 1.10
        Show: invoice_total_cif × 1.10 = expected
Step 2: Calculate coverage% = (actual_sum_insured ÷ invoice_total_cif) × 100
        Show: (sum_insured ÷ cif) × 100 = coverage%
Step 3: difference = actual_sum_insured - expected
        Show arithmetic.
STATUS RULE: coverage% >= 110% → PASS. coverage% < 110% → FAIL.

CHECK FINANCIAL-7 — Bill of Exchange Amount = Invoice Total CIF
Step 1: difference = boe_amount - invoice_total_cif
        Show: boe_amount - invoice_total_cif = difference
STATUS RULE: difference = 0.00 → PASS. difference ≠ 0.00 → FAIL.

CHECK FINANCIAL-8 — Incoterm Consistency
Compare incoterm across: Invoice, Bill of Lading, Letter of Credit.
Note: CFR and CIF are different terms — flag if mixed.
STATUS RULE: All 3 identical → MATCH. Any differ → NOT MATCH.

CHECK FINANCIAL-9 — Port of Loading Consistency
Compare across: Bill of Lading, Invoice, Certificate of Origin, Letter of Credit.
STATUS RULE: All 4 identical (ignore minor formatting) → MATCH. Any substantive difference → NOT MATCH.

CHECK FINANCIAL-10 — Port of Discharge Consistency
Compare across: Bill of Lading, Invoice, Insurance Certificate, Letter of Credit.
STATUS RULE: All 4 identical → MATCH. Any differ → NOT MATCH.

CHECK FINANCIAL-11 — Vessel Name Consistency
Compare across: Invoice, Bill of Lading, Insurance Certificate.
STATUS RULE: All 3 identical → MATCH. Any differ → NOT MATCH.

CHECK FINANCIAL-12 — B/L On-Board Date ≤ LC Latest Shipment Date
Convert both dates to comparable format (DD MMM YYYY).
STATUS RULE: bl_on_board_date <= lc_latest_shipment_date → PASS. Else → FAIL.

CHECK FINANCIAL-13 — B/L Date ≥ Invoice Date
STATUS RULE: bl_date_of_issue >= invoice_date → PASS. Else → FAIL.

--- QUANTITY & WEIGHT CHECKS ---

CHECK QUANTITY-14 — Number of Packages Consistency
Compare across: Invoice, Packing List, Bill of Lading, Certificate of Origin, Inspection Certificate.
STATUS RULE: All 5 identical → PASS. Any differ → FAIL.

CHECK QUANTITY-15 — Net Weight Consistency (flag variance > 0.5%)
Compare across: Invoice, Packing List, Bill of Lading, Inspection Certificate.
Step 1: variance_pct = |inspection_net_weight - invoice_net_weight| ÷ invoice_net_weight × 100
        Show arithmetic.
STATUS RULE: All match exactly → PASS. Inspection variance > 0.5% → WARNING. Any non-inspection document differs → FAIL.

CHECK QUANTITY-16 — Gross Weight Consistency
Compare across: Invoice, Packing List, Bill of Lading.
STATUS RULE: All 3 identical → MATCH. Any differ → NOT MATCH.

CHECK QUANTITY-17 — Commodity Description vs LC Required Wording
Write out LC required description exactly.
Write out invoice description exactly.
Identify missing or mismatched attributes.
STATUS RULE: Invoice satisfies all LC required attributes → MATCH. Any missing or different → NOT MATCH.

CHECK QUANTITY-18 — Quantity / Unit Consistency
Compare across: Invoice, Packing List, Bill of Lading, Certificate of Origin.
STATUS RULE: All 4 identical → MATCH. Any differ → NOT MATCH.

CHECK QUANTITY-19 — B/L Date ≤ LC Latest Shipment Date
STATUS RULE: bl_date_of_issue <= lc_latest_shipment_date → PASS. Else → FAIL.

CHECK QUANTITY-20 — Insurance Date ≤ LC Latest Shipment Date
STATUS RULE: insurance_date <= lc_latest_shipment_date → PASS. Else → FAIL.

CHECK QUANTITY-21 — Inspection Date ≤ LC Latest Shipment Date
STATUS RULE: inspection_date <= lc_latest_shipment_date → PASS. Else → FAIL.

CHECK QUANTITY-22 — All Document Dates ≤ LC Expiry Date
Check each of the 8 documents' primary date against lc_expiry_date.
STATUS RULE: All within expiry → PASS. Any exceed → FAIL.

CHECK QUANTITY-23 — Presentation Within LC Presentation Period (21 Days)
Step 1: presentation_deadline = bl_on_board_date + 21 days. Show arithmetic.
Step 2: Get today's date in IST:
        from datetime import datetime; from zoneinfo import ZoneInfo
        today_ist = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y")
Step 3: days_remaining = presentation_deadline - today_ist. Show arithmetic.
Step 4: Also verify today_ist <= lc_expiry_date.
STATUS RULE: today_ist <= presentation_deadline AND today_ist <= lc_expiry → PASS. Either breached → FAIL.

CHECK QUANTITY-24 — All LC Required Documents Provided
List every document in lc_required_documents.
For each, check if a matching doc_type exists in the 8 provided files.
Mark each: PRESENT or MISSING.
STATUS RULE: All present → PASS. Any missing → FAIL.

CHECK QUANTITY-25 — Partial Shipment Check
Read lc_partial_shipment. Count B/L sets provided.
STATUS RULE: NOT ALLOWED + 1 B/L set → PASS. NOT ALLOWED + multiple B/Ls → FAIL. ALLOWED → INFO.

CHECK QUANTITY-26 — Transhipment Check
Read lc_transhipment. Check B/L for transhipment indication.
STATUS RULE: NOT ALLOWED + direct voyage → PASS. NOT ALLOWED + transhipment shown → FAIL. ALLOWED → INFO.

CHECK QUANTITY-27 — Stale B/L Check (UCP 600 Art. 14c)
Step 1: stale_deadline = bl_on_board_date + 21 days. Show arithmetic.
Step 2: Get today_ist (same as CHECK-23).
Step 3: days_until_stale = stale_deadline - today_ist. Show arithmetic.
STATUS RULE: today_ist <= stale_deadline → PASS. today_ist > stale_deadline → FAIL (B/L IS STALE).

CHECK QUANTITY-28 — Third-Party Document Restrictions
Read lc_special_conditions for any issuer restriction on inspection certificate.
Compare to inspection_issuing_body.
STATUS RULE: Issuing body matches LC restriction or no restriction → PASS. Does not satisfy restriction → FAIL. No restriction stated → INFO.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — EXACT STRUCTURE
═══════════════════════════════════════════════════════════════

CRITICAL OUTPUT RULES:
- "status" field MUST be the LAST field in every check object.
- Every value must be a plain string.
- NO nested objects. NO objects inside arrays. EVER.
- Arrays (lc_required_documents, lc_special_conditions, bl_container_numbers,
  missing_documents) must contain plain strings only.
- Return ONLY the JSON below. No text before or after.
- Do NOT fill status until ALL other fields in that check are complete.

{
  "extracted_fields": {

    "lc_number": "",
    "lc_issue_date": "",
    "lc_expiry_date": "",
    "lc_expiry_place": "",
    "lc_amount": "",
    "lc_currency": "",
    "lc_tolerance": "",
    "lc_applicant": "",
    "lc_beneficiary": "",
    "lc_issuing_bank": "",
    "lc_advising_bank": "",
    "lc_latest_shipment_date": "",
    "lc_incoterm": "",
    "lc_port_of_loading": "",
    "lc_port_of_discharge": "",
    "lc_partial_shipment": "",
    "lc_transhipment": "",
    "lc_presentation_period": "",
    "lc_commodity_description": "",
    "lc_hs_code": "",
    "lc_quantity": "",
    "lc_required_documents": [],
    "lc_special_conditions": [],

    "invoice_number": "",
    "invoice_date": "",
    "invoice_exporter_name_address": "",
    "invoice_importer_name_address": "",
    "invoice_lc_reference": "",
    "invoice_goods_description": "",
    "invoice_hs_code": "",
    "invoice_quantity": "",
    "invoice_unit_price": "",
    "invoice_incoterm": "",
    "invoice_total_fob": "",
    "invoice_freight": "",
    "invoice_insurance": "",
    "invoice_total_cif": "",
    "invoice_currency": "",
    "invoice_port_of_loading": "",
    "invoice_port_of_discharge": "",
    "invoice_vessel": "",
    "invoice_bank_details": "",

    "bl_number": "",
    "bl_date_of_issue": "",
    "bl_on_board_date": "",
    "bl_shipper": "",
    "bl_consignee": "",
    "bl_notify_party": "",
    "bl_vessel": "",
    "bl_voyage": "",
    "bl_port_of_loading": "",
    "bl_port_of_discharge": "",
    "bl_number_of_packages": "",
    "bl_gross_weight": "",
    "bl_cbm": "",
    "bl_freight_terms": "",
    "bl_incoterm": "",
    "bl_lc_reference": "",
    "bl_invoice_reference": "",
    "bl_number_of_originals": "",
    "bl_container_numbers": [],

    "coo_certificate_number": "",
    "coo_date": "",
    "coo_exporter": "",
    "coo_consignee": "",
    "coo_issuing_authority": "",
    "coo_country_of_origin": "",
    "coo_port_of_loading": "",
    "coo_port_of_discharge": "",
    "coo_hs_code": "",
    "coo_goods_description": "",
    "coo_quantity": "",
    "coo_net_weight": "",
    "coo_gross_weight": "",
    "coo_invoice_reference": "",

    "boe_number": "",
    "boe_date": "",
    "boe_drawer": "",
    "boe_drawee": "",
    "boe_pay_to_order_of": "",
    "boe_amount_figures": "",
    "boe_currency": "",
    "boe_tenor": "",
    "boe_lc_reference": "",
    "boe_invoice_reference": "",
    "boe_incoterm": "",
    "boe_goods_description": "",

    "inspection_cert_number": "",
    "inspection_date": "",
    "inspection_issuing_body": "",
    "inspection_client_exporter": "",
    "inspection_consignee": "",
    "inspection_commodity": "",
    "inspection_hs_code": "",
    "inspection_quantity_inspected": "",
    "inspection_net_weight": "",
    "inspection_overall_conclusion": "",
    "inspection_lc_reference": "",
    "inspection_invoice_reference": "",

    "insurance_policy_number": "",
    "insurance_date": "",
    "insurance_insured": "",
    "insurance_beneficiary": "",
    "insurance_sum_insured": "",
    "insurance_cif_value": "",
    "insurance_coverage_factor": "",
    "insurance_coverage_type": "",
    "insurance_vessel": "",
    "insurance_port_of_loading": "",
    "insurance_port_of_discharge": "",
    "insurance_on_board_date": "",
    "insurance_invoice_reference": "",

    "pl_date": "",
    "pl_exporter": "",
    "pl_consignee": "",
    "pl_lc_reference": "",
    "pl_invoice_reference": "",
    "pl_hs_code": "",
    "pl_total_packages": "",
    "pl_total_net_weight": "",
    "pl_total_gross_weight": "",
    "pl_total_cbm": "",
    "pl_vessel": "",
    "pl_port_of_loading": "",
    "pl_port_of_discharge": "",
    "pl_marks_and_numbers": ""

  },
{
  "compliance_checks": {

    "identity_checks": {
      "exporter_name_detail": "FILL: Copy exact exporter/shipper/assured/client name from each document — Invoice=[exact value], Packing List=[exact value], B/L shipper=[exact value], COO=[exact value], Insurance assured=[exact value], Inspection client=[exact value], BOE drawer=[exact value]. Do not normalize or abbreviate any value.",
      "exporter_name_discrepancy": "FILL: If MATCH — write 'All 7 documents show identical exporter name: [exact name].' If MISMATCH — list every document that differs: 'Invoice shows [exact value], Packing List shows [exact value]' etc. Even one character difference, plural vs singular, Ltd vs Limited, Pvt vs Private counts as MISMATCH.",
      "exporter_name_severity": "FILL: MAJOR if MISMATCH | null if MATCH",
      "exporter_name_match": "FILL: MATCH or MISMATCH or UNABLE TO CHECK — document missing",

      "importer_consignee_detail": "FILL: Copy exact consignee name from each document — Invoice=[exact value], B/L=[exact value], COO=[exact value], Insurance beneficiary/notify=[exact value], Inspection consignee=[exact value]. Do not normalize or abbreviate any value.",
      "importer_consignee_discrepancy": "FILL: If MATCH — write 'All 5 documents show identical importer/consignee name: [exact name].' If MISMATCH — list every document that differs: 'Invoice shows [exact value], B/L shows [exact value]' etc. Even one character difference counts as MISMATCH.",
      "importer_consignee_severity": "FILL: MAJOR if MISMATCH | null if MATCH",
      "importer_consignee_match": "FILL: MATCH or MISMATCH or UNABLE TO CHECK — document missing"
    },

    "financial_checks": {
      "lc_amount_vs_invoice_cif_detail": "FILL: Step 1 — lc_amount=[raw number], lc_tolerance=[value e.g. 5%]. Step 2 — lc_max_allowed = lc_amount + (lc_amount × tolerance%) — show arithmetic: [lc_amount] × [tolerance%] = [X], then [lc_amount] + [X] = [lc_max_allowed]. Step 3 — invoice_total_cif=[raw number]. Step 4 — Compare: [lc_max_allowed] vs [invoice_total_cif]. If lc_max_allowed >= invoice_total_cif → 'LC amount covers Invoice CIF in full.' If not → 'LC amount is insufficient — shortfall of [difference].'",
      "lc_amount_vs_invoice_cif_severity": "FILL: CRITICAL if FAIL | null if PASS",
      "lc_amount_vs_invoice_cif_status": "FILL: PASS or FAIL or UNABLE TO CHECK — document missing",

      "invoice_arithmetic_fob_detail": "FILL: Step 1 — Extract raw numbers: lc_unit_price=[raw number], total_quantity_kg=[raw number]. Step 2 — Calculate: [lc_unit_price] × [total_quantity_kg] = [your calculated result — compute this yourself, do not copy from document]. Step 3 — invoice_fob_stated=[raw number from invoice]. Step 4 — difference = [calculated] - [stated] = [difference value]. Step 5 — If difference = 0 → 'Arithmetic correct.' If difference ≠ 0 → 'Arithmetic incorrect — [lc_unit_price] × [total_quantity_kg] = [calculated] but Invoice states FOB [stated]. Difference = [difference].'",
      "invoice_arithmetic_fob_severity": "FILL: MAJOR if FAIL | null if PASS",
      "invoice_arithmetic_fob_status": "FILL: PASS if difference = 0.00 | FAIL if difference ≠ 0.00 — set this AFTER computing difference",

      "invoice_arithmetic_cif_detail": "FILL: Step 1 — Extract raw numbers: invoice_fob=[raw number], invoice_freight=[raw number], invoice_insurance=[raw number]. Step 2 — Calculate: [invoice_fob] + [invoice_freight] + [invoice_insurance] = [your calculated result]. Step 3 — invoice_total_cif_stated=[raw number from invoice]. Step 4 — difference = [calculated] - [stated] = [difference value]. If difference ≤ 1.00 → 'Arithmetic correct — FOB + Freight + Insurance = [calculated] matches Invoice CIF [stated].' If difference > 1.00 → 'Arithmetic incorrect — [fob] + [freight] + [insurance] = [calculated] but Invoice states CIF [stated]. Difference = [difference].'",
      "invoice_arithmetic_cif_severity": "FILL: MAJOR if FAIL | null if PASS",
      "invoice_arithmetic_cif_status": "FILL: PASS if difference ≤ 1.00 | FAIL if difference > 1.00 — set this AFTER computing difference",

      "insurance_coverage_check_detail": "FILL: Step 1 — Extract raw numbers: insurance_sum_insured=[raw number], invoice_total_cif=[raw number]. Step 2 — Calculate coverage%: ([sum_insured] ÷ [invoice_total_cif]) × 100 = [your calculated %]. Step 3 — Required minimum: 110%. Step 4 — expected_sum_insured = [invoice_total_cif] × 1.10 = [expected]. Step 5 — difference = [actual_sum_insured] - [expected] = [difference]. If coverage% >= 110% → 'Coverage adequate — sum insured [value] = [calculated]% of CIF [value], meets 110% requirement.' If coverage% < 110% → 'Coverage inadequate — sum insured [value] = [calculated]% of CIF [value], below required 110%.'",
      "insurance_coverage_check_severity": "FILL: CRITICAL if FAIL | null if PASS",
      "insurance_coverage_check_status": "FILL: PASS if coverage% >= 110% | FAIL if coverage% < 110% | UNABLE TO CHECK — document missing — set this AFTER computing coverage%",

      "boe_amount_vs_invoice_cif_detail": "FILL: Step 1 — Extract raw numbers: boe_amount=[raw number], invoice_total_cif=[raw number]. Step 2 — difference = [boe_amount] - [invoice_total_cif] = [difference value]. Step 3 — If difference = 0 → 'BOE amount [value] exactly matches Invoice CIF [value].' If difference ≠ 0 → 'BOE amount [value] does not match Invoice CIF [value]. Difference = [difference].'",
      "boe_amount_vs_invoice_cif_severity": "FILL: CRITICAL if FAIL | null if PASS",
      "boe_amount_vs_invoice_cif_status": "FILL: PASS if difference = 0.00 | FAIL if difference ≠ 0.00 | UNABLE TO CHECK — document missing — set this AFTER computing difference",

      "incoterm_consistency_detail": "FILL: Invoice incoterm=[exact value], B/L incoterm=[exact value], LC incoterm=[exact value]. If ALL identical → 'All 3 documents show the same incoterm: [value].' If ANY differ → '[document] shows [value] while others show [value].' Note: CFR and CIF are different — flag explicitly if mixed.",
      "incoterm_consistency_severity": "FILL: MAJOR if NOT MATCH | null if MATCH",
      "incoterm_consistency_status": "FILL: MATCH or NOT MATCH or UNABLE TO CHECK — document missing",

      "port_of_loading_detail": "FILL: Invoice=[exact value], B/L=[exact value], COO=[exact value], LC=[exact value]. If ALL identical → 'All 4 documents show the same port of loading: [value].' If ANY differ → '[document] shows [value] while others show [value].'",
      "port_of_loading_severity": "FILL: MAJOR if NOT MATCH | null if MATCH",
      "port_of_loading_status": "FILL: MATCH or NOT MATCH or UNABLE TO CHECK — document missing",

      "port_of_discharge_detail": "FILL: Invoice=[exact value], B/L=[exact value], Insurance=[exact value], LC=[exact value]. If ALL identical → 'All 4 documents show the same port of discharge: [value].' If ANY differ → '[document] shows [value] while others show [value].'",
      "port_of_discharge_severity": "FILL: MAJOR if NOT MATCH | null if MATCH",
      "port_of_discharge_status": "FILL: MATCH or NOT MATCH or UNABLE TO CHECK — document missing",

      "vessel_consistency_detail": "FILL: Invoice=[exact value], B/L=[exact value], Insurance=[exact value]. If ALL identical → 'All 3 documents show the same vessel: [value].' If ANY differ → '[document] shows [value] while others show [value].'",
      "vessel_consistency_severity": "FILL: MAJOR if NOT MATCH | null if MATCH",
      "vessel_consistency_status": "FILL: MATCH or NOT MATCH or UNABLE TO CHECK — document missing",

      "bl_onboard_vs_lc_shipment_deadline_detail": "FILL: bl_on_board_date=[value], lc_latest_shipment_date=[value]. Convert both to DD MMM YYYY for comparison. If bl_on_board_date <= lc_latest_shipment_date → 'Shipment within LC deadline.' If bl_on_board_date > lc_latest_shipment_date → 'CRITICAL — B/L on-board date exceeds LC latest shipment date by [X] days.'",
      "bl_onboard_vs_lc_shipment_deadline_severity": "FILL: CRITICAL if FAIL | null if PASS",
      "bl_onboard_vs_lc_shipment_deadline_status": "FILL: PASS if bl_on_board_date <= lc_latest_shipment_date | FAIL if bl_on_board_date > lc_latest_shipment_date | UNABLE TO CHECK — document missing",

      "bl_date_vs_invoice_date_detail": "FILL: invoice_date=[value], bl_date_of_issue=[value]. Convert both to DD MMM YYYY. If invoice_date <= bl_date → 'Invoice date precedes or equals B/L date — acceptable.' If invoice_date > bl_date → 'Red flag — Invoice date [value] is after B/L date [value], implying goods were shipped before invoice was raised.'",
      "bl_date_vs_invoice_date_severity": "FILL: MAJOR if FAIL | null if PASS",
      "bl_date_vs_invoice_date_status": "FILL: PASS if invoice_date <= bl_date | FAIL if invoice_date > bl_date | UNABLE TO CHECK — document missing"
    },

    "quantity_weight_checks": {
      "package_count_detail": "FILL: Invoice=[exact value], Packing List=[exact value], B/L=[exact value], COO=[exact value], Inspection=[exact value]. If ALL identical → 'All 5 documents show the same package count: [value].' If ANY differ → '[document] shows [value] while others show [value].'",
      "package_count_severity": "FILL: MAJOR if MISMATCH | null if MATCH",
      "package_count_status": "FILL: MATCH or MISMATCH or UNABLE TO CHECK — document missing",

      "net_weight_detail": "FILL: Invoice=[exact value], Packing List=[exact value], B/L=[exact value], Inspection=[exact value]. Step 1 — variance_mt = |inspection_net_weight - invoice_net_weight| = [value]. Step 2 — variance% = ([variance_mt] ÷ [invoice_net_weight]) × 100 = [calculated%]. Show arithmetic explicitly. If variance% <= 0.5% → 'Net weight values within 0.5% tolerance — acceptable.' If variance% > 0.5% → 'Net weight variance of [X]% exceeds 0.5% tolerance — Inspection shows [value] vs Invoice [value].'",
      "net_weight_severity": "FILL: MAJOR if variance > 0.5% and non-inspection doc differs | MINOR if only inspection variance > 0.5% | null if all match",
      "net_weight_status": "FILL: MATCH if all identical | WARNING if inspection variance > 0.5% | MISMATCH if non-inspection documents differ | UNABLE TO CHECK — document missing",

      "gross_weight_detail": "FILL: Invoice=[exact value], Packing List=[exact value], B/L=[exact value]. If ALL identical → 'Gross weight matches across all 3 documents: [value].' If ANY differ → '[document] shows [value] while others show [value].'",
      "gross_weight_severity": "FILL: MINOR if MISMATCH | null if MATCH",
      "gross_weight_status": "FILL: MATCH or MISMATCH or UNABLE TO CHECK — document missing",

      "commodity_description_detail": "FILL: LC required commodity description=[exact wording from LC]. Invoice goods description=[exact wording from Invoice]. Compare attribute by attribute. List any missing attributes: [list]. List any mismatched attributes: [list]. If all LC attributes satisfied → 'Invoice description conforms to LC commodity description.' If any deviation → 'Material deviation — LC requires [value] but Invoice states [value]. Specific differences: [list].'",
      "commodity_description_severity": "FILL: MAJOR if NOT MATCH | null if MATCH",
      "commodity_description_status": "FILL: MATCH or NOT MATCH or UNABLE TO CHECK — document missing",

      "hs_code_detail": "FILL: Invoice=[exact value], Packing List=[exact value], COO=[exact value], Inspection=[exact value]. If ALL identical → 'All 4 documents show the same HS code: [value].' If ANY differ → '[document] shows [value] while others show [value].'",
      "hs_code_severity": "FILL: MAJOR if MISMATCH | null if MATCH",
      "hs_code_status": "FILL: MATCH or MISMATCH or UNABLE TO CHECK — document missing",

      "quantity_unit_detail": "FILL: Invoice=[exact value], Packing List=[exact value], B/L=[exact value], COO=[exact value]. Compare each value exactly character by character. If ALL identical → 'All 4 documents show the same quantity and unit: [value].' If ANY single document shows a different value → 'MISMATCH — [document] shows [value] while others show [value].'",
      "quantity_unit_severity": "FILL: MAJOR if MISMATCH | null if MATCH",
      "quantity_unit_status": "FILL: MATCH or MISMATCH or UNABLE TO CHECK — document missing"
    },

    "date_checks": {
      "date_invoice_vs_bl_detail": "FILL: invoice_date=[value], bl_on_board_date=[value]. Convert both to DD MMM YYYY. If invoice_date <= bl_on_board_date → 'Date sequence correct — Invoice date precedes or equals B/L on-board date.' If invoice_date > bl_on_board_date → 'Date sequence violation — Invoice date [value] is after B/L on-board date [value].'",
      "date_invoice_vs_bl_severity": "FILL: MAJOR if FAIL | null if PASS",
      "date_invoice_vs_bl_status": "FILL: PASS if invoice_date <= bl_on_board_date | FAIL if invoice_date > bl_on_board_date | UNABLE TO CHECK — document missing",

      "date_bl_vs_lc_shipment_detail": "FILL: bl_date_of_issue=[value], lc_latest_shipment_date=[value]. Convert both to DD MMM YYYY. If bl_date <= lc_latest_shipment_date → 'B/L date within LC latest shipment date.' If bl_date > lc_latest_shipment_date → 'CRITICAL — B/L date [value] exceeds LC latest shipment date [value].'",
      "date_bl_vs_lc_shipment_severity": "FILL: CRITICAL if FAIL | null if PASS",
      "date_bl_vs_lc_shipment_status": "FILL: PASS if bl_date <= lc_latest_shipment_date | FAIL if bl_date > lc_latest_shipment_date | UNABLE TO CHECK — document missing",

      "date_insurance_vs_bl_detail": "FILL: insurance_date=[value], bl_on_board_date=[value]. Convert both to DD MMM YYYY. If insurance_date <= bl_on_board_date → 'Insurance issued before or at shipment — acceptable.' If insurance_date > bl_on_board_date → 'CRITICAL — Insurance date [value] is after B/L on-board date [value] — goods were not insured at time of shipment.'",
      "date_insurance_vs_bl_severity": "FILL: CRITICAL if FAIL | null if PASS",
      "date_insurance_vs_bl_status": "FILL: PASS if insurance_date <= bl_on_board_date | FAIL if insurance_date > bl_on_board_date | UNABLE TO CHECK — document missing",

      "date_inspection_vs_bl_detail": "FILL: inspection_date=[value], bl_on_board_date=[value]. Convert both to DD MMM YYYY. If inspection_date <= bl_on_board_date → 'Inspection completed before loading — acceptable.' If inspection_date > bl_on_board_date → 'Inspection date [value] is after B/L on-board date [value] — goods were loaded before inspection.'",
      "date_inspection_vs_bl_severity": "FILL: MAJOR if FAIL | null if PASS",
      "date_inspection_vs_bl_status": "FILL: PASS if inspection_date <= bl_on_board_date | FAIL if inspection_date > bl_on_board_date | UNABLE TO CHECK — document missing",

      "date_all_vs_lc_expiry_detail": "FILL: lc_expiry_date=[value]. Check each document's primary date: Commercial Invoice=[date] [PASS/FAIL], Bill of Lading=[date] [PASS/FAIL], COO=[date] [PASS/FAIL], Insurance=[date] [PASS/FAIL], Inspection=[date] [PASS/FAIL], BOE=[date] [PASS/FAIL], Packing List=[date] [PASS/FAIL], LC itself=[date] [PASS/FAIL]. If ALL <= lc_expiry → 'All document dates are within LC expiry date.' If ANY exceed → '[document] dated [value] exceeds LC expiry date [value] — CRITICAL.'",
      "date_all_vs_lc_expiry_severity": "FILL: CRITICAL if any document exceeds LC expiry | null if all within",
      "date_all_vs_lc_expiry_status": "FILL: PASS if all dates <= lc_expiry | FAIL if any date > lc_expiry | UNABLE TO CHECK — document missing",

      "presentation_period_detail": "FILL: bl_on_board_date=[value]. Step 1 — presentation_deadline = bl_on_board_date + 21 days = [calculated date — show arithmetic: DD MMM YYYY + 21 = DD MMM YYYY]. Step 2 — today_ist = [get current date using: from datetime import datetime; from zoneinfo import ZoneInfo; datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%d %b %Y')]. Step 3 — days_remaining = [presentation_deadline - today_ist — show arithmetic]. Step 4 — Also check today_ist <= lc_expiry_date=[value]. If today_ist <= presentation_deadline AND today_ist <= lc_expiry → 'Presentation is within the 21-day window — [X] days remaining to deadline, [Y] days to LC expiry.' If either breached → state which condition failed and by how many days.",
      "presentation_period_severity": "FILL: CRITICAL if FAIL | null if PASS",
      "presentation_period_status": "FILL: PASS if today_ist <= presentation_deadline AND today_ist <= lc_expiry | FAIL if either condition breached | UNABLE TO CHECK — document missing — set this AFTER computing all date comparisons",

      "stale_bl_detail": "FILL: bl_on_board_date=[value]. Step 1 — stale_deadline = bl_on_board_date + 21 days = [calculated date — show arithmetic: DD MMM YYYY + 21 = DD MMM YYYY]. Step 2 — today_ist = [get current IST date same as above]. Step 3 — days_elapsed = today_ist - bl_on_board_date = [value]. Step 4 — days_until_stale = stale_deadline - today_ist = [value]. If today_ist <= stale_deadline → 'B/L is not stale — [X] days elapsed, [Y] days remaining before stale.' If today_ist > stale_deadline → 'B/L IS STALE — [X] days elapsed since on-board date, exceeding 21-day UCP 600 limit by [Y] days.'",
      "stale_bl_severity": "FILL: MAJOR if FAIL (stale) | null if PASS",
      "stale_bl_status": "FILL: PASS if today_ist <= stale_deadline | FAIL if today_ist > stale_deadline | UNABLE TO CHECK — document missing — set this AFTER computing stale_deadline vs today_ist"
    },

    "lc_compliance_checks": {
      "lc_docs_checklist_detail": "FILL: LC requires the following documents: [list all from lc_required_documents]. Check each against the 8 provided files. Present: [list matched]. Missing: [list unmatched or 'None']. Non-conforming: [list or 'None']. If all present and conforming → 'All LC-required documents are present and conforming.' If any missing → 'Missing documents: [list]. These must be submitted before presentation.'",
      "lc_docs_checklist_severity": "FILL: CRITICAL if any MISSING | MAJOR if NON-CONFORMING | null if all PRESENT",
      "lc_docs_checklist_status": "FILL: PASS if all present and conforming | FAIL if any missing or non-conforming | UNABLE TO CHECK — document missing",

      "lc_doc_01_required": "FILL: exact document name as written in LC documents_required list — item 1",
      "lc_doc_01_remark": "FILL: state which provided file covers this requirement and any conformity note, or state why it is missing",
      "lc_doc_01_status": "FILL: PRESENT or MISSING or NON-CONFORMING",

      "lc_doc_02_required": "FILL: exact document name as written in LC documents_required list — item 2",
      "lc_doc_02_remark": "FILL: state which provided file covers this requirement and any conformity note, or state why it is missing",
      "lc_doc_02_status": "FILL: PRESENT or MISSING or NON-CONFORMING",

      "lc_doc_03_required": "FILL: exact document name as written in LC documents_required list — item 3",
      "lc_doc_03_remark": "FILL: state which provided file covers this requirement and any conformity note, or state why it is missing",
      "lc_doc_03_status": "FILL: PRESENT or MISSING or NON-CONFORMING",

      "lc_doc_04_required": "FILL: exact document name as written in LC documents_required list — item 4",
      "lc_doc_04_remark": "FILL: state which provided file covers this requirement and any conformity note, or state why it is missing",
      "lc_doc_04_status": "FILL: PRESENT or MISSING or NON-CONFORMING",

      "lc_doc_05_required": "FILL: exact document name as written in LC documents_required list — item 5",
      "lc_doc_05_remark": "FILL: state which provided file covers this requirement and any conformity note, or state why it is missing",
      "lc_doc_05_status": "FILL: PRESENT or MISSING or NON-CONFORMING",

      "lc_doc_06_required": "FILL: exact document name as written in LC documents_required list — item 6",
      "lc_doc_06_remark": "FILL: state which provided file covers this requirement and any conformity note, or state why it is missing",
      "lc_doc_06_status": "FILL: PRESENT or MISSING or NON-CONFORMING",

      "lc_doc_07_required": "FILL: exact document name as written in LC documents_required list — item 7",
      "lc_doc_07_remark": "FILL: state which provided file covers this requirement and any conformity note, or state why it is missing",
      "lc_doc_07_status": "FILL: PRESENT or MISSING or NON-CONFORMING",

      "partial_shipment_detail": "FILL: lc_partial_shipment=[exact LC value — ALLOWED or NOT ALLOWED]. Number of B/L sets presented=[count]. If NOT ALLOWED and single full B/L set → 'Compliant — single full B/L set presented as required.' If NOT ALLOWED and multiple sets → 'Non-compliant — [X] B/L sets presented but LC prohibits partial shipment.' If ALLOWED → 'LC permits partial shipment — no restriction applies.'",
      "partial_shipment_severity": "FILL: MAJOR if FAIL | null if PASS or INFO",
      "partial_shipment_status": "FILL: PASS if NOT ALLOWED and single B/L | FAIL if NOT ALLOWED and multiple B/Ls | INFO if ALLOWED | UNABLE TO CHECK — document missing",

      "transhipment_detail": "FILL: lc_transhipment=[exact LC value — ALLOWED or NOT ALLOWED]. B/L routing=[direct voyage or via transhipment port at [value]]. If NOT ALLOWED and B/L shows direct voyage → 'Compliant — B/L shows direct voyage with no transhipment.' If NOT ALLOWED and transhipment port shown → 'Non-compliant — B/L shows transhipment via [port] but LC prohibits transhipment.' If ALLOWED → 'LC permits transhipment — no restriction applies.'",
      "transhipment_severity": "FILL: MAJOR if FAIL | null if PASS or INFO",
      "transhipment_status": "FILL: PASS if NOT ALLOWED and direct voyage | FAIL if NOT ALLOWED and transhipment shown | INFO if ALLOWED | UNABLE TO CHECK — document missing",

      "third_party_docs_detail": "FILL: LC clause on third-party documents=[state exact LC clause from lc_special_conditions, or 'No restriction stated']. Inspection certificate issuing body=[exact value from inspection_issuing_body]. If issuing body satisfies LC requirement → 'Inspection certificate issuer [value] is acceptable per LC terms.' If issuing body does not satisfy LC requirement → 'Non-compliant — LC requires [value] but certificate was issued by [value].' If no restriction → 'No third-party document restriction stated in LC.'",
      "third_party_docs_severity": "FILL: MAJOR if FAIL | null if PASS or INFO",
      "third_party_docs_status": "FILL: PASS if issuer meets LC requirement | FAIL if issuer does not meet LC requirement | INFO if no restriction stated | UNABLE TO CHECK — document missing"
    }
  },

  "summary": {
    "missing_documents": ["FILL: list any of the 8 expected document types not present in the input — plain strings only, e.g. 'Phytosanitary Certificate', 'Beneficiary Certificate'. If none missing write empty array []"],
    "total_checks_run": "FILL: count of all checks executed across all categories — must be a number string e.g. '28'",
    "total_passed": "FILL: count of checks with status PASS or MATCH — must be a number string",
    "total_failed": "FILL: count of checks with status FAIL or NOT MATCH or MISMATCH — must be a number string",
    "critical_count": "FILL: count of checks where severity = CRITICAL — must be a number string",
    "major_count": "FILL: count of checks where severity = MAJOR — must be a number string",
    "minor_count": "FILL: count of checks where severity = MINOR — must be a number string",
    "overall_verdict": "FILL: CLEAN PRESENTATION if total_failed = 0 | DISCREPANT PRESENTATION if total_failed > 0",
    "overall_summary": "FILL: 2-3 sentence plain English summary — state total checks run, how many passed, how many failed, list the critical
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