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


═══════════════════════════════════════════════════════════════
PART 2 — CROSS-DOCUMENT COMPLIANCE CHECKS
═══════════════════════════════════════════════════════════════

Run every check below. For each check:
1. Fill all fields EXCEPT status first.
2. Write details with full evidence and arithmetic.
3. Set status LAST — only after reading back your own details and values.


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

    "lc_number": "FILL: LC number exactly as printed on the Letter of Credit",
    "lc_issue_date": "FILL: date of issue from LC in DD MMM YYYY format",
    "lc_expiry_date": "FILL: expiry date from LC in DD MMM YYYY format",
    "lc_expiry_place": "FILL: place of expiry as stated in LC",
    "lc_amount": "FILL: LC amount as a plain number string e.g. 148000.00",
    "lc_currency": "FILL: currency code e.g. USD",
    "lc_tolerance": "FILL: tolerance as stated e.g. +/- 5%",
    "lc_applicant": "FILL: full name and address of applicant (buyer) exactly as in LC",
    "lc_beneficiary": "FILL: full name and address of beneficiary (seller) exactly as in LC",
    "lc_issuing_bank": "FILL: full name, address and SWIFT of issuing bank exactly as in LC",
    "lc_advising_bank": "FILL: full name, address and SWIFT of advising bank exactly as in LC",
    "lc_latest_shipment_date": "FILL: latest shipment date from LC in DD MMM YYYY format",
    "lc_incoterm": "FILL: incoterm exactly as stated in LC e.g. CIF Antwerp",
    "lc_port_of_loading": "FILL: port of loading exactly as stated in LC",
    "lc_port_of_discharge": "FILL: port of discharge exactly as stated in LC",
    "lc_partial_shipment": "FILL: ALLOWED or NOT ALLOWED exactly as stated in LC",
    "lc_transhipment": "FILL: ALLOWED or NOT ALLOWED exactly as stated in LC",
    "lc_presentation_period": "FILL: presentation period exactly as stated in LC e.g. 21 days after B/L date",
    "lc_commodity_description": "FILL: commodity description exactly word-for-word as written in LC",
    "lc_hs_code": "FILL: HS code exactly as stated in LC",
    "lc_quantity": "FILL: quantity exactly as stated in LC e.g. 800 Bags x 60 kg = 48.000 MT",
    "lc_required_documents": ["FILL: each required document as a separate plain string exactly as listed in LC — one item per document"],
    "lc_special_conditions": ["FILL: each special condition as a separate plain string exactly as stated in LC — empty array [] if none"],

    "invoice_number": "FILL: invoice number exactly as printed on Commercial Invoice",
    "invoice_date": "FILL: invoice date in DD MMM YYYY format",
    "invoice_exporter_name_address": "FILL: full exporter name and address exactly as printed on invoice — preserve every word and character",
    "invoice_importer_name_address": "FILL: full importer/consignee name and address exactly as printed on invoice — preserve every word and character",
    "invoice_lc_reference": "FILL: LC number referenced on invoice",
    "invoice_goods_description": "FILL: goods description exactly as written on invoice — preserve all words, grade, process, packing details",
    "invoice_hs_code": "FILL: HS code exactly as stated on invoice",
    "invoice_quantity": "FILL: quantity exactly as stated on invoice e.g. 800 Bags / 48.000 MT",
    "invoice_unit_price": "FILL: unit price exactly as stated on invoice e.g. USD 3.20 per kg",
    "invoice_incoterm": "FILL: incoterm exactly as stated on invoice e.g. CFR Antwerp",
    "invoice_total_fob": "FILL: total FOB value as plain number string e.g. 144000.00",
    "invoice_freight": "FILL: freight value as plain number string e.g. 8200.00",
    "invoice_insurance": "FILL: insurance value as plain number string e.g. 1800.00",
    "invoice_total_cif": "FILL: total CIF value as plain number string e.g. 154000.00",
    "invoice_currency": "FILL: currency code e.g. USD",
    "invoice_port_of_loading": "FILL: port of loading exactly as stated on invoice",
    "invoice_port_of_discharge": "FILL: port of discharge exactly as stated on invoice",
    "invoice_vessel": "FILL: vessel name exactly as stated on invoice",
    "invoice_bank_details": "FILL: full bank details as stated on invoice including bank name, account number, IFSC, SWIFT",

    "bl_number": "FILL: B/L number exactly as printed on Bill of Lading",
    "bl_date_of_issue": "FILL: date of issue from B/L in DD MMM YYYY format",
    "bl_on_board_date": "FILL: on-board notation date from B/L in DD MMM YYYY format — this is the shipped on board date, not just issue date",
    "bl_shipper": "FILL: shipper name and address exactly as printed on B/L — preserve every word and character",
    "bl_consignee": "FILL: consignee exactly as printed on B/L e.g. TO THE ORDER OF [name] or named consignee — preserve every word",
    "bl_notify_party": "FILL: notify party exactly as printed on B/L",
    "bl_vessel": "FILL: vessel name exactly as printed on B/L",
    "bl_voyage": "FILL: voyage number exactly as printed on B/L",
    "bl_port_of_loading": "FILL: port of loading exactly as printed on B/L",
    "bl_port_of_discharge": "FILL: port of discharge exactly as printed on B/L",
    "bl_number_of_packages": "FILL: number of packages exactly as stated in cargo description on B/L e.g. 800 Bags",
    "bl_gross_weight": "FILL: gross weight exactly as stated on B/L e.g. 48.960 MT",
    "bl_cbm": "FILL: volume in CBM exactly as stated on B/L",
    "bl_freight_terms": "FILL: PREPAID or COLLECT exactly as stated on B/L",
    "bl_incoterm": "FILL: incoterm exactly as stated on B/L",
    "bl_lc_reference": "FILL: LC number referenced on B/L",
    "bl_invoice_reference": "FILL: invoice number referenced on B/L",
    "bl_number_of_originals": "FILL: number of original B/L sets e.g. THREE (3)",
    "bl_container_numbers": ["FILL: each container number as a separate plain string e.g. MSCU8901234"],

    "coo_certificate_number": "FILL: certificate number exactly as printed on Certificate of Origin",
    "coo_date": "FILL: certificate date in DD MMM YYYY format",
    "coo_exporter": "FILL: exporter name and address exactly as printed on COO — preserve every word and character",
    "coo_consignee": "FILL: consignee name and address exactly as printed on COO — preserve every word and character",
    "coo_issuing_authority": "FILL: issuing authority name exactly as printed on COO",
    "coo_country_of_origin": "FILL: country of origin exactly as stated on COO",
    "coo_port_of_loading": "FILL: port of loading exactly as stated on COO",
    "coo_port_of_discharge": "FILL: port of discharge exactly as stated on COO",
    "coo_hs_code": "FILL: HS code exactly as stated on COO",
    "coo_goods_description": "FILL: goods description exactly as written on COO — preserve all words",
    "coo_quantity": "FILL: quantity exactly as stated on COO",
    "coo_net_weight": "FILL: net weight exactly as stated on COO e.g. 48.000 MT",
    "coo_gross_weight": "FILL: gross weight exactly as stated on COO e.g. 48.960 MT",
    "coo_invoice_reference": "FILL: invoice number referenced on COO",

    "boe_number": "FILL: BOE number exactly as printed on Bill of Exchange",
    "boe_date": "FILL: BOE date in DD MMM YYYY format",
    "boe_drawer": "FILL: drawer name and address exactly as printed on BOE — preserve every word and character",
    "boe_drawee": "FILL: drawee bank name and address exactly as printed on BOE",
    "boe_pay_to_order_of": "FILL: pay to order of value exactly as printed on BOE",
    "boe_amount_figures": "FILL: amount in figures as plain number string e.g. 154000.00",
    "boe_currency": "FILL: currency code e.g. USD",
    "boe_tenor": "FILL: tenor exactly as stated on BOE e.g. AT SIGHT",
    "boe_lc_reference": "FILL: LC number referenced on BOE",
    "boe_invoice_reference": "FILL: invoice number referenced on BOE",
    "boe_incoterm": "FILL: incoterm exactly as stated on BOE",
    "boe_goods_description": "FILL: goods description exactly as stated on BOE",

    "inspection_cert_number": "FILL: certificate number exactly as printed on Inspection Certificate",
    "inspection_date": "FILL: inspection date in DD MMM YYYY format",
    "inspection_issuing_body": "FILL: issuing body name exactly as printed on Inspection Certificate",
    "inspection_client_exporter": "FILL: client/exporter name exactly as printed on Inspection Certificate — preserve every word and character",
    "inspection_consignee": "FILL: consignee name exactly as printed on Inspection Certificate — preserve every word and character",
    "inspection_commodity": "FILL: commodity description exactly as stated on Inspection Certificate",
    "inspection_hs_code": "FILL: HS code exactly as stated on Inspection Certificate",
    "inspection_quantity_inspected": "FILL: quantity inspected exactly as stated e.g. 49.200 MT",
    "inspection_net_weight": "FILL: net weight found exactly as stated as plain number string e.g. 49.200",
    "inspection_overall_conclusion": "FILL: overall conclusion exactly as stated e.g. APPROVED",
    "inspection_lc_reference": "FILL: LC number referenced on Inspection Certificate or null if not present",
    "inspection_invoice_reference": "FILL: invoice number referenced on Inspection Certificate",

    "insurance_policy_number": "FILL: policy number exactly as printed on Insurance Certificate",
    "insurance_date": "FILL: certificate date in DD MMM YYYY format",
    "insurance_insured": "FILL: insured/assured name and address exactly as printed on Insurance Certificate — preserve every word and character",
    "insurance_beneficiary": "FILL: beneficiary name exactly as printed on Insurance Certificate — preserve every word and character",
    "insurance_sum_insured": "FILL: sum insured as plain number string e.g. 169400.00",
    "insurance_cif_value": "FILL: CIF base value as plain number string e.g. 154000.00",
    "insurance_coverage_factor": "FILL: coverage factor exactly as stated e.g. 110% of CIF",
    "insurance_coverage_type": "FILL: coverage type exactly as stated e.g. ICC (A) All Risks",
    "insurance_vessel": "FILL: vessel name exactly as printed on Insurance Certificate",
    "insurance_port_of_loading": "FILL: port of loading exactly as stated on Insurance Certificate",
    "insurance_port_of_discharge": "FILL: port of discharge exactly as stated on Insurance Certificate",
    "insurance_on_board_date": "FILL: on-board date in DD MMM YYYY format",
    "insurance_invoice_reference": "FILL: invoice number referenced on Insurance Certificate",

    "pl_date": "FILL: packing list date in DD MMM YYYY format",
    "pl_exporter": "FILL: exporter name exactly as printed on Packing List — preserve every word and character",
    "pl_consignee": "FILL: consignee name exactly as printed on Packing List — preserve every word and character",
    "pl_lc_reference": "FILL: LC number referenced on Packing List",
    "pl_invoice_reference": "FILL: invoice number referenced on Packing List",
    "pl_hs_code": "FILL: HS code exactly as stated on Packing List",
    "pl_total_packages": "FILL: total number of packages exactly as stated e.g. 800 Bags",
    "pl_total_net_weight": "FILL: total net weight exactly as stated e.g. 48000 kg or 48.000 MT",
    "pl_total_gross_weight": "FILL: total gross weight exactly as stated e.g. 48960 kg or 48.960 MT",
    "pl_total_cbm": "FILL: total volume exactly as stated e.g. 148.0 CBM",
    "pl_vessel": "FILL: vessel name exactly as printed on Packing List",
    "pl_port_of_loading": "FILL: port of loading exactly as stated on Packing List",
    "pl_port_of_discharge": "FILL: port of discharge exactly as stated on Packing List",
    "pl_marks_and_numbers": "FILL: marks and numbers exactly as printed on Packing List — preserve all lines"


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