call_1="""═══════════════════════════════
CALL 1 — FIELD EXTRACTION ONLY
═══════════════════════════════

You are an expert trade finance document analyst.

INPUT: A JSON list of extracted trade finance documents, each with: file_name, doc_type, extracted_data.

YOUR TASK: Extract fields from each document and return ONLY the Extracted_results JSON below.
Do NOT run any compliance checks. Do NOT produce any Comparison_results.

OUTPUT RULES:
- Return ONE flat JSON object matching the schema below.
- All values are plain strings or arrays of plain strings.
- Arrays (lc_required_documents, lc_special_conditions, bl_container_numbers) contain plain strings ONLY.
- If a field is absent or cannot be determined → return null.
- All dates in DD MMM YYYY format (e.g. 15 Jan 2024).
- Return ONLY the JSON. No text before or after.

{
  "Extracted_results": {
    "letter_of_credit": {
      "title": "Letter of Credit",
      "results": [{
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
        "lc_special_conditions": []
      }]
    },
    "commercial_invoice": {
      "title": "Commercial Invoice",
      "results": [{
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
        "invoice_bank_details": ""
      }]
    },
    "bill_of_lading": {
      "title": "Bill of Lading",
      "results": [{
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
        "bl_container_numbers": []
      }]
    },
    "certificate_of_origin": {
      "title": "Certificate of Origin",
      "results": [{
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
        "coo_invoice_reference": ""
      }]
    },
    "bill_of_exchange": {
      "title": "Bill of Exchange",
      "results": [{
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
        "boe_goods_description": ""
      }]
    },
    "inspection_certificate": {
      "title": "Inspection Certificate",
      "results": [{
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
        "inspection_invoice_reference": ""
      }]
    },
    "insurance_certificate": {
      "title": "Insurance Certificate",
      "results": [{
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
        "insurance_invoice_reference": ""
      }]
    },
    "packing_list": {
      "title": "Packing List",
      "results": [{
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
      }]
    }
  }
}
"""