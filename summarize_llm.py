import os
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()


system_prompt = """

You are a trade finance operations assistant for a bank.

You will be given structured data extracted from:
- A Letter of Credit (LC)
- Commercial Invoice
- Transport document (Bill of Lading or Air Waybill)
- Certificate of Origin (if present)

Your task is to:
1. Check document completeness against LC requirements
2. Check consistency across documents
3. Check compliance with LC conditions
4. Identify discrepancies clearly and factually
5. Summarize the overall status in simple banking language

Rules:
- Do NOT assume intent or make commercial judgments
- Treat LC terms as strict
- Highlight only document-based issues
- Classify each issue as:
  - Minor discrepancy
  - Major discrepancy
  - Compliant

Output format:
- Overall Status: COMPLIANT / DISCREPANCIES FOUND
- Summary (2–3 lines)
- Detailed Findings (bullet points)
- Missing Documents (if any)

Here is the extracted data:
Sample JSON file format: [
  {
    "file_name": "Proforma Invoice -1(filled).pdf",
    "doc_type": "INVOICE",
    "extracted_data": {
      "invoice_type": "PROFORMA INVOICE",
      "invoice_number": "PI-CB-CHN-00125",
      "invoice_date": "01-Mar-2026",
      "seller_name": "Shenzhen Golden Tech Equipment Co., Ltd",
      "seller_address": "Nanshan District, Shenzhen, People's Republic of China",
      "buyer_name": "Desert Horizon Trading LLC",
      "buyer_address": "Deira, Dubai, United Arab Emirates",
      "goods_description": "Industrial Electrical Control Panels",
      "hs_code": null,
      "quantity": "20",
      "unit_price": "2,500",
      "total_amount": "50,000",
      "currency": "USD",
      "incoterms": "CIF",
      "port_of_loading": "Shenzhen Bao'an International Airport",
      "port_of_discharge": "Dubai International Airport",
      "country_of_origin": "China",
      "payment_terms": "Letter of Credit - At Sight"
    }
  },
  {
    "file_name": "COURIER-FORM-AIRWAY-EXPRESS -1(filled1).pdf",
    "doc_type": "COURIER_DISPATCH_ADVICE",
    "extracted_data": {
      "exporter_name": "Shenzhen Golden Tech Equipment Co., Ltd",
      "exporter_address": "Nanshan District, Shenzhen, People's Republic of China, Postal Code: 518052",
      "importer_or_bank_name": "China Bank",
      "importer_or_bank_address": "Trade Operations Department, Shenzhen, People's Republic of China, Postal Code: 518040",
      "courier_company": "Airway Express Logistics Dubai LLC",
      "courier_awb_number": "AE-CR-99018",
      "dispatch_date": "16/Mar/2026",
      "contract_number": null,
      "invoice_number": null,
      "documents_sent": [
        "Non-negotiable documents"
      ],
      "authorized_signatory": "Li Wei",
      "exporter_stamp_present": true
    }
  },
  {
    "file_name": "Edited_Air_Way_Bill_sample.pdf",
    "doc_type": "AIR_WAYBILL",
    "extracted_data": {
      "awb_number": "99881234",
      "awb_type": "Air Waybill",
      "shipper_name": "Shenzhen Golden Tech Equipment Co., Ltd",
      "shipper_address": "Nanshan District, Shenzhen, People's Republic of China",
      "consignee_name": "Emirates NBD Bank PJSC",
      "consignee_address": "Trade Finance Operations, Dubai, United Arab Emirates",
      "airport_of_origin": "Shenzhen Bao'an International Airport and Shenzhen, China",
      "airport_of_destination": "Dubai International Airport, Dubai, UAE",
      "flight_number": null,
      "flight_date": "2026-03-14",
      "goods_description": "Industrial Electrical Control Panels Packed in 10 Wooden Crates",
      "number_of_packages": "10",
      "gross_weight": "2000 kg",
      "freight_terms": "Prepaid",
      "carrier_or_courier_stamp_present": "Yes"
    }
  }
]
     
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

    def extract(self, normalized_doc: dict) -> dict:

        
        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(normalized_doc)},
            ],
        )

        raw_output = response.choices[0].message.content.strip()

        # Debug / audit log
        print("\n🏦 TRADE FINANCE COMPLIANCE SUMMARY:\n")
        print(raw_output)

        return {
            "compliance_summary": raw_output
        }
