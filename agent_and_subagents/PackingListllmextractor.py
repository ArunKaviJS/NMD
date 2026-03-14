import os
import json
import re
from openai import AzureOpenAI


class PackingListLLMExtractor:
    """
    Extract mandatory fields from PackingList
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
        Extract PackingList mandatory fields
        """

        system_prompt = """
        You are a Trade Finance Packing List Extraction Engine.

        Document Type: PACKING LIST

        Rules:
        - Extract values ONLY if explicitly present in the document
        - DO NOT guess or infer
        - DO NOT explain or add commentary
        - If a field is not clearly mentioned, return null
        - Output MUST be valid JSON only — no markdown, no extra text
        - Do NOT add extra fields
        - Do NOT rename fields

        Field Mapping Rules:
        - exporter = Exporter full name + address as one string
        - consignee = Consignee full name + address as one string
        - invoice_reference = Invoice number referenced at the top of the packing list
        - packing_list_date = Date of the packing list
        - lc_number = LC number referenced
        - port_of_loading = Port where goods are loaded
        - port_of_discharge = Destination port
        - vessel = Vessel name
        - ship_date = Shipment / sailing date
        - hs_code = HS Code of the goods (should be consistent across all packages)
        - commodity_description = General goods description (from package details, first row is enough)
        - package_breakdown = Array of objects — one entry per package row (excluding TOTALS row).
                            Each object must have exactly these keys:
                                "pkg_number"      : package reference number (e.g. "P001")
                                "description"     : goods description for that package
                                "no_of_bags"      : number of bags/units in that package (integer)
                                "bag_range"       : bag number range if mentioned (e.g. "1-1000")
                                "net_wt_per_bag"  : net weight per individual bag with unit (e.g. "25.000 kg")
                                "gross_wt_per_bag": gross weight per individual bag with unit (e.g. "25.250 kg")
                                "total_net_wt"    : total net weight for this package row with unit
                                "total_gross_wt"  : total gross weight for this package row with unit
                                "marks_numbers"   : marks & numbers for this package (e.g. "AAEPL/2026 Pallet 1 of 5")
        - total_packages = Total number of packages / bags from TOTALS row (e.g. "5,000 Bags")
        - total_pallets = Total number of pallets if mentioned (e.g. "5 Pallets")
        - total_net_weight = Total net weight from TOTALS / Shipment Summary row with unit
        - total_gross_weight = Total gross weight from TOTALS / Shipment Summary row with unit
        - total_volume_cbm = Total volume in CBM from Shipment Summary (e.g. "86.00 CBM")
        - container_details = Array of objects — one entry per container listed.
                            Each object must have exactly these keys:
                                "container_number": container number (e.g. "TCKU3456789")
                                "seal_number"     : seal number (e.g. "SL-2026-44217")
                                "size_type"       : container size and type (e.g. "20' FCL / Dry")
                                "packages_loaded" : number of packages loaded in this container
        - marks_and_numbers = Full marks & numbers block as written at the bottom of the document
                            (the multi-line stencil/shipping marks block, not per-package marks)

        Required JSON Schema:

        {
        "exporter": null,
        "consignee": null,
        "invoice_reference": null,
        "packing_list_date": null,
        "lc_number": null,
        "port_of_loading": null,
        "port_of_discharge": null,
        "vessel": null,
        "ship_date": null,
        "hs_code": null,
        "commodity_description": null,
        "package_breakdown": [],
        "total_packages": null,
        "total_pallets": null,
        "total_net_weight": null,
        "total_gross_weight": null,
        "total_volume_cbm": null,
        "container_details": [],
        "marks_and_numbers": null
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
        print("\n✈️ RAW PackingListLLMExtractor LLM OUTPUT:\n", raw_output)

        try:
            return self._safe_json_parse(raw_output)
        except Exception as e:
            print("❌ PackingListLLMExtractor extraction failed:", str(e))
            return {
                "error": "PackingListLLMExtractor_EXTRACTION_FAILED",
                "raw_llm_output": raw_output
            }
