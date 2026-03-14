import os
import json
import re
from openai import AzureOpenAI


class BillOfLadingLLMExtractor:
    """
    Extract mandatory fields from BillOfLading
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
        Extract BillOfLading mandatory fields
        """

        system_prompt = """
        You are a Trade Finance Bill of Lading Extraction Engine.

        Document Type: OCEAN / MARINE BILL OF LADING

        Rules:
        - Extract values ONLY if explicitly present in the document
        - DO NOT guess or infer
        - DO NOT explain or add commentary
        - If a field is not clearly mentioned, return null
        - Output MUST be valid JSON only — no markdown, no extra text
        - Do NOT add extra fields
        - Do NOT rename fields

        Field Mapping Rules:
        - bl_number = Bill of Lading number / B/L No at the top of the document
        - date_of_issue = Date the B/L was issued (NOT the on-board date)
        - place_of_issue = Place where the B/L was issued
        - on_board_date = The actual "Date on Board" / "Shipped on Board" date —
                        this is critical and is DIFFERENT from date of issue.
                        Look for "Date on Board", "On Board Date", "Laden on Board" fields specifically.
        - shipper = Shipper / Exporter full name + address as one string
        - consignee = Consignee as written — could be "TO ORDER OF [name]" or a named party.
                    Capture exact wording including "TO ORDER OF" if present.
        - notify_party = Notify party full name + address + contact details as one string
        - vessel_name = Name of the vessel (e.g. "MV Ocean Star")
        - voyage_number = Voyage number / Voy No (e.g. "VOY-2026-042")
        - port_of_loading = Port where goods were loaded onto the vessel
        - port_of_discharge = Port where goods are to be discharged
        - place_of_delivery = Final place of delivery if different from port of discharge
        - container_numbers = array of all container numbers listed in cargo details
        - seal_numbers = array of all seal numbers corresponding to containers
        - goods_description = Full description of goods as written in the cargo/description column
        - hs_code = HS Code found within the goods description
        - number_of_packages = Total number of packages with unit
                            (e.g. "5,000 BAGS") — extract from NO. OF PKGS column or TOTALS row
        - gross_weight = Total gross weight with unit (e.g. "126.250 MT") from TOTALS row
        - volume_cbm = Total volume in CBM (e.g. "86.00 CBM") from TOTALS row
        - freight_terms = "FREIGHT PREPAID" or "FREIGHT COLLECT" as stated
        - incoterm = Trade term stated within the goods description or freight section
                    (e.g. "CIF Jebel Ali", "FOB Chennai")
        - number_of_originals = Number of original B/L copies issued (e.g. "THREE (3)")
        - eta_destination = ETA at destination port if mentioned
        - lc_number = Letter of Credit number referenced on the B/L
        - invoice_reference = Commercial Invoice number referenced on the B/L
        - booking_reference = Booking reference number if present
        - carrier = Name of the shipping line / carrier issuing the B/L

        Required JSON Schema:

        {
        "bl_number": null,
        "date_of_issue": null,
        "place_of_issue": null,
        "on_board_date": null,
        "shipper": null,
        "consignee": null,
        "notify_party": null,
        "vessel_name": null,
        "voyage_number": null,
        "port_of_loading": null,
        "port_of_discharge": null,
        "place_of_delivery": null,
        "container_numbers": [],
        "seal_numbers": [],
        "goods_description": null,
        "hs_code": null,
        "number_of_packages": null,
        "gross_weight": null,
        "volume_cbm": null,
        "freight_terms": null,
        "incoterm": null,
        "number_of_originals": null,
        "eta_destination": null,
        "lc_number": null,
        "invoice_reference": null,
        "booking_reference": null,
        "carrier": null
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
        print("\n✈️ RAW BillOfLadingLLMExtractor LLM OUTPUT:\n", raw_output)

        try:
            return self._safe_json_parse(raw_output)
        except Exception as e:
            print("❌ BillOfLadingLLMExtractor extraction failed:", str(e))
            return {
                "error": "BillOfLadingLLMExtractor_EXTRACTION_FAILED",
                "raw_llm_output": raw_output
            }
