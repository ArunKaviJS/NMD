import os
import json
import re
from openai import AzureOpenAI
from datetime import datetime, timezone, timedelta


class LetterOfCreditLLMExtractor:
    """
    Extract mandatory fields from LETTER OF CREDIT (LC)
    Banks reject documents if LC data mismatches
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
    
    
    def _get_ist_date(self) -> str:
        """Return current date in Indian Standard Time (IST = UTC+5:30) as DD MMM YYYY string."""
        ist = timezone(timedelta(hours=5, minutes=30))
        return datetime.now(ist).strftime("%d %B %Y")
    
    def _calculate_presentation_status(self, extracted: dict) -> dict:
        """
        Calculate presentation deadline (latest_shipment_date + 21 days)
        and whether it has expired based on current IST date.
        Returns two extra keys to merge into extracted result.
        """
        try:
            ist = timezone(timedelta(hours=5, minutes=30))
            today = datetime.now(ist).date()

            latest_shipment_raw = extracted.get("latest_shipment_date")
            if not latest_shipment_raw:
                return {
                    "presentation_deadline": None,
                    "presentation_period_expired": "Unable to calculate — latest shipment date not found"
                }

            # Try multiple date formats
            for fmt in ("%d %B %Y", "%B %d, %Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d %b %Y"):
                try:
                    shipment_date = datetime.strptime(latest_shipment_raw.strip(), fmt).date()
                    break
                except ValueError:
                    continue
            else:
                return {
                    "presentation_deadline": None,
                    "presentation_period_expired": f"Unable to parse shipment date: {latest_shipment_raw}"
                }

            presentation_deadline = shipment_date + timedelta(days=21)
            is_expired = today > presentation_deadline

            return {
                "presentation_deadline": presentation_deadline.strftime("%d %B %Y"),
                "presentation_period_expired": (
                    f"YES — Expired on {presentation_deadline.strftime('%d %B %Y')} "
                    f"(Today IST: {today.strftime('%d %B %Y')})"
                    if is_expired else
                    f"NO — Deadline is {presentation_deadline.strftime('%d %B %Y')} "
                    f"(Today IST: {today.strftime('%d %B %Y')}, "
                    f"{(presentation_deadline - today).days} day(s) remaining)"
                )
            }

        except Exception as e:
            return {
                "presentation_deadline": None,
                "presentation_period_expired": f"Calculation error: {str(e)}"
            }

    def extract(self, normalized_doc):
        """
        Extract LETTER OF CREDIT mandatory fields
        """
        current_ist_date = self._get_ist_date()

        system_prompt = """
       You are a Trade Finance Letter of Credit (LC) Extraction Engine.

        Document Type: IRREVOCABLE DOCUMENTARY LETTER OF CREDIT

        Today's Date (Indian Standard Time): {current_ist_date}

        Rules:
        - Extract values ONLY if explicitly present in the document
        - DO NOT guess or infer
        - DO NOT explain or add commentary
        - If a field is not clearly mentioned, return null
        - Output MUST be valid JSON only — no markdown, no extra text
        - Do NOT add extra fields
        - Do NOT rename fields

        Field Mapping Rules:
        - applicant = Buyer / Applicant (full name + address as one string)
        - beneficiary = Seller / Beneficiary (full name + address as one string)
        - issuing_bank = Bank issuing the LC (name + SWIFT if present)
        - advising_bank = Bank advising the LC to beneficiary (name + SWIFT if present)
        - lc_amount = numeric amount as string (e.g. "125000.00")
        - currency = currency code (e.g. "USD")
        - tolerance = tolerance percentage (e.g. "+/- 5%")
        - incoterm = trade term (e.g. "CIF Jebel Ali")
        - partial_shipment = "ALLOWED" or "NOT ALLOWED"
        - transhipment = "ALLOWED" or "NOT ALLOWED"
        - commodity_description = exact wording of goods/commodity as written in LC
        - hs_code = HS / Harmonized Code of goods
        - quantity = full quantity with units as written
        - unit_price = unit price with basis as written
        - documents_required = array of strings, each document listed as a separate item
        - presentation_period = presentation period explicitly stated in LC (e.g. "21 days from B/L date"), return null if not stated
        - special_conditions = array of strings, each special condition as a separate item
        - latest_shipment_date = parse and return in format "DD Month YYYY" (e.g. "20 March 2026")

        Required JSON Schema:

        {{
        "lc_number": null,
        "date_of_issue": null,
        "date_of_expiry": null,
        "place_of_expiry": null,
        "applicant": null,
        "beneficiary": null,
        "issuing_bank": null,
        "advising_bank": null,
        "lc_amount": null,
        "currency": null,
        "tolerance": null,
        "incoterm": null,
        "port_of_loading": null,
        "port_of_discharge": null,
        "latest_shipment_date": null,
        "partial_shipment": null,
        "transhipment": null,
        "commodity_description": null,
        "hs_code": null,
        "quantity": null,
        "unit_price": null,
        "documents_required": [],
        "presentation_period": null,
        "special_conditions": []
        }}"""

        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(normalized_doc)},
            ],
        )

        raw_output = response.choices[0].message.content
        

        # Debug – disable in production
        print("\n🏦 RAW LC LLM OUTPUT:\n", raw_output)

        try:
            extracted = self._safe_json_parse(raw_output)

            # Calculate presentation deadline & expiry status in Python (IST)
            presentation_status = self._calculate_presentation_status(extracted)
            extracted.update(presentation_status)

            return extracted
        
        except Exception as e:
            print("❌ LC extraction failed:", str(e))
            return {
                "error": "LC_EXTRACTION_FAILED",
                "raw_llm_output": raw_output
            }
