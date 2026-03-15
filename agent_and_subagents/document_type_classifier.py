import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


class DocumentTypeClassifier:
    """
    Classifies Trade Finance documents into one of:
    - INVOICE
    - BILL_OF_EXCHANGE
    - BILL_OF_LADING
    - CERTIFICATE_OF_ORIGIN
    - INSPECTION_CERTIFICATE
    - INSURANCE_CERTIFICATE
    - LETTER_OF_CREDIT
    - PACKING_LIST
    """

    ALLOWED_TYPES = [
       "INVOICE",
        "BILL_OF_EXCHANGE",
        "BILL_OF_LADING",
        "CERTIFICATE_OF_ORIGIN",
        "INSPECTION_CERTIFICATE",
        "INSURANCE_CERTIFICATE",
        "LETTER_OF_CREDIT",
        "PACKING_LIST",
    ]

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        if not self.deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT is not set")

    def classify(self, document):
        """
        Classifies a trade finance document into exactly one of the 8 allowed types.
        Returns the type string e.g. "INVOICE", "BILL_OF_LADING" etc.
        Raises ValueError if classification result is not in ALLOWED_TYPES.
        """

        system_prompt = """You are a Trade Finance document classifier used by a bank.

                        Your task is to read the document text and classify it into EXACTLY ONE of these 8 types:

                        INVOICE
                        BILL_OF_EXCHANGE
                        BILL_OF_LADING
                        CERTIFICATE_OF_ORIGIN
                        INSPECTION_CERTIFICATE
                        INSURANCE_CERTIFICATE
                        LETTER_OF_CREDIT
                        PACKING_LIST

                        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        IDENTIFICATION RULES FOR EACH TYPE:
                        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                        1. INVOICE
                        KEY SIGNALS:
                        - Title contains "COMMERCIAL INVOICE" or "PROFORMA INVOICE"
                        - Contains: Invoice Number, Invoice Date, Seller/Exporter, Buyer/Consignee
                        - Contains pricing table: Description, HS Code, Quantity, Unit Price, Total Amount
                        - Contains FOB/CIF value breakdown (Sub Total, Freight, Insurance, Total CIF)
                        - Contains bank details for payment (SWIFT, Account No, IFSC)
                        - References L/C number
                        MUST NOT confuse with: Packing List (no pricing), Bill of Exchange (no goods table)

                        2. BILL_OF_EXCHANGE
                        KEY SIGNALS:
                        - Title contains "BILL OF EXCHANGE"
                        - Contains: "PAY TO THE ORDER OF" — the classic negotiable instrument phrase
                        - Contains: "AT SIGHT" or usance period (e.g. "90 days after sight")
                        - Contains: Drawer (Beneficiary/Exporter) and Drawee (Issuing Bank)
                        - Contains: Amount in figures AND in words
                        - References: L/C Number, L/C Date, L/C Type (Irrevocable, Sight)
                        - Contains: Confirming Bank, Invoice No, B/L No
                        - Legal reference: "Negotiable Instruments Act" or "UCP 600"
                        - Contains: "BANKER'S ENDORSEMENT / ACCEPTANCE" section
                        MUST NOT confuse with: Invoice (no "PAY TO THE ORDER OF"), LC (no drawer/drawee)

                        3. BILL_OF_LADING
                        KEY SIGNALS:
                        - Title contains "BILL OF LADING" (B/L)
                        - Issued by a SHIPPING LINE / CARRIER (not a bank, not a chamber)
                        - Contains: B/L Number, Shipper, Consignee, Notify Party
                        - Contains: Vessel Name, Voyage Number, Port of Loading, Port of Discharge
                        - Contains: Container Number, Seal Number, cargo description table
                        - Contains: "DATE ON BOARD" (the shipped-on-board date)
                        - Contains: Freight Terms (PREPAID / COLLECT), No. of Original B/Ls
                        - Signed by: Carrier / Authorised Agent / Master
                        MUST NOT confuse with: Airway Bill (sea vs air), Packing List (no vessel/container)

                        4. CERTIFICATE_OF_ORIGIN
                        KEY SIGNALS:
                        - Title contains "CERTIFICATE OF ORIGIN"
                        - Issued by: Chamber of Commerce or authorised government body (e.g. DGFT)
                        - Contains numbered fields: Field 1 (Exporter), Field 2 (Consignee),
                            Field 4 (Country of Origin), Field 5 (Country of Destination)
                        - Contains: Declaration by Exporter (Field 13) AND Certification by Authority (Field 14)
                        - Mentions: "NON-PREFERENTIAL" or "PREFERENTIAL"
                        - Contains: Cert No, HS Code, Quantity, Net Weight, Gross Weight
                        MUST NOT confuse with: Inspection Certificate (no origin declaration), Invoice

                        5. INSPECTION_CERTIFICATE
                        KEY SIGNALS:
                        - Title contains "INSPECTION CERTIFICATE" or "QUALITY CERTIFICATE"
                        - Issued by: Third-party inspection company (e.g. SGS, Bureau Veritas, Intertek)
                        - Contains: Inspection Findings table with Parameters, Specifications, Results, Conformity
                        - Contains: Inspector Name, Inspector ID, Accreditation details
                        - Contains: Inspection Location (warehouse address)
                        - Contains: Quantity Verified table (Stated vs Inspected vs Variance)
                        - Contains: CONCLUSION statement ("GOODS FOUND TO BE IN CONFORMITY...")
                        MUST NOT confuse with: Certificate of Origin (no test parameters), Insurance Certificate

                        6. INSURANCE_CERTIFICATE
                        KEY SIGNALS:
                        - Title contains "INSURANCE CERTIFICATE" or "MARINE CARGO INSURANCE"
                        - Issued by: Insurance company (e.g. New India Assurance, Oriental Insurance)
                        - Contains: Policy Number, Sum Insured, Coverage Basis (CIF + 10%)
                        - Contains: Institute Cargo Clauses type — ICC (A), ICC (B), or ICC (C)
                        - Contains: War Clauses, Strikes Clauses (INCLUDED / NOT INCLUDED)
                        - Contains: Insured (Assured) and Beneficiary
                        - Contains: Claims Payable At + Survey Agent at Destination
                        - Contains: Voyage From / Voyage To / On Board Date
                        - IRDAI registration number (for Indian insurers)
                        MUST NOT confuse with: Inspection Certificate (no policy/premium), LC

                        7. LETTER_OF_CREDIT
                        KEY SIGNALS:
                        - Title contains "LETTER OF CREDIT" or "DOCUMENTARY CREDIT"
                        - Issued by: A BANK (Issuing Bank)
                        - Contains: LC Number, Date of Issue, Expiry Date, Expiry Place
                        - Contains: Applicant (Buyer) and Beneficiary (Seller)
                        - Contains: Advising Bank details
                        - Contains: LC Amount with Tolerance (e.g. +/- 5%)
                        - Contains: "DOCUMENTS REQUIRED" checklist (Invoice, B/L, Packing List, etc.)
                        - Contains: "SPECIAL CONDITIONS" section
                        - References: UCP 600, ICC Publication No. 600
                        - Contains: Latest Shipment Date, Partial Shipment / Transhipment allowed/not
                        MUST NOT confuse with: Bill of Exchange (no documents required list), Invoice

                        8. PACKING_LIST
                        KEY SIGNALS:
                        - Title contains "PACKING LIST"
                        - Contains: Package breakdown table with PKG NO., Description per package
                        - Contains per-row: No. of Bags, Net Wt/Bag, Gross Wt/Bag, Total Net, Total Gross
                        - Contains: Marks & Numbers per package AND a master marks block at bottom
                        - Contains: Container Information section (Container No., Seal No., Size/Type)
                        - Contains: Shipment Summary (Total Packages, Total Net Weight, Total Gross, CBM)
                        - NO pricing / unit prices / invoice amounts
                        MUST NOT confuse with: Invoice (no pricing in packing list), Bill of Lading

                        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        PRIORITY / DISAMBIGUATION RULES:
                        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                        - If "CERTIFICATE OF ORIGIN" appears in title → CERTIFICATE_OF_ORIGIN (not INSPECTION_CERTIFICATE)
                        - If "INSPECTION CERTIFICATE" appears + issued by SGS/BV/Intertek → INSPECTION_CERTIFICATE
                        - If "INSURANCE" or "MARINE CARGO" appears + Policy No + Sum Insured → INSURANCE_CERTIFICATE
                        - If "BILL OF EXCHANGE" appears + "PAY TO THE ORDER OF" → BILL_OF_EXCHANGE (not INVOICE)
                        - If "BILL OF LADING" appears + Shipping Line issuer + Container details → BILL_OF_LADING
                        - If "PACKING LIST" appears + NO prices → PACKING_LIST (not INVOICE)
                        - If "LETTER OF CREDIT" + issued by bank + Documents Required list → LETTER_OF_CREDIT
                        - If "COMMERCIAL INVOICE" + pricing table + bank details → INVOICE

                        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        OUTPUT RULE — CRITICAL:
                        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        Output ONLY the document type string. Nothing else.
                        No explanation. No punctuation. No quotes.
                        Valid outputs: INVOICE | BILL_OF_EXCHANGE | BILL_OF_LADING | CERTIFICATE_OF_ORIGIN |
                                    INSPECTION_CERTIFICATE | INSURANCE_CERTIFICATE | LETTER_OF_CREDIT | PACKING_LIST"""

        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(document, indent=2)},
            ],
        )

        result = response.choices[0].message.content.strip()

        # Hard safety guard
        if result not in self.ALLOWED_TYPES:
            raise ValueError(f"Unexpected classification result: {result}")

        return result
