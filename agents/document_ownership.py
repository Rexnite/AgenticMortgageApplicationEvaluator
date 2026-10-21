import json

from agents.base_agent import BaseAgent
from models.document_ownership import DocumentOwnership


class DocumentOwnershipAgent(BaseAgent):
    name = "document_ownership"
    input_fields = ("passport", "employment", "payslips", "bank_statements")
    output_fields = ("document_ownership",)

    def extract(self, state):
        # Ownership needs names and document positions, not salaries, transactions,
        # passport numbers, or bank-account identifiers.
        evidence = {
            "passport": {"given_names": state.passport.given_names, "surname": state.passport.surname},
            "employment": {"employer_name": state.employment.employer_name},
            "payslips": [
                {"document_index": index, "employee_name": item.employee_name}
                for index, item in enumerate(state.payslips, start=1)
            ],
            "bank_statements": [
                {"document_index": index, "account_holder_name": item.account_holder_name}
                for index, item in enumerate(state.bank_statements, start=1)
            ],
        }

        prompt = f"""
            Ownership evidence

            {json.dumps(evidence)}

            You are a UK mortgage document verification specialist.

            Determine whether every submitted document belongs to the same mortgage applicant.

            The applicant should primarily be identified from the passport.

            If the passport is incomplete, infer the applicant from employment and the majority of matching documents.

            For every document determine

            - owner_name
            - belongs_to_applicant
            - confidence
            - reason

            Do not compare document quality.

            Only determine ownership.

            Return JSON.

            [
              {{
                  "document_type":"Payslip",
                  "document_index":1,
                  "owner_name":"John Robert Smith",
                  "belongs_to_applicant":true,
                  "confidence":0.98,
                  "reason":"Employee name matches applicant."
              }}
            ]
            """

        result = self.chat_json(
            "",
            prompt, list[DocumentOwnership]
        )
        

        document_ownership = [
            DocumentOwnership(**item)
            for item in result
        ]

        return document_ownership
    
    
    def applicant_bank_statements(self, state):

        valid = []
        
        bank_ownership = [
            o for o in state.document_ownership
            if o.document_type == "Bank Statement"
        ]

        for ownership, bank_statement in zip(
            bank_ownership,
            state.bank_statements
        ):

            if ownership.document_type == "Bank Statement":

                if ownership.belongs_to_applicant:
                    valid.append(bank_statement)
                else:
                    state.rejected_documents.append(bank_statement)

        return valid
    
    def applicant_payslips(self, state):

        valid = []
        
        payslip_ownership = [
            o for o in state.document_ownership
            if o.document_type == "Payslip"
        ]

        for ownership, payslip in zip(
            payslip_ownership,
            state.payslips
        ):

            if ownership.document_type == "Payslip":

                if ownership.belongs_to_applicant:
                    valid.append(payslip)
                else:
                    state.rejected_documents.append(payslip)

        return valid
    
    def run(self, state):

        state.current_step = "Document Ownership"

        state.document_ownership = self.extract(state)

        state.payslips = self.applicant_payslips(state)

        state.bank_statements = self.applicant_bank_statements(state)

        state.completed_steps.append(
            "Document Ownership"
        )

        return state
