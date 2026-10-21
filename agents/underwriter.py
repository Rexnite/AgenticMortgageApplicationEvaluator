from agents.base_agent import BaseAgent
from models.underwriting import UnderwritingDecision



class UnderwriterAgent(BaseAgent):
    name = "underwriter"
    input_fields = ("identity", "employmentverification", "income", "bank", "fraud",
        "affordability", "policy_results", "policy_assessment", "policy_context", "reflection")
    output_fields = ("underwriting",)

    def extract(self, state) -> UnderwritingDecision:
        
        prompt = f"""
        
            You are a senior UK mortgage underwriter.

            You have been provided with verified structured information extracted by previous specialist AI agents.

            The extraction and financial calculations have already been completed.

            Do NOT recalculate any values.
            Do NOT infer facts that are not supported by the supplied evidence.
            Do NOT invent relationships between employers, banks, or documents.
            If information is missing or inconsistent, identify it as a concern and recommend manual review.

            Your role is to receive the following assessment already completed:

            1. Identity Verification.
            2. Employment Verification.
            3. Income Verification
            4. Bank Verification.
            5. Fraud Assessment.
            6. Affordability assessment.

            Your Job is to recommend one of the following: 
            Approve
            Refer
            Decline
            
            Provide explanation for recommendation

            Specialist evidence (do not rediscover facts from raw documents):
            Identity: {state.identity.model_dump_json(indent=2)}
            Employment verification: {state.employmentverification.model_dump_json(indent=2)}
            Income verification: {state.income.model_dump_json(indent=2)}
            Bank verification: {state.bank.model_dump_json(indent=2)}
            Fraud: {state.fraud.model_dump_json(indent=2)}
            Affordability: {state.affordability.model_dump_json(indent=2)}
            Policy results: {[r.model_dump() for r in state.policy_results]}
            Policy summary: {state.policy_assessment.model_dump_json(indent=2)}
            Quality control: {state.reflection.model_dump_json(indent=2) if state.reflection else None}

            Retrieved policy context (supporting context and provenance only):
            {[item.model_dump() for item in state.policy_context]}

            The deterministic policy results above are authoritative.
            Retrieved context must never override or change those PASS / REFER / FAIL results.
            If no retrieved context is supplied, do not invent policy content.

            Fraud interpretation rules:
            • DETECTED means the stated inconsistency or quality condition was observed; it does not by itself prove fraud.
            • SUSPECTED is a concern requiring review, not a factual fraud accusation.
            • NOT_ASSESSED means evidence or technical capability was unavailable; never describe it as passed.
            • Do not claim document tampering, forgery, signature fraud, or image manipulation unless a supplied signal explicitly supports it.

            Decision Rules

            • Missing identity information → Refer
            • Conflicting employer information → Refer
            • Missing salary evidence → Refer
            • Fraud indicators → Refer or Decline depending on severity
            • Affordability score below 50 → Decline
            • Affordability score 50–70 → Refer
            • Affordability score above 70 with no significant concerns → Approve

            Return JSON matching exactly this schema:
            
            {{
              "decision": "Approve | Refer | Decline",
              "risk_rating": "Low | Medium | High",
              "confidence": 0.95,
              "strengths": [],
              "concerns": [],
              "conditions": [],
              "summary": ""
            }}
            """

        result = self.chat_json(
            "",
            prompt, UnderwritingDecision
        )
        
        underwriter = UnderwritingDecision(**result)

        return underwriter
    
    
    def run(self, state):

        state.current_step = "Underwriting Decision"

        state.underwriting = self.extract(state)
        state.route = state.underwriting.decision.upper()

        state.completed_steps.append("Underwriting Decision")

        return state
