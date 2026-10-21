from agents.base_agent import BaseAgent
from models.affordability import AffordabilityData
from services.calculations import annual_income, average_monthly_expenses, average_monthly_income, debt_to_income


class AffordabilityAgent(BaseAgent):
    name = "affordability"
    input_fields = ("payslips", "bank_statements")
    output_fields = ("affordability",)

    def extract(self, state) -> AffordabilityData:
        annual = annual_income(state.employment, state.payslips)
        gross = average_monthly_income(state.payslips, gross=True)
        net = average_monthly_income(state.payslips)
        expenses = average_monthly_expenses(state.bank_statements)
        disposable = net - expenses if net is not None and expenses is not None else None
        ratio = debt_to_income(expenses, gross)
        score = max(0, min(100, round(100 - ratio * 100))) if ratio is not None else None
        if score is not None and disposable is not None and disposable <= 0:
            score = min(score, 40)
        recommendation = "Refer" if score is None else "Decline" if score < 50 else "Refer" if score < 70 else "Approve"
        return AffordabilityData(
            annual_salary=annual, average_monthly_gross_income=gross,
            average_monthly_net_income=net, average_monthly_expenses=expenses,
            disposable_income=disposable, debt_to_income_ratio=ratio,
            affordability_score=score, recommendation=recommendation,
            confidence=1.0 if None not in (gross, net, expenses) else 0.5,
        )

    def run(self, state):
        state.current_step = "Affordability Assessment"
        state.affordability = self.extract(state)
        state.completed_steps.append("Affordability Assessment")
        return state
