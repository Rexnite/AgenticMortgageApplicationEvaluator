from pathlib import Path

from agents.document import DocumentAgent

agent = DocumentAgent()

document = agent.classify_file(
    Path("sample_application/payslip1.pdf")
)

print(document)