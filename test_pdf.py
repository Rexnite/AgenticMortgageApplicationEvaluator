from pathlib import Path

from tools.pdf_reader import PDFReader

text = PDFReader.extract_text(
    Path("sample_application/payslip1.pdf")
)

print(text[:1000])