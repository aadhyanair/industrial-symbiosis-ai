from html import unescape
from pathlib import Path
from zipfile import ZipFile
import re


source = Path("data/material_compatibitlity.csv.docx")
target = Path("data/processed/material_industry_compatibility.csv")

xml = ZipFile(source).read("word/document.xml").decode("utf-8")
text = re.sub(r"<w:br[^>]*/>", "\n", xml)
text = re.sub(r"</w:p>", "\n", text)
text = re.sub(r"<[^>]+>", "", text)
text = unescape(text)

header = "chemical_name,primary_naics"
start = text.find(header)
if start < 0:
    raise ValueError("The uploaded document does not contain the expected compatibility CSV header.")

csv_text = text[start:].strip()
rows = [row.strip() for row in csv_text.splitlines() if row.strip()]
if len(rows) < 2:
    raise ValueError("The uploaded document does not contain compatibility records.")

target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"Created {target} with {len(rows) - 1:,} records.")