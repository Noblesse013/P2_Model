"""
Extract images from System_Report.pdf into the figures/ folder.
Requires: pymupdf (install via `pip install pymupdf`).
"""
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "System_Report.pdf"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not PDF_PATH.exists():
    print(f"PDF not found: {PDF_PATH}")
    raise SystemExit(1)

print(f"Opening {PDF_PATH}")
doc = fitz.open(str(PDF_PATH))
count = 0
for pno in range(len(doc)):
    page = doc[pno]
    images = page.get_images(full=True)
    if not images:
        continue
    for i, img in enumerate(images, start=1):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        ext = base_image.get("ext", "png")
        out_name = f"system_report_page{pno+1}_img{i}.{ext}"
        out_path = OUT_DIR / out_name
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {out_path}")
        count += 1

print(f"Extraction complete. {count} images saved to {OUT_DIR}")
