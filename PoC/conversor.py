from pathlib import Path
from pdf2image import convert_from_path
import pytesseract

# No Ubuntu/WSL normalmente é só "tesseract" mesmo
pytesseract.pytesseract.tesseract_cmd = "tesseract"

def ocr_pdf_to_txt(pdf_path: Path, lang: str = "por", dpi: int = 300) -> None:
    print(f"\n=== Processando {pdf_path.name} ===")
    pages = convert_from_path(str(pdf_path), dpi=dpi)

    texto_total = []
    for i, page in enumerate(pages):
        print(f"OCR da página {i+1}/{len(pages)} de {pdf_path.name}...")
        txt = pytesseract.image_to_string(page, lang=lang)
        texto_total.append(txt)

    texto_final = "\n\n".join(texto_total)

    out_path = pdf_path.with_suffix(".txt")  # mesmo nome, extensão .txt
    with out_path.open("w", encoding="utf-8") as f:
        f.write(texto_final)

    print(f"-> Salvo: {out_path}")

if __name__ == "__main__":
    pasta = Path(".")  # pasta atual; mude se quiser outra
    pdfs = sorted(pasta.glob("*.pdf"))

    if not pdfs:
        print("Nenhum PDF encontrado na pasta atual.")
    else:
        for pdf in pdfs:
            ocr_pdf_to_txt(pdf)
