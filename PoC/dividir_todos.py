from pathlib import Path
from pypdf import PdfReader, PdfWriter
import copy

# === CONFIGURAÇÕES ===
# Pasta onde estão os PDFs (use "." para pasta atual)
INPUT_DIR = Path(".")

# === PROCESSAMENTO ===
for pdf_path in INPUT_DIR.glob("*.pdf"):
    # pular arquivos que já são resultados (_1 ou _2)
    if pdf_path.stem.endswith("_1") or pdf_path.stem.endswith("_2"):
        print(f"Pulando (já processado): {pdf_path.name}")
        continue

    print(f"Processando: {pdf_path.name}")

    reader = PdfReader(str(pdf_path))

    if len(reader.pages) != 1:
        print(
            f"  [AVISO] {pdf_path.name} tem {len(reader.pages)} páginas. "
            "Este script supõe PDFs com 1 página, então vou pular."
        )
        continue

    page = reader.pages[0]

    # Tamanho da página
    mb = page.mediabox
    x0 = float(mb.left)
    y0 = float(mb.bottom)
    x1 = float(mb.right)
    y1 = float(mb.top)

    # Meio vertical (corte horizontal: cima/baixo)
    mid_y = (y0 + y1) / 2.0

    # Metade de cima
    page_top = copy.deepcopy(page)
    page_top.mediabox.bottom = mid_y
    page_top.cropbox.bottom = mid_y

    # Metade de baixo
    page_bottom = copy.deepcopy(page)
    page_bottom.mediabox.top = mid_y
    page_bottom.cropbox.top = mid_y

    # Nomes de saída
    stem = pdf_path.stem  # nome sem extensão
    out1 = pdf_path.with_name(f"{stem}_1.pdf")
    out2 = pdf_path.with_name(f"{stem}_2.pdf")

    # Grava arquivos
    writer_top = PdfWriter()
    writer_top.add_page(page_top)
    with open(out1, "wb") as f:
        writer_top.write(f)

    writer_bottom = PdfWriter()
    writer_bottom.add_page(page_bottom)
    with open(out2, "wb") as f:
        writer_bottom.write(f)

    print(f"  -> Gerado: {out1.name}, {out2.name}")

print("Concluído.")
