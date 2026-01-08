from pypdf import PdfReader, PdfWriter
import copy

# === CONFIGURAÇÕES ===
INPUT_PDF = "169_170.pdf"              # seu PDF original (com 1 página)
OUTPUT_TOP = "atividade_parte1.pdf"    # metade de cima
OUTPUT_BOTTOM = "atividade_parte2.pdf" # metade de baixo

# === LEITURA DO PDF ===
reader = PdfReader(INPUT_PDF)

if len(reader.pages) != 1:
    raise ValueError("Este script supõe que o PDF tenha exatamente 1 página.")

page = reader.pages[0]

# Caixa original da página (tamanho)
mb = page.mediabox
x0 = float(mb.left)
y0 = float(mb.bottom)
x1 = float(mb.right)
y1 = float(mb.top)

# Ponto médio na vertical (meio da página)
mid_y = (y0 + y1) / 2.0

# === DUPLICAR PÁGINA E DEFINIR CORTES ===
# Metade de cima
page_top = copy.deepcopy(page)
page_top.mediabox.bottom = mid_y
page_top.cropbox.bottom = mid_y

# Metade de baixo
page_bottom = copy.deepcopy(page)
page_bottom.mediabox.top = mid_y
page_bottom.cropbox.top = mid_y

# === GERAR PDF DA METADE DE CIMA ===
writer_top = PdfWriter()
writer_top.add_page(page_top)

with open(OUTPUT_TOP, "wb") as f:
    writer_top.write(f)

# === GERAR PDF DA METADE DE BAIXO ===
writer_bottom = PdfWriter()
writer_bottom.add_page(page_bottom)

with open(OUTPUT_BOTTOM, "wb") as f:
    writer_bottom.write(f)

print("Feito!")
print(f"Metade de cima salva em: {OUTPUT_TOP}")
print(f"Metade de baixo salva em: {OUTPUT_BOTTOM}")
