# evidences_api.py
# ------------------------------------------------------------
# Intelli3 - Evidence annotator (generic / low-bias version)
# - Uses YAKE to extract candidate terms
# - Uses Ollama (chat) to assign MI considering PARAGRAPH CONTEXT
# - Optionally extracts DiscursiveStrategy and ContextObject with MI
# - Minimal heuristics as fallback (generic TRIGGERS), no hard-gates
# ------------------------------------------------------------

from __future__ import annotations

import json
import re
import sys
import time
import shutil
import platform
import subprocess
from typing import List, Dict, Any, Callable, Optional, Tuple

import requests
from requests.exceptions import RequestException
from unidecode import unidecode
import yake

WORD_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+", re.UNICODE)

def _norm(s: str) -> str:
    """Normalize string for matching: lower + strip + remove accents."""
    return unidecode((s or "").strip().lower())

DEFAULT_MODEL = "mistral:7b-instruct"
OLLAMA_HOST = "http://127.0.0.1:11434"

INTELLIGENCES = [
    "Linguistic",
    "Logical-Mathematical",
    "Spatial",
    "Bodily-Kinesthetic",
    "Musical",
    "Interpersonal",
    "Intrapersonal",
    "Naturalist",
]

TRIGGERS: Dict[str, Dict[str, List[str]]] = {
    # -------------------------
    # Logical–Mathematical
    # -------------------------
    "Logical-Mathematical": {
        "pt": [
            # operações e raciocínio
            "raciocínio lógico", "lógica", "dedução", "inferência", "argumento lógico",
            "resolver problema", "solução", "estratégia de resolução", "analisar", "análise",
            "hipótese", "testar hipótese", "conjectura", "provar", "demonstração",
            "causa e efeito", "relações", "correlação", "padrão", "regularidade",

            # matemática escolar
            "cálculo", "calcular", "conta", "aritmética", "álgebra", "equação", "inequação",
            "função", "expressão", "variável", "constante", "fórmula", "teorema",
            "geometria analítica", "probabilidade", "estatística", "média", "mediana",
            "moda", "desvio padrão", "amostra", "frequência", "percentual", "porcentagem",
            "proporção", "razão", "fração", "regra de três",
            "medição", "medir", "unidade de medida", "comprimento", "massa", "peso", "volume",
            "tempo", "velocidade", "área", "perímetro",

            # organização lógico-formal
            "sequência", "seriação", "ordenar", "ordenação", "crescente", "decrescente",
            "comparar", "comparação", "maior que", "menor que", "igual", "equivalente",
            "classificar", "classificação", "categorizar", "agrupar", "critério",
            "conjunto", "subconjunto", "pertence", "interseção", "união",

            # representações e dados
            "tabela", "gráfico", "diagrama de barras", "histograma", "linha do tempo",
            "dados", "coletar dados", "analisar dados", "métrica", "medida", "estimativa",
            "algoritmo", "procedimento", "passo a passo", "fluxo lógico", "regra",
            "if", "então", "senão", "condição"
        ],
        "en": [
            "logical reasoning", "logic", "deduction", "inference", "logical argument",
            "problem solving", "solution", "strategy", "analyze", "analysis",
            "hypothesis", "test a hypothesis", "conjecture", "proof", "demonstration",
            "cause and effect", "relationships", "correlation", "pattern", "regularity",

            "calculation", "calculate", "arithmetic", "algebra", "equation", "inequality",
            "function", "expression", "variable", "constant", "formula", "theorem",
            "analytic geometry", "probability", "statistics", "mean", "median", "mode",
            "standard deviation", "sample", "frequency", "percentage", "ratio", "proportion",
            "fraction", "rule of three",

            "measurement", "measure", "unit of measure", "length", "mass", "weight", "volume",
            "time", "speed", "area", "perimeter",

            "sequence", "seriation", "sort", "ordering", "ascending", "descending",
            "compare", "comparison", "greater than", "less than", "equal", "equivalent",
            "classify", "classification", "categorize", "group", "criteria",
            "set", "subset", "belongs to", "intersection", "union",

            "table", "chart", "bar chart", "histogram", "timeline",
            "data", "collect data", "data analysis", "metric", "estimate",
            "algorithm", "procedure", "step by step", "rule",
            "if", "then", "else", "condition"
        ],
    },

    # -------------------------
    # Spatial (Espacial–visual)
    # -------------------------
    "Spatial": {
        "pt": [
            "mapa", "mapear", "cartografia", "planta baixa", "croqui",
            "diagrama", "fluxograma", "organograma", "esquema", "ilustração",
            "gráfico", "infográfico", "desenho", "esboço", "projeção",
            "geometria", "forma geométrica", "polígono", "sólidos geométricos",
            "cubo", "esfera", "cilindro", "cone", "pirâmide",
            "simetria", "assimetria", "perspectiva", "profundidade",
            "coordenadas", "eixo", "plano cartesiano", "escala",
            "orientação espacial", "direita", "esquerda", "acima", "abaixo",
            "frente", "atrás", "norte", "sul", "leste", "oeste",
            "visualizar", "visualização", "representação visual", "imagem mental",
            "rotacionar", "girar", "virar", "dobrar", "espelhar", "reflexão",
            "montagem espacial", "encaixe", "puzzle", "quebra-cabeça", "labirinto",
            "cores", "formas", "contornos", "figura", "posição", "trajeto", "rota"
        ],
        "en": [
            "map", "mapping", "cartography", "floor plan", "sketch",
            "diagram", "flowchart", "org chart", "schema", "illustration",
            "chart", "infographic", "drawing", "projection",
            "geometry", "geometric shape", "polygon", "3D solids",
            "cube", "sphere", "cylinder", "cone", "pyramid",
            "symmetry", "asymmetry", "perspective", "depth",
            "coordinates", "axis", "cartesian plane", "scale",
            "spatial orientation", "right", "left", "up", "down", "above", "below",
            "front", "back", "north", "south", "east", "west",
            "visualize", "visualization", "visual representation", "mental image",
            "rotate", "turn", "flip", "fold", "mirror", "reflection",
            "spatial assembly", "fit", "puzzle", "maze",
            "colors", "shapes", "outlines", "figure", "position", "path", "route"
        ],
    },

    # -------------------------
    # Linguistic
    # -------------------------
    "Linguistic": {
        "pt": [
            "leitura", "ler", "leitor", "compreensão leitora", "interpretação de texto",
            "texto", "gênero textual", "narrativa", "conto", "crônica", "poema", "poesia",
            "argumentação", "opinião", "debater ideias", "dissertar",
            "escrita", "escrever", "redação", "reescrita", "produção textual",
            "resumo", "paráfrase", "síntese", "resenha", "relatório", "artigo",
            "vocabulário", "léxico", "semântica", "sinônimo", "antônimo",
            "gramática", "morfologia", "sintaxe", "ortografia", "pontuação",
            "coerência", "coesão", "concordância", "regência",
            "pronúncia", "sonorização", "articulação", "fala", "oralidade",
            "contar história", "storytelling", "rimar", "rima", "aliteração",
            "definir", "explicar", "descrever", "interpretar", "analisar texto",
            "comunicação", "expressão verbal", "apresentação oral", "seminário",
            "dialogar", "entrevista", "questionário", "escrever cartas", "relatar"
        ],
        "en": [
            "reading", "read", "reader", "reading comprehension", "text interpretation",
            "text", "text genre", "narrative", "story", "chronicle", "poem", "poetry",
            "argumentation", "opinion", "discuss ideas", "essay writing",
            "writing", "rewrite", "text production",
            "summary", "paraphrase", "synthesis", "review", "report", "article",
            "vocabulary", "lexicon", "semantics", "synonym", "antonym",
            "grammar", "morphology", "syntax", "spelling", "punctuation",
            "coherence", "cohesion",
            "pronunciation", "articulation", "speech", "orality",
            "tell a story", "storytelling", "rhyme", "alliteration",
            "define", "explain", "describe", "interpret", "text analysis",
            "communication", "verbal expression", "oral presentation", "seminar",
            "dialogue", "interview", "questionnaire", "write letters", "narrate"
        ],
    },

    # -------------------------
    # Musical
    # -------------------------
    "Musical": {
        "pt": [
            "música", "musical", "melodia", "ritmo", "harmonia", "cantar", "cantiga",
            "canção", "instrumento", "violão", "flauta", "tambor", "percussão",
            "batida", "compasso", "tempo", "andamento", "pulso",
            "som", "sons", "auditivo", "ouvir", "escutar", "audição",
            "timbre", "altura", "intensidade", "volume", "grave", "agudo",
            "padrão sonoro", "repetição rítmica", "palmas", "bater palmas",
            "rimar cantando", "jogo rítmico", "eco musical", "imitar sons",
            "paisagem sonora", "sons da natureza", "sons do ambiente"
        ],
        "en": [
            "music", "musical", "melody", "rhythm", "harmony", "sing", "song",
            "instrument", "guitar", "flute", "drum", "percussion",
            "beat", "meter", "tempo", "pulse",
            "sound", "auditory", "listen", "hearing",
            "timbre", "pitch", "intensity", "loudness", "volume", "bass", "treble",
            "sound pattern", "rhythmic repetition", "clap", "clapping",
            "rhythm game", "musical echo", "imitate sounds",
            "soundscape", "sounds of nature", "ambient sounds"
        ],
    },

    # -------------------------
    # Bodily–Kinesthetic
    # -------------------------
    "Bodily-Kinesthetic": {
        "pt": [
            "coordenação motora", "motricidade", "movimento", "corpo",
            "manipular", "manusear", "pegar", "segurar", "apertar", "puxar",
            "recortar", "colar", "dobrar", "rasgar", "pintar", "modelar",
            "montar", "desmontar", "encaixar", "empilhar", "construir",
            "cortar", "amassar", "moldar", "esculpir",
            "gesto", "gestual", "mímica", "expressão corporal",
            "dramatização", "teatro", "encenar", "role-play",
            "dançar", "pular", "correr", "equilibrar", "coordenação",
            "atividade prática", "mão na massa", "experimento prático",
            "jogo físico", "brincadeira", "psicomotricidade"
        ],
        "en": [
            "motor coordination", "motor skills", "movement", "body",
            "manipulate", "handle", "grab", "hold", "press", "pull",
            "cut", "glue", "fold", "tear", "paint", "model",
            "assemble", "disassemble", "fit", "stack", "build",
            "sculpt", "mold",
            "gesture", "mime", "body expression",
            "dramatization", "theater", "act out", "role-play",
            "dance", "jump", "run", "balance", "coordination",
            "hands-on", "practical activity", "physical game", "psychomotor"
        ],
    },

    # -------------------------
    # Interpersonal
    # -------------------------
    "Interpersonal": {
        "pt": [
            "em grupo", "em dupla", "trabalho em grupo", "trabalho colaborativo",
            "colaboração", "cooperar", "cooperação", "ajuda mútua",
            "negociação", "consenso", "mediação", "liderança",
            "debate", "discussão", "roda de conversa", "argumentar em grupo",
            "apresentar para a turma", "apresentação coletiva",
            "papéis", "papel no grupo", "responsabilidade compartilhada",
            "dinâmica de grupo", "atividade coletiva", "interação social",
            "ouvir o colega", "feedback", "peer feedback", "avaliar pares"
        ],
        "en": [
            "group work", "pair work", "teamwork", "collaborative work",
            "collaboration", "cooperate", "cooperation", "mutual help",
            "negotiation", "consensus", "mediation", "leadership",
            "debate", "discussion", "circle time",
            "present to the class", "group presentation",
            "roles", "group roles", "shared responsibility",
            "group dynamics", "collective activity", "social interaction",
            "listen to peers", "feedback", "peer feedback", "peer assessment"
        ],
    },

    # -------------------------
    # Intrapersonal
    # -------------------------
    "Intrapersonal": {
        "pt": [
            "reflexão pessoal", "refletir", "autoavaliação", "autoconhecimento",
            "autorregulação", "metacognição", "metacognitivo",
            "diário", "diário de bordo", "portfólio", "processo", "autoria",
            "metas pessoais", "objetivos pessoais", "monitorar progresso",
            "emoção", "sentimento", "motivação", "interesse", "valores",
            "tomada de decisão", "escolha pessoal", "autonomia",
            "mindfulness", "atenção plena", "consciência", "identidade"
        ],
        "en": [
            "personal reflection", "reflect", "self-assessment", "self-knowledge",
            "self-regulation", "metacognition", "metacognitive",
            "journal", "learning journal", "portfolio", "process", "authorship",
            "personal goals", "track progress",
            "emotion", "feeling", "motivation", "interest", "values",
            "decision making", "personal choice", "autonomy",
            "mindfulness", "self-awareness", "identity"
        ],
    },

    # -------------------------
    # Naturalist
    # -------------------------
    "Naturalist": {
        "pt": [
            # seres vivos e classificação
            "animal", "animais", "planta", "plantas", "árvore", "árvores",
            "flor", "flores", "folha", "folhas", "semente", "sementes",
            "inseto", "insetos", "ave", "aves", "mamífero", "mamíferos",
            "réptil", "répteis", "anfíbio", "anfíbios", "peixe", "peixes",
            "fauna", "flora", "ser vivo", "seres vivos", "organismo", "organismos",
            "espécie", "espécies", "classificação biológica", "taxonomia",
            "gênero", "família", "reino", "identificar espécies", "catalogar",

            # ecologia e ambiente
            "natureza", "meio ambiente", "ambiental", "ecossistema", "bioma",
            "habitat", "nicho ecológico", "cadeia alimentar", "teia alimentar",
            "biodiversidade", "conservação", "preservação", "sustentabilidade",
            "poluição", "reciclagem", "reuso", "lixo", "resíduos",
            "clima", "tempo", "chuva", "temperatura", "solo", "rocha", "mineral",
            "água", "rios", "mar", "oceano", "lago", "floresta", "mata",
            "campo", "cerrado", "pantanal", "amazônia", "atlântico",

            # práticas típicas (observação/coleta)
            "observar", "observação da natureza", "coletar", "coleta", "amostra",
            "classificar plantas", "classificar animais", "identificar", "categorizar seres vivos",
            "horta", "jardinagem", "plantio", "cultivar", "compostagem",
            "trilha", "saída de campo", "campo", "exploração", "investigação ambiental",

             "naturalista", "naturalistas",
             "produto natural", "produtos naturais", "produto da natureza", "produtos da natureza",
             "coleção de produtos naturais", "coleções de produtos naturais",
             "coleta de produtos naturais", "coleta na natureza",
             "geologia", "geológico", "geologica", "geológicas", "elementos da natureza",
        ],
        "en": [
            "animal", "animals", "plant", "plants", "tree", "trees",
            "flower", "flowers", "leaf", "leaves", "seed", "seeds",
            "insect", "insects", "bird", "birds", "mammal", "mammals",
            "reptile", "reptiles", "amphibian", "amphibians", "fish", "fishes",
            "fauna", "flora", "living being", "living beings", "organism", "organisms",
            "species", "biological classification", "taxonomy", "genus", "family", "kingdom",
            "identify species", "catalog",

            "nature", "environment", "environmental", "ecosystem", "biome",
            "habitat", "ecological niche", "food chain", "food web",
            "biodiversity", "conservation", "preservation", "sustainability",
            "pollution", "recycling", "reuse", "waste", "residue",
            "climate", "weather", "rain", "temperature", "soil", "rock", "mineral",
            "water", "rivers", "sea", "ocean", "lake", "forest",
            "field trip", "outdoor observation", "collect", "collection", "sample",
            "garden", "gardening", "planting", "cultivate", "composting",
            "exploration", "environmental investigation"
        ],
    },
}

STOP_TOKENS = set(
    [
        "de","da","do","das","dos","a","o","as","os","e","em","no","na","nos","nas",
        "para","por","com","sem","um","uma","uns","umas","ao","aos","à","às",
        "que","se","sua","seu","suas","seus","ou","como",
        "preparacao","utilizacao",
        "professor","professora","aluno","alunos","aluna","alunas",
        "deve","devem","devera","deverao","deveria",
        "estimular","estimula","estimulo","estimulos",
        "atividade","atividades","tarefa","tarefas",
        "grupo","grupos","classe","turma",
        "planejar","planeja","planejado","planejada","planejamento",
        "solicitar","solicite","fazer","faz","faca","facao","feito",
        "possivel","sempre","antes","depois","durante","todo","toda",        
        "the","a","an","and","or","of","to","in","on","for","with","without","as","by","from","at","is","are","be","been",
    ]
)

RE_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$")
RE_MANY_PUNCT = re.compile(r"^[\s\-\_\=\*•·\.\,\:\;]{4,}$")

def clean_for_extraction(text: str) -> str:
    """
    Generic cleaning before YAKE/LLM:
    - fixes hyphenation across whitespace: "poste-\nrior" -> "posterior"
    - drops pure page-number lines and separator noise
    - normalizes spaces
    """
    t = (text or "")
    # remove rótulos muito frequentes que poluem YAKE/LLM
    t = re.sub(r"\b(Preparação|Preparacao|Utilização|Utilizacao)\s*:\s*", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"(\w)-\s+(\w)", r"\1\2", t, flags=re.UNICODE)

    lines: List[str] = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        if RE_PAGE_NUMBER_LINE.match(s):
            continue
        if RE_MANY_PUNCT.match(s):
            continue
        lines.append(ln)

    t = "\n".join(lines)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _clip01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def _yake_confidence(score: float) -> float:
    """
    YAKE: quanto menor o score, melhor.
    Converte o score em uma confiança no intervalo [0,1] usando 1/(1+score).
    """
    try:
        s = float(score)
    except Exception:
        return 0.5
    return _clip01(1.0 / (1.0 + max(0.0, s)))

def _pos_relevance(paragraph: str, term: str, floor: float = 0.50) -> float:
    """
    Aproxima a relevância pela posição do termo no texto:
    ocorrências mais cedo implicam maior relevância.
    """
    p = (paragraph or "")
    t = (term or "")
    if not p.strip() or not t.strip():
        return 0.75
    i = p.lower().find(t.lower())
    if i < 0:
        return 0.75
    pos = i / max(1, len(p))
    r = 1.0 - 0.50 * pos
    return _clip01(max(floor, r))

def _assign_confidence(source: str) -> float:
    source = (source or "").lower().strip()
    if source == "llm":
        return 0.90
    if source == "trigger":
        return 0.65
    if source == "forced":
        return 0.50
    return 0.60

def _ollama_up(host: str) -> bool:
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        return r.ok
    except Exception:
        return False

def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def start_ollama_daemon_if_needed(host: str, poll_sec: float = 1.0) -> None:
    if _ollama_up(host):
        return

    if not _which("ollama"):
        print("[WARN] 'ollama' não encontrado no PATH; não vou iniciar daemon automaticamente.", file=sys.stderr)
        return

    print("[INFO] Iniciando 'ollama serve'…", flush=True)
    popen_kwargs = {}
    if platform.system() == "Windows":
        popen_kwargs.update({"creationflags": 0x08000000})
    else:
        popen_kwargs.update({"start_new_session": True})

    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **popen_kwargs)
    except Exception as e:
        raise SystemExit(f"[ERRO] Não foi possível iniciar 'ollama serve': {e}")

    while not _ollama_up(host):
        time.sleep(poll_sec)

def ensure_model_available(model: str) -> None:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if r.ok:
            data = r.json()
            tags = [t.get("name") for t in data.get("models", []) if isinstance(t, dict)]
            if model in tags:
                return
    except Exception:
        pass

    if not _which("ollama"):
        raise SystemExit("[ERRO] 'ollama' não encontrado no PATH. Instale o Ollama ou ajuste o PATH.")

    try:
        print(f"[INFO] Baixando modelo {model} (ollama pull)...", flush=True)
        subprocess.run(["ollama", "pull", model], check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"[ERRO] Falha no 'ollama pull {model}': {e}")

def wait_ollama_ready(host: str, model: str, poll_sec: float = 1.0):
    while True:
        try:
            r = requests.get(f"{host}/api/tags", timeout=3)
            if r.ok:
                break
        except Exception:
            pass
        time.sleep(poll_sec)

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": "Responda APENAS com JSON."},
            {"role": "user", "content": '{"ping": true}'}
        ],
        "options": {"temperature": 0.0}
    }
    while True:
        try:
            r = requests.post(f"{host}/api/chat", json=payload, timeout=15)
            if r.ok:
                break
        except Exception:
            pass
        time.sleep(poll_sec)

def _extract_balanced_json(text: str) -> Optional[str]:
    s = text
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None

def _sanitize_json_guess(s: str) -> str:
    if "```" in s:
        s = s.replace("```json", "```").strip()
        parts = s.split("```")
        s = parts[1] if len(parts) >= 3 else parts[0]
    s = s.strip()
    guess = _extract_balanced_json(s) or s
    guess = re.sub(r",\s*([}\]])", r"\1", guess)
    return guess

def extract_keywords(text: str, n: int = 1, lang: str = "pt") -> List[Tuple[str, float]]:
    kw_extractor = yake.KeywordExtractor(lan=lang, n=n)
    return kw_extractor.extract_keywords(text)

def extract_keywords_1_3(text: str, lang: str = "pt") -> List[Tuple[str, float, int]]:
    """
    Return list of (term, best_score, n) choosing best (lowest) score per term for n=1..3.
    """
    bag: Dict[str, Tuple[float, int]] = {}
    for n in (1, 2, 3):
        for term, score in extract_keywords(text, n=n, lang=lang):
            t = (term or "").strip()
            if not t:
                continue
            prev = bag.get(t)
            if prev is None or float(score) < float(prev[0]):
                bag[t] = (float(score), int(n))
    out = [(t, s, n) for t, (s, n) in bag.items()]
    out.sort(key=lambda x: x[1])
    return out

def _canonical_tokens(term: str) -> List[str]:
    toks = [t for t in WORD_RE.findall(_norm(term)) if t and (t not in STOP_TOKENS) and len(t) > 2]
    return toks

def reduce_keyword_overlaps(items: List[Tuple[str, float, int]], max_keep: int = 12) -> List[Tuple[str, float, int]]:
    """
    Redução genérica de sobreposição:
    - normaliza os tokens e descarta vazios
    - mantém o melhor score por forma canônica
    - remove termos contidos em termos mais longos já mantidos
    """
    best: Dict[str, Tuple[str, float, int]] = {}

    for term, score, n in items:
        toks = _canonical_tokens(term)
        if not toks:
            continue
        can = " ".join(toks)
        prev = best.get(can)
        if prev is None or float(score) < float(prev[1]):
            best[can] = (can, float(score), int(n))

    kept = list(best.values())
    kept.sort(key=lambda x: x[1])

    final: List[Tuple[str, float, int]] = []
    for term, score, n in kept:
        drop = False
        for t2, _s2, _n2 in final:
            if term != t2 and (f" {term} " in f" {t2} "):
                drop = True
                break
        if not drop:
            final.append((term, score, n))
        if len(final) >= int(max_keep):
            break

    return final

def _matches_any(text_norm: str, triggers: List[str]) -> bool:
    for t in triggers:
        tn = _norm(t)
        if not tn:
            continue

        # token único: aceita flexões (prefixo)
        if " " not in tn and "-" not in tn and len(tn) >= 5:
            if re.search(rf"\b{re.escape(tn)}\w*\b", text_norm):
                return True

        # frase: match mais literal
        if re.search(rf"(^|\W){re.escape(tn)}($|\W)", text_norm):
            return True
        if tn in text_norm and len(tn) >= 5:
            return True
    return False

def map_text_to_intelligences_trigger(text: str, lang_hint: str = "pt") -> List[str]:
    """
    Mapeamento conservador por gatilhos:
    retorna as Inteligências Múltiplas (MIs) cujas strings de gatilho
    aparecem explicitamente no texto.
    """
    tn = _norm(text)
    hits: List[str] = []
    for mi in INTELLIGENCES:
        tr = TRIGGERS.get(mi, {})
        pool = tr.get(lang_hint, []) + tr.get("pt", []) + tr.get("en", [])
        if _matches_any(tn, pool):
            hits.append(mi)
    return hits

def ensure_nonempty_intelligences(text: str) -> List[str]:
    """
    Rede de segurança muito leve:
    retorna resultados apenas quando há sinais fortes; caso contrário,
    retorna uma lista vazia ([]).

    Importante: esta função opera sobre _norm(text)
    (texto em minúsculas e sem acentuação via unidecode).
    """
    t = _norm(text)

    # -----------------------------
    # Naturalist (seres vivos / ambiente)
    # -----------------------------
    if re.search(r"\b("
             r"animais?|planta(s)?|flora|fauna|ser(es)? vivo(s)?|organism|"
             r"especi|taxonom|biolog|ecossistem|bioma|habitat|"
             r"naturez|ambient|ecolog|biodivers|conserv|preserv|sustent|poluic|recicl|"
             r"naturalist(a|ic)(s)?|produt(o|os) natural(is)?|"
             r"solo|rocha(s)?|mineral|minerai(s)?|concha(s)?|cogumelo(s)?|folha(s)?|"
             r"rio(s)?|riach(o|os)?|oceano|mar|florest|mata|cerrad|pantanal|amazoni|"
             r"geolog"
             r")\b", t):
        return ["Naturalist"]

    # -----------------------------
    # Logical–Mathematical
    # -----------------------------
    if re.search(r"\b(calcul|equac|inequ|estat|probab|media|median|moda|desvio|amostr|frequenc|porcent|percent|propor|razao|frac|regra de tres|medid|unidade de medida|tabel|graf|compar|orden|seriac|sequenc|padrao|classific|categori|agrupar|criteri|conjunt|algoritm|proced|passo a passo|condic|se entao|if then)\b", t):
        return ["Logical-Mathematical"]

    # -----------------------------
    # Linguistic (saída verbal/escrita explícita)
    # -----------------------------
    if re.search(r"\b(ler|leitur|escrev|redac|reescrit|relat|descrev|explic|justific|argument|"
                r"apresent|seminar|dialog|entrevist|questionari|"
                r"nome(a|ar|ando|ie|iem)|"
                r"verbaliz(a|ar|ando)|"
                r"pronunc(i|ia|iar|iando)|"
                r"dizer|responder|indag|pergunt)\b", t):
        return ["Linguistic"]

    # -----------------------------
    # Spatial (artefatos/ações visuais-espaciais fortes)
    # -----------------------------
    if re.search(r"\b(mapa|cartograf|planta baixa|croqui|diagram|fluxogram|organogram|esquem|ilustr|infograf|desenh|esboc|projec|geometr|poligon|coordenad|eixo|plano cartesian|escala|orientacao espacial|direita|esquerda|acima|abaixo|frente|atras|norte|sul|leste|oeste|simetri|perspect|profund|visualiz|representacao visual|imagem mental|rotacion|girar|virar|dobrar|espelhar|reflexa|puzzle|quebra[- ]cabec|labirint)\b", t):
        return ["Spatial"]

    # -----------------------------
    # Musical (sinal auditivo/ritmo/melodia forte)
    # -----------------------------
    if re.search(r"\b(music|melod|ritm|harmon|cantar|cancao|cantig|instrument|percuss|batid|compass|andament|pulso|som\b|sons\b|auditiv|ouvir|escut|audi(c|)a(o|)|timbre|altura|intensidad|volume|grave|agud|padrao sonor|palmas|bater palmas|eco musical|imitar sons|paisagem sonor)\b", t):
        return ["Musical"]

    # -----------------------------
    # Bodily–Kinesthetic (manipulação/movimento físico)
    # -----------------------------
    if re.search(r"\b(coordenacao motora|motric|psicomotric|moviment|corpo|manipul|manuse|pegar|segur|apert|puxar|recort|colar|dobr|rasg|pintar|modelar|moldar|esculp|montar|desmont|encaix|empilh|constru|dramatiz|teatr|encen|mimic|gest|expressao corporal|dancar|pular|correr|equilibr|mao na massa|hands[- ]on|atividade pratica|experimento pratic|jogo fisic|brincadeir)\b", t):
        return ["Bodily-Kinesthetic"]

    # -----------------------------
    # Interpersonal (interação social/colaboração)
    # -----------------------------
    if re.search(r"\b(em grupo|trabalho em grupo|em dupla|duplas\b|colabor|cooper|ajuda mutua|negoci|consens|mediac|lider|debate|discussa(o|) em grupo|roda de conversa|atividade coletiva|interacao social|ouvir o colega|feedback|peer feedback|avaliar pares|peer assessment|papel no grupo|responsabilidade compartilh)\b", t):
        return ["Interpersonal"]

    # -----------------------------
    # Intrapersonal (reflexão/autoavaliação/metacognição)
    # -----------------------------
    if re.search(r"\b(reflexa(o|) pessoal|reflet|autoavali|autoconhec|autorregul|metacogn|diario|diario de bordo|portfolio|metas pessoais|objetivos pessoais|monitorar progresso|emoc(a|)o|sentiment|motivac|valores|tomada de decisao|escolha pessoal|autonom|mindful|atencao plena|conscienc|identidad)\b", t):
        return ["Intrapersonal"]

    return []

def idxs_to_labels_safe(idxs: Any) -> List[str]:
    labs: List[str] = []
    if not isinstance(idxs, list):
        return labs
    for x in idxs:
        try:
            xi = int(x)
        except Exception:
            continue
        if 0 <= xi < len(INTELLIGENCES):
            labs.append(INTELLIGENCES[xi])
    return labs

def normalize_items_with_intelligence(items: List[Dict[str, Any]], lang: str = "pt") -> List[Dict[str, Any]]:
    """
    Expande os itens para que cada registro contenha exatamente
    um rótulo de Inteligência Múltipla (MI).

    Chaves de saída:
    role, text, lang, intelligence, confidence, relevance
    """
    out_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for it in items or []:
        role = (it.get("role") or "").strip()
        text = (it.get("text") or "").strip()
        if not role or not text:
            continue

        base_conf = _clip01(it.get("confidence", 0.75))
        base_rel = _clip01(it.get("relevance", 0.75))

        labels = idxs_to_labels_safe(it.get("mi_idx"))
        assign_src = "llm" if labels else ""

        lab_direct = it.get("intelligence")
        if isinstance(lab_direct, str) and lab_direct in INTELLIGENCES and lab_direct not in labels:
            labels.append(lab_direct)
            if not assign_src:
                assign_src = "llm"

        if not labels:
            labels = map_text_to_intelligences_trigger(text, lang_hint=lang)
            assign_src = "trigger" if labels else ""

        if not labels:
            labels = ensure_nonempty_intelligences(text)
            assign_src = "forced" if labels else ""

        if not labels:
            continue

        conf = _clip01(base_conf * _assign_confidence(assign_src or "trigger"))

        for lab in labels:
            key = (role, text.lower(), lab)
            rec = out_map.get(key)
            if rec is None:
                out_map[key] = {
                    "role": role,
                    "text": text,
                    "lang": (it.get("lang") or lang or "pt"),
                    "intelligence": lab,
                    "confidence": conf,
                    "relevance": base_rel,
                }
            else:
                rec["confidence"] = max(float(rec.get("confidence", 0.0)), conf)
                rec["relevance"] = max(float(rec.get("relevance", 0.0)), base_rel)

    return list(out_map.values())

def call_llm_assign_for_keywords(
    keywords: List[str],
    paragraph: str = "",
    lang_hint: str = "pt",
    allow_zero: bool = False,
) -> Dict[str, List[int]]:
    """
    Retorna: {keyword -> [mi_idx, ...]} (máx. 0..8)

    - Utiliza o contexto do parágrafo para reduzir erros do tipo
    “recurso ≠ objetivo”.
    - Se allow_zero=True, o LLM pode retornar [] para termos ambíguos
    ou apenas de apoio.
    """
    if not keywords:
        return {}

    if not _ollama_up(OLLAMA_HOST):
        return {kw: [] for kw in keywords}

    opts = [{"idx": i, "label": lab} for i, lab in enumerate(INTELLIGENCES)]

    # sys_msg = (
    #     "Você classifica cada item (ROLE::texto) em qualquer quantidade de inteligências (0..8), usando o CONTEXTO."
    #     "Regra geral: só atribua uma inteligência quando houver ativação cognitiva clara (objetivo/tarefa), não por metadiscurso."
    #     "Se o item for editorial, rótulo, descrição neutra, ou apenas nome de material sem uso cognitivo, retorne []."
    #     "Se retornar múltiplos índices, ordene do mais relevante (primário) ao menos relevante."
    #     "Responda apenas em JSON."
    # )
    sys_msg = (
    "Você deve classificar cada item recebido (ROLE::texto) em 0..N inteligências, "
    "usando o CONTEXTO do parágrafo. "
    "IMPORTANTE: existem exatamente 8 inteligências, índices 0..7. "
    "Regra geral: só atribua inteligências quando houver ativação cognitiva clara (objetivo/tarefa), "
    "não por metadiscurso (título, preparação, observação editorial, catálogo, etc.). "
    "Se o item for apenas material/recurso/objeto sem uso cognitivo explícito, retorne []. "
    "Saída: JSON estrito, sem texto extra."
    )

    # contract = {"items": [{"text": "keyword_exata", "mi_idx": [2]}]}
    # contract = {"items": [{"text": "keyword_exata", "mi_idx": [2,0,4]}]}
    contract = {"items": [{"text": "ROLE::texto_exato", "mi_idx": [0,4]}]}    

    allow_rule = (
        "- Você PODE retornar mi_idx vazio [] se a keyword for apenas recurso/material ou for ambígua."
        if allow_zero
        else "- SEMPRE retorne pelo menos 1 índice por keyword (escolha a melhor MI)."
    )

    user_msg = f"""
CONTEXTO EDUCACIONAL (para desambiguar intenção vs recurso):
\"\"\"{paragraph}\"\"\"

ITENS PARA MAPEAMENTO (retorne EXATAMENTE cada string como enviada; mesma ordem; mesma quantidade):
{json.dumps(keywords, ensure_ascii=False)}

INTELIGÊNCIAS (índices 0..7):
{json.dumps(opts, ensure_ascii=False)}

========================
REGRAS GERAIS
========================
- Você pode retornar mi_idx vazio [] se o item for apenas:
  * editorial/metadiscurso (preparação, utilização, existe no mercado, observação, número de página)
  * rótulo/cabeçalho
  * nome de material/objeto sem operação cognitiva explícita
- Se retornar múltiplos índices, ordene por centralidade (primário primeiro).
- Evite “chute”: se não houver evidência, prefira [].

========================
ROLE-AWARE (COMO DECIDIR)
========================

1) Keyword::...
- Só marque MI se o termo indicar uma OPERAÇÃO/OBJETIVO cognitivo por si:
  Ex.: "ordem crescente", "comparar tamanhos", "classificar por cor", "medir", "sequência", "padrão", "tabela", "gráfico".
- Se for só nome de coisa ("cartões", "figuras", "saco plástico", "anéis"), prefira [].
  Exceção: objetos intrinsecamente visuais/espaciais ("mapa", "diagrama", "labirinto") podem ser Spatial.

2) DiscursiveStrategy::...
- Este é o sinal mais forte. Classifique pela ação principal do aluno.
- NÃO marque Logical só porque aparece "identificar" ou "classificar" de forma vaga.
  Logical-Mathematical (1) só quando houver CRITÉRIO explícito/forma de raciocínio:
  (maior/menor, ordem, sequência, por tamanho, por número, contagem, medida, regra, padrão, comparação objetiva, tabela/gráfico, algoritmo).
- Se a ação for “dizer o nome”, “verbalizar”, “pronunciar”, “sonorizar”, isso é Linguistic (0), não Logical.

3) ContextObject::...
- Objetos só ganham MI quando o objeto implica claramente uma inteligência no CONTEXTO.
  Ex.: "mapa/diagrama" -> Spatial; "instrumento/ritmo" -> Musical; "montar/encaixar" -> Bodily.
- Caso contrário, retorne [].

========================
NATURALIST (7) — REGRAS IMPORTANTES
========================
- Naturalist (7) entra quando o FOCO cognitivo é natureza/seres vivos/ambiente:
  observar/identificar espécies, flora/fauna, habitat, ecossistema, cadeia alimentar,
  classificação biológica (espécie/gênero/família), conservação/sustentabilidade etc.
- REGRA DE DOMÍNIO (IMPORTANTE):
  Se o item (Keyword/ContextObject/DiscursiveStrategy) contém entidades naturais explícitas
  (ex.: rochas, minerais, folhas, conchas, cogumelos, rios, plantas, animais, ecossistema),
  então Naturalist (7) PODE ser atribuído mesmo quando parece “material”,
  porque essas entidades são o CONTEÚDO da aprendizagem (não só ferramenta).  
- Se "animais" aparecerem apenas como FIGURAS/EXEMPLOS para treino de linguagem ou memória,
  Naturalist pode ser [] ou, no máximo, secundária — não force.

========================
BODILY vs LOGICAL (3 vs 1)
========================
- Bodily-Kinesthetic (3) quando há manipulação física (encaixar, montar, recortar, colar, empilhar, mover).
- Se houver manipulação + critério explícito (ex.: ordenar por tamanho enquanto encaixa),
  pode retornar [1,3] ou [3,1] conforme a centralidade (critério vs movimento).

========================
LINGUÍSTICA (0) — REGRA RESTRITIVA
========================
- Linguistic (0) entra quando há PRODUÇÃO/COMUNICAÇÃO como parte explícita da tarefa:
  relatar, descrever, explicar, justificar, argumentar, registrar por escrito, apresentar oralmente,
  discutir ideias, produzir resumo/síntese.
- Se o objetivo principal for outro (ex.: ordenar, comparar, classificar), Linguistic (0) pode aparecer
  como SECUNDÁRIA (depois da MI principal), porque é o “canal de saída”/evidência.
- NÃO marque Linguistic quando o item for apenas editorial/metadiscurso (preparação, utilização, catálogo),
  ou quando “linguagem” está implícita apenas por existir texto.

========================
ANTI-METADISCURSO (RETORNE [])
========================
Se o item for meta/editorial como:
"descreve", "apresenta", "menciona", "existe no mercado", "nome", "preparação",
"utilização", "material", "título", "página", "capítulo"
→ retorne [].

========================
FORMATO DE SAÍDA (JSON ESTRITO)
========================
- Retorne um objeto JSON com a chave "items".
- Deve haver exatamente o mesmo número de itens que a lista de entrada.
- Cada item deve ter:
  {{"text": "<string_exata_do_item_de_entrada>", "mi_idx": [<idx...>]}}

CONTRACT (exemplo de forma; não copie textos do exemplo):
{json.dumps(contract, ensure_ascii=False, indent=2)}
"""

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": DEFAULT_MODEL,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                "options": {"temperature": 0.0},
            },
            timeout=180,
        )
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content", "").strip()
    except RequestException:
        return {kw: [] for kw in keywords}

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        cleaned = _sanitize_json_guess(content)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {kw: [] for kw in keywords}

    items = parsed.get("items", []) if isinstance(parsed, dict) else []
    tmp: Dict[str, List[int]] = {}

    for it in items:
        text = str(it.get("text", "")).strip()
        idxs = it.get("mi_idx", [])
        if not text:
            continue
        fixed: List[int] = []
        if isinstance(idxs, list):
            for x in idxs:
                try:
                    xi = int(x)
                except Exception:
                    continue
                if 0 <= xi < len(INTELLIGENCES) and xi not in fixed:
                    fixed.append(xi)
        # tmp[text] = fixed[:2]
        tmp[text] = fixed

    out: Dict[str, List[int]] = {}
    for kw in keywords:
        out[kw] = tmp.get(kw, [])
    return out

def call_llm_discourse_and_context(
    paragraph: str,
    lang_hint: str = "pt",
    map_mi: bool = True,    
) -> List[Dict[str, Any]]:
    """
    Retorna itens brutos no formato:
      {"role":"DiscursiveStrategy","text":"...","mi_idx":[2]}
      {"role":"ContextObject","text":"...","mi_idx":[1]}
    """
    if not (paragraph or "").strip():
        return []
    if not _ollama_up(OLLAMA_HOST):
        return []

    opts = [{"idx": i, "label": lab} for i, lab in enumerate(INTELLIGENCES)]

    sys_msg = (
        "Você é um analista de dinâmicas de aprendizagem. "
        "Tarefa: decompor um parágrafo pedagógico em (1) DiscursiveStrategy (ações do aluno) "
        "e (2) ContextObject (recursos/materiais). "
        "REGRA CRÍTICA: NÃO INVENTE NADA. Cada 'text' deve ser uma SUBSTRING LITERAL do parágrafo "
        "(copie exatamente as palavras como aparecem). "
        "Não parafraseie, não resuma, não traga exemplos. "
        "Se não houver evidência literal, não retorne o item. "
        "Responda estritamente em JSON."
        if not map_mi else
        "Você é um analista de dinâmicas de aprendizagem especializado na teoria de Gardner. "
        "Tarefa: decompor um parágrafo pedagógico em (1) DiscursiveStrategy (ações do aluno) "
        "e (2) ContextObject (recursos/materiais) e atribuir mi_idx (0..7) quando houver ativação clara. "
        "REGRA CRÍTICA: NÃO INVENTE NADA. Cada 'text' deve ser uma SUBSTRING LITERAL do parágrafo "
        "(copie exatamente as palavras como aparecem). "
        "Não parafraseie, não resuma, não traga exemplos. "
        "Se não houver evidência literal, não retorne o item. "
        "Responda estritamente em JSON."
    )

    contract = (
        {"items": [{"role": "DiscursiveStrategy", "text": "estratégia"},
                   {"role": "ContextObject", "text": "objeto"}]}
        if not map_mi else
        {"items": [{"role": "DiscursiveStrategy", "text": "estratégia", "mi_idx": [2]},
                   {"role": "ContextObject", "text": "objeto", "mi_idx": [1]}]}
    )    

    user_msg = f"""
Analise o parágrafo e extraia a estrutura da atividade pedagógica.

PARÁGRAFO (fonte única da verdade):
\"\"\"{paragraph}\"\"\"

OPÇÕES DE INTELIGÊNCIAS (índices 0..7):
{json.dumps(opts, ensure_ascii=False)}

========================
O QUE EXTRAIR (FOCO EM AÇÕES DO ALUNO)
========================

1) DiscursiveStrategy (AÇÃO/TAREFA DO ALUNO)
- Extraia ações observáveis que o aluno deve executar.
- Deve ser curta e concreta, com 1 verbo principal E, quando houver, o complemento/objeto direto (até ~30 palavras),
  ex.: "descobrirem os elementos da natureza", "relatarem os elementos da natureza", "registrar essas sensibilidades".
- O campo "text" DEVE ser uma SUBSTRING LITERAL do parágrafo (copie exatamente).
- NÃO retorne itens editoriais/metadiscursivos como: "preparação", "utilização", "existe no mercado", "apresenta", "menciona", "descrição".
- Se não houver ação do aluno no parágrafo, retorne 0 DiscursiveStrategy.

2) ContextObject (RECURSO/MATERIAL)
- Extraia itens físicos/recursos citados e usados na atividade (ex.: cartões, figuras, saco plástico, etc.).
- O campo "text" DEVE ser uma SUBSTRING LITERAL do parágrafo (copie exatamente).
- Se o parágrafo não mencionar recurso/material de forma explícita, retorne 0 ContextObject.

========================
REGRAS IMPORTANTES (ANTI-ALUCINAÇÃO / ANTI-METADISCURSO)
========================
- NÃO invente ações/objetos que não estão no parágrafo.
- NÃO copie exemplos do enunciado: use apenas trechos do PARÁGRAFO.
- NÃO parafraseie: copie literalmente.
- NÃO repita itens: evite duplicatas.
- Se o trecho for apenas editorial/catálogo (ex.: "Existe no comércio..."), retorne 0 DiscursiveStrategy.
  Retorne ContextObject apenas se houver um objeto literal e relevante no texto.

========================
DIRETRIZES DE MAPEAMENTO (mi_idx) — SOMENTE SE map_mi=True
========================
- Use mi_idx apenas quando houver ativação cognitiva clara associada ao item.
- Se estiver incerto, retorne mi_idx [].
- Lembrete de índices: 0=Linguistic, 1=Logical-Mathematical, 2=Spatial, 3=Bodily-Kinesthetic, 4=Musical, 5=Interpersonal, 6=Intrapersonal, 7=Naturalist.

Heurísticas seguras:
- Logical-Mathematical (1): quando a ação envolve critério explícito (ordenar por tamanho, comparar, medir, contar, sequência/padrão, tabela/gráfico).
- Linguistic (0): quando a ação envolve linguagem como objetivo (ler, escrever, dizer nomes, pronunciar, verbalizar, sonorizar).
- Bodily-Kinesthetic (3): quando há manipulação/movimento (pegar, tirar, encaixar, montar, recortar, colar).
- Spatial (2): quando envolve forma/organização espacial (mapa, diagrama, labirinto, formas, posição/orientação).
- Musical (4): quando envolve ritmo/melodia/canto/sons como foco.
- Interpersonal (5): quando há dinâmica social explícita (em dupla/grupo, debater, negociar, colaborar).
- Intrapersonal (6): quando há reflexão/autoavaliação/autoconhecimento explícito.
- Naturalist (7): quando o FOCO cognitivo é natureza/seres vivos/ambiente (espécies, habitat, ecossistema, observação da natureza).

========================
FORMATO DE SAÍDA (JSON ESTRITO)
========================
Retorne um objeto JSON com a chave "items".
- Se map_mi=False: cada item tem apenas "role" e "text".
- Se map_mi=True: cada item tem "role", "text" e "mi_idx".
- "role" deve ser exatamente "DiscursiveStrategy" ou "ContextObject".
- "text" deve ser literal do parágrafo.

CONTRACT:
{json.dumps(contract, ensure_ascii=False, indent=2)}
"""

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": DEFAULT_MODEL,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                "options": {"temperature": 0.0},
            },
            timeout=180,
        )
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content", "").strip()
    except RequestException:
        return []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        cleaned = _sanitize_json_guess(content)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return []

    items = parsed.get("items", []) if isinstance(parsed, dict) else []
    out: List[Dict[str, Any]] = []
    par_low = (paragraph or "").lower()

    for it in items:
        role = str(it.get("role", "")).strip()
        text = str(it.get("text", "")).strip()
        if role not in ("DiscursiveStrategy", "ContextObject") or not text:
            continue
        # anti-alucinação residual: exige substring literal do parágrafo
        if text.lower() not in par_low:
            continue            

        mi_idx = it.get("mi_idx", [])
        fixed: List[int] = []
        if isinstance(mi_idx, list):
            for x in mi_idx:
                try:
                    xi = int(x)
                except Exception:
                    continue
                # if 0 <= xi < len(INTELLIGENCES):
                #     fixed.append(xi)
                if 0 <= xi < len(INTELLIGENCES) and xi not in fixed:
                    fixed.append(xi)                    

        # out.append({"role": role, "text": text, "mi_idx": fixed[:2]})
        out.append({"role": role, "text": text, "mi_idx": fixed})

    return out

def annotate_one_paragraph_keywords_with_llm(
    text: str,
    lang: str = "pt",
    use_n_1_3: bool = True,
    min_score: float = 0.35, 
    use_llm: bool = True,
    allow_keyword_zero_mi: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retorna itens de evidência contendo:
    role, text, lang, intelligence, confidence, relevance
    """
    paragraph = clean_for_extraction(text)
    if not paragraph:
        return []

    llm_available = bool(use_llm) and _ollama_up(OLLAMA_HOST)

    items_raw: List[Dict[str, Any]] = []
    seen = set()

    kws_scored = extract_keywords_1_3(paragraph, lang=lang) if use_n_1_3 else [
        (t, float(s), 1) for (t, s) in extract_keywords(paragraph, n=1, lang=lang)
    ]

    kws_scored.sort(key=lambda x: x[1]) 
    keywords_meta_raw = kws_scored[:160]

    if isinstance(min_score, (int, float)):
        filtered = [x for x in keywords_meta_raw if float(x[1]) <= float(min_score)]
        if len(filtered) >= 10:
            keywords_meta_raw = filtered

    keywords_meta = reduce_keyword_overlaps(keywords_meta_raw, max_keep=40)

    dc_no_mi: List[Dict[str, Any]] = []
    if llm_available:
        dc_no_mi = call_llm_discourse_and_context(paragraph=paragraph, lang_hint=lang, map_mi=False)

    candidates: List[str] = []
    def _add_candidate(s: str):
        if s and s not in candidates:
            candidates.append(s)

    for kw, _kw_score, _kw_n in keywords_meta:
        _add_candidate(f"Keyword::{kw}")

    for it in dc_no_mi:
        role = (it.get("role") or "").strip()
        t = (it.get("text") or "").strip()
        if role in ("DiscursiveStrategy", "ContextObject") and t:
            _add_candidate(f"{role}::{t}")

    llm_map_all: Dict[str, List[int]] = {}
    if llm_available and candidates:
        llm_map_all = call_llm_assign_for_keywords(
            keywords=candidates,
            paragraph=paragraph,
            lang_hint=lang,
            allow_zero=True,  # permitir "nenhuma MI" com mais liberdade
        )
    else:
        llm_map_all = {c: [] for c in candidates}

    for kw, kw_score, _kw_n in keywords_meta:
        key_txt = f"Keyword::{kw}"
        idxs = llm_map_all.get(key_txt, []) or []

        boost_labs = map_text_to_intelligences_trigger(kw, lang_hint=lang)
        if "Naturalist" in boost_labs:
            idx_nat = INTELLIGENCES.index("Naturalist")  # 7
            if idx_nat not in idxs:
                idxs.append(idx_nat)

        assign_src = "llm" if idxs else ""

        if not idxs:
            labs = map_text_to_intelligences_trigger(kw, lang_hint=lang)
            # idxs = [INTELLIGENCES.index(l) for l in labs if l in INTELLIGENCES][:2]
            idxs = [INTELLIGENCES.index(l) for l in labs if l in INTELLIGENCES]            
            if idxs:
                assign_src = "trigger"

        if not idxs and not allow_keyword_zero_mi:
            labs = ensure_nonempty_intelligences(kw)
            idxs = [INTELLIGENCES.index(l) for l in labs if l in INTELLIGENCES][:1]
            if idxs:
                assign_src = "forced"

        if not idxs:
            continue

        conf = _clip01(_yake_confidence(kw_score) * _assign_confidence(assign_src or "trigger"))
        rel = _pos_relevance(paragraph, kw)

        # for idx_mi in idxs[:2]:
        for idx_mi in idxs:        
            if 0 <= idx_mi < len(INTELLIGENCES):
                entry = {
                    "role": "Keyword",
                    "text": kw,
                    "lang": lang,
                    "mi_idx": [idx_mi],
                    "confidence": conf,
                    "relevance": rel,
                }
                key = (entry["role"], entry["text"].lower(), idx_mi)
                if key not in seen:
                    items_raw.append(entry)
                    seen.add(key)

    for it in dc_no_mi:
        role = (it.get("role") or "").strip()
        t = (it.get("text") or "").strip()
        if role not in ("DiscursiveStrategy", "ContextObject") or not t:
            continue
        key_txt = f"{role}::{t}"
        idxs = llm_map_all.get(key_txt, []) or []

        boost_labs = map_text_to_intelligences_trigger(t, lang_hint=lang)
        if "Naturalist" in boost_labs:
            idx_nat = INTELLIGENCES.index("Naturalist")
            if idx_nat not in idxs:
                idxs.append(idx_nat)

        assign_src = "llm" if idxs else ""

        if not idxs:
            labs = map_text_to_intelligences_trigger(t, lang_hint=lang)
            # idxs = [INTELLIGENCES.index(l) for l in labs if l in INTELLIGENCES][:2]
            idxs = [INTELLIGENCES.index(l) for l in labs if l in INTELLIGENCES]            
            if idxs:
                assign_src = "trigger"

        if not idxs:
            continue

        base_conf = 0.85
        conf = _clip01(base_conf * _assign_confidence(assign_src or "trigger"))
        rel = _pos_relevance(paragraph, t)

        # for idx_mi in idxs[:2]:
        for idx_mi in idxs:        
            entry = {
                "role": role,
                "text": t,
                "lang": lang,
                "mi_idx": [idx_mi],
                "confidence": conf,
                "relevance": rel,
            }
            key = (entry["role"], entry["text"].lower(), idx_mi)
            if key not in seen:
                items_raw.append(entry)
                seen.add(key)

    # 3) Normalize -> one MI per record
    items = normalize_items_with_intelligence(items_raw, lang=lang)

    return items

def annotate_doc_keywords_with_llm(
    doc: Dict[str, Any],
    use_n_1_3: bool = True,
    min_score: float = 0.60,
    use_llm: bool = True
) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    paras = doc.get("paragraphs", []) or []

    for p in paras:
        k = p.get("k")
        text = p.get("text", "") or ""
        lang = (p.get("lang") or "pt").strip().lower()

        if k is None or not text.strip():
            continue

        items = annotate_one_paragraph_keywords_with_llm(
            text=text,
            lang=lang if lang in ("pt", "en", "es") else "pt",
            use_n_1_3=use_n_1_3,
            min_score=min_score,
            use_llm=use_llm,
            allow_keyword_zero_mi=True,
        )
        out[int(k)] = items

    return out

def annotate_chunks_keywords_with_llm(
    chunks: List[str],
    lang: str = "pt",
    use_n_1_3: bool = True,
    min_score: float = 0.35,
    use_llm: bool = True
) -> Dict[int, List[Dict[str, Any]]]:
    result: Dict[int, List[Dict[str, Any]]] = {}
    for i, chunk in enumerate(chunks or []):
        doc = {"doc_id": f"chunk:{i}", "paragraphs": [{"k": 0, "text": chunk, "lang": lang}]}
        out = annotate_doc_keywords_with_llm(doc, use_n_1_3=use_n_1_3, min_score=min_score, use_llm=use_llm)
        result[i] = out.get(0, [])
    return result

def annotate_with_llm_default(**overrides):
    """
    Factory: returns annotate_fn(doc) -> Dict[int, List[evidences]]
    Defaults are generic; override as needed.
    """
    defaults = dict(use_n_1_3=True, min_score=0.35, use_llm=True)
    cfg = {**defaults, **overrides}

    def _runner(doc: dict) -> Dict[int, List[Dict[str, Any]]]:
        return annotate_doc_keywords_with_llm(
            doc,
            use_n_1_3=cfg["use_n_1_3"],
            min_score=cfg["min_score"],
            use_llm=cfg["use_llm"],
        )
    return _runner

def _read_doc_from_json(path: str) -> dict:
    import io
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_payload_from_doc(
    doc: dict,
    annotate_fn: Callable[[dict], Dict[int, List[Dict[str, Any]]]],
) -> dict:
    doc_id = doc.get("doc_id") or doc.get("id") or ""
    evidences_by_par = annotate_fn(doc)
    return {"doc_id": doc_id, "evidences": evidences_by_par}

def make_payload_from_s1_json_file(
    in_path: str,
    annotate_fn: Callable[[dict], Dict[int, List[Dict[str, Any]]]],
) -> Dict[int, List[Dict[str, Any]]]:
    doc = _read_doc_from_json(in_path)
    return annotate_fn(doc)

if __name__ == "__main__":
    import argparse
    import io

    ap = argparse.ArgumentParser(description="Generic MI evidence annotator (YAKE + LLM with paragraph context).")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON ({doc_id, paragraphs:[{k,text,lang}]})")
    ap.add_argument("--out", dest="out_path", default="-", help="Output JSON path (or '-' for stdout)")
    ap.add_argument("--offline", action="store_true", help="Offline mode: do not use LLM")
    ap.add_argument("--n13", action="store_true", help="Use YAKE n=1..3 (default True in this file)")
    ap.add_argument("--min-score", type=float, default=0.35, help="YAKE max score to keep (lower is better)")
    args = ap.parse_args()

    try:
        with io.open(args.in_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except FileNotFoundError:
        print(f"[ERRO] Arquivo não encontrado: {args.in_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERRO] JSON inválido em {args.in_path}: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.offline:
        start_ollama_daemon_if_needed(OLLAMA_HOST)
        ensure_model_available(DEFAULT_MODEL)
        print("[INFO] Aguardando Ollama e modelo ficarem prontos...", file=sys.stderr)
        wait_ollama_ready(OLLAMA_HOST, DEFAULT_MODEL)

    out_dict = annotate_doc_keywords_with_llm(
        doc,
        use_n_1_3=True if args.n13 else True,
        min_score=args.min_score,
        use_llm=not args.offline
    )

    if args.out_path == "-" or args.out_path.strip() == "":
        print(json.dumps(out_dict, ensure_ascii=False, indent=2))
    else:
        try:
            with io.open(args.out_path, "w", encoding="utf-8") as f:
                json.dump(out_dict, f, ensure_ascii=False, indent=2)
            print(f"[OK] Saída escrita em {args.out_path}")
        except Exception as e:
            print(f"[ERRO] Falha ao escrever {args.out_path}: {e}", file=sys.stderr)
            sys.exit(1)
