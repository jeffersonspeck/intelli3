from __future__ import annotations

import re
import json
import time
import shutil
import platform
import subprocess
import sys
from typing import List, Dict, Any, Callable, Optional
from math import exp

import requests
from requests.exceptions import RequestException
from unidecode import unidecode
import yake

# ------------------------------------------------------------
# Config LLM (Ollama + mistral:7b-instruct)
# ------------------------------------------------------------
DEFAULT_MODEL = "mistral:7b-instruct"
OLLAMA_HOST  = "http://127.0.0.1:11434"

# ------------------------------------------------------------
# Inteligências (lista canônica que a LLM enxergará por índice)
# ------------------------------------------------------------
INTELLIGENCES = [
    "Logical-Mathematical",
    "Spatial",
    "Linguistic",
    "Musical",
    "Bodily-Kinesthetic",
    "Interpersonal",
    "Intrapersonal",
    "Naturalistic",
]

# ------------------------------------------------------------
# TRIGGERS genéricos (fallback heurístico)
# ------------------------------------------------------------
TRIGGERS = {
    "Logical-Mathematical": {
        "pt": [
            "cálculo",
            "equação",
            "modelo matemático",
            "raciocínio lógico",
            "prova",
            "demonstração",
            "análise quantitativa",
            "inferência lógica",
            "estrutura formal",
            "regra formal"
        ],
        "en": [
            "calculation",
            "equation",
            "mathematical model",
            "logical reasoning",
            "proof",
            "demonstration",
            "quantitative analysis",
            "logical inference",
            "formal structure",
            "formal rule"
        ],
    },
    "Spatial": {
        "pt": ["figura","gráfico","mapa","espaço","forma","trajetória","reta","ângulo",
               "orientação","diagrama","visual"],
        "en": ["figure","graph","map","space","shape","trajectory","line","angle","orientation","diagram","visual"],
    },
    "Linguistic": {
        "pt": [
            "leitura", "ler", "interpretação de texto", "compreensão leitora",
            "escrita", "escrever", "redação", "produção textual",
            "vocabulário", "palavra", "sinônimo", "antônimo",
            "gramática", "ortografia", "pontuação",
            "poema", "rima", "figura de linguagem",
            "argumentação", "tese", "contra-argumento", "debate regrado",
            "resumo", "paráfrase"
        ],
        "en": [
            "reading", "text interpretation", "reading comprehension",
            "writing", "essay", "text production",
            "vocabulary", "word", "synonym", "antonym",
            "grammar", "spelling", "punctuation",
            "poem", "rhyme", "figure of speech",
            "argumentation", "thesis", "counterargument", "structured debate",
            "summary", "paraphrase"
        ],
    },
    "Musical": {
        "pt": ["música","melodia","ritmo","harmonia","batida","cantar"],
        "en": ["music","melody","rhythm","harmony","beat","sing"],
    },
    "Bodily-Kinesthetic": {
        "pt": ["corpo","movimento","gesto","tocar","construir","manipular","atividade física","equilíbrio"],
        "en": ["body","movement","gesture","touch","build","manipulate","physical activity","balance"],
    },
    "Interpersonal": {
        "pt": ["dupla","grupo","turma","colaboração","debate","discussão","parceria","interação"],
        "en": ["pair","group","class","collaboration","debate","discussion","partner","interaction"],
    },
    "Intrapersonal": {
        "pt": ["reflexão","autoavaliação","autoconhecimento","sentimento","emoção","metacogni"],
        "en": ["reflection","self-assessment","self-knowledge","feeling","emotion","metacogn"],
    },
    "Naturalistic": {
        "pt": ["natureza","ambiente","ecossistema","classificação biológica","espécie","campo"],
        "en": ["nature","environment","ecosystem","biological classification","species","fieldwork"],
    },
}

# --- Linguistic: evidência explícita (forte) ---
RE_LINGUISTIC_EXPLICIT = re.compile(
    r"\b("
    # PT-BR (objetivo linguístico explícito)
    r"leitura|ler|"
    r"interpreta(ç|c)(a|ã)o\s+de\s+texto|"
    r"compreens(ã|a)o\s+leitora|"
    r"escrita|escrever|"
    r"red(a|á)(ç|c)(a|ã)o|"
    r"produ(ç|c)(a|ã)o\s+textual|"
    r"produ(ç|c)(a|ã)o\s+de\s+texto|"
    r"vocabul(a|á)rio|"
    r"sin(o|ô)nimo(s)?|ant(o|ô)nimo(s)?|"
    r"gram(a|á)tica|ortografi(a|á)|pontua(ç|c)(a|ã)o|"
    r"poema|rima|met(a|á)fora|figura\s+de\s+linguagem|"
    r"argumenta(ç|c)(a|ã)o|tese|contra-argumento|debate\s+regrado|"
    r"resumo|par(a|á)frase"
    r"|"
    # EN (explicit language-learning objective)
    r"reading|read|text\s+interpretation|reading\s+comprehension|"
    r"writing|write|essay|text\s+production|"
    r"vocabulary|synonym(s)?|antonym(s)?|"
    r"grammar|spelling|punctuation|"
    r"poem|rhyme|metaphor|figure\s+of\s+speech|"
    r"argumentation|thesis|counterargument|structured\s+debate|"
    r"summary|paraphrase"
    r")\b",
    re.IGNORECASE
)

# Coisas que parecem "linguagem" mas normalmente são meio/instrução, não objetivo.
RE_LINGUISTIC_WEAK_ONLY = re.compile(
    r"\b("
    r"verbaliz|falar|dizer|explicar|comentar|responder|relatar|"
    r"conceito(s)?|no(ç|c)(a|ã)o|habilidade(s)?|regra(s)?|"
    r"observ(ar|e)|orienta(ç|c)(a|ã)o|utiliza(ç|c)(a|ã)o"
    r"|"
    r"say|tell|speak|explain|comment|answer|report|describe|"
    r"concept(s)?|notion(s)?|skill(s)?|rule(s)?|"
    r"observe|guidance|instruction(s)?|use|using|utilization"
    r")\b",
    re.IGNORECASE
)


STOP_KW_PT = {
    "habilidade", "habilidades", "nocao", "noção", "conceito", "conceitos",
    "preparacao", "preparação", "utilizacao", "utilização",
    "alunos", "aluno", "turma", "atividade", "atividades",
    "deverao", "deverão", "devem", "objetivo", "objetivos",
    "oportunidade", "oportunidades", "posteriores",
    "exist(e|em)", "mercado", "etc"
}

def is_generic_keyword(term: str) -> bool:
    t = _norm(term)
    if not t:
        return True
    if t in {_norm(x) for x in STOP_KW_PT}:
        return True
    # muito curto ou só “palavra solta” típica
    if len(t) <= 3:
        return True
    return False


def linguistic_is_allowed(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if RE_LINGUISTIC_EXPLICIT.search(t):
        return True
    # se só tem "verbalizar/conceito/habilidade/regras" → NÃO
    if RE_LINGUISTIC_WEAK_ONLY.search(t):
        return False
    return False


# ------------------------------------------------------------
# Helpers numéricos (c(e), r(e))
# ------------------------------------------------------------
def _clip01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

def _yake_confidence(score: float) -> float:
    """
    Converte score do YAKE (menor=melhor) para confiança em [0,1].
    Monótono: score -> maior => confiança menor.
    """
    try:
        s = float(score)
    except Exception:
        return 0.5
    # 1/(1+s) costuma ser estável para scores do YAKE
    return _clip01(1.0 / (1.0 + max(0.0, s)))

def _pos_relevance(paragraph: str, term: str, floor: float = 0.50) -> float:
    """
    Aproxima r(e) pela posição do termo no parágrafo:
      - ocorrências mais cedo => r(e) maior
      - se não encontrado => r(e) default (0.75)
    """
    p = (paragraph or "")
    t = (term or "")
    if not p.strip() or not t.strip():
        return 0.75
    i = p.lower().find(t.lower())
    if i < 0:
        return 0.75
    pos = i / max(1, len(p))
    r = 1.0 - 0.50 * pos   # cai até 0.5 ao final
    return _clip01(max(floor, r))

def _assign_confidence(source: str) -> float:
    """
    Confiança do passo de atribuição MI:
      - llm: decisão explícita
      - trigger: heurística por TRIGGERS
      - forced: fallback forte (ensure_nonempty_intelligences)
    """
    source = (source or "").lower().strip()
    if source == "llm":
        return 0.90
    if source == "trigger":
        return 0.70
    if source == "forced":
        return 0.50
    return 0.60

# ------------------------------------------------------------
# Fábrica de anotador
# ------------------------------------------------------------
def annotate_with_llm_default(**overrides):
    """
    Fábrica: devolve annotate_fn(doc) -> Dict[int, List[Dict[str, Any]]]
    Defaults ajustados para YAKE (menor=melhor):
      - min_score é tratado como LIMIAR MÁXIMO (score <= min_score)
      - default 0.35 tende a produzir evidências suficientes para S3/S4
    """
    defaults = dict(use_n_1_3=True, min_score=0.35, use_heuristic_fallback=True, use_llm=True)
    cfg = {**defaults, **overrides}

    def _runner(doc: dict) -> Dict[int, List[Dict[str, Any]]]:
        return annotate_doc_keywords_with_llm(
            doc,
            use_n_1_3=cfg["use_n_1_3"],
            min_score=cfg["min_score"],
            use_heuristic_fallback=cfg["use_heuristic_fallback"],
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
    """
    Lê JSON no formato S1 ({doc_id, paragraphs:[{k, text, lang?}, ...]})
    e retorna dicionário {k -> [evidências]}.
    """
    doc = _read_doc_from_json(in_path)
    return annotate_fn(doc)

def _ollama_up(host: str) -> bool:
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        return r.ok
    except Exception:
        return False

# ------------------------------------------------------------
# Utilidades de normalização / matching (heurística)
# ------------------------------------------------------------
def _norm(s: str) -> str:
    return unidecode((s or "").strip().lower())

def _matches_any(token_norm: str, triggers: List[str]) -> bool:
    for t in triggers:
        t = _norm(t)
        if not t:
            continue
        if re.search(rf"(^|\W){re.escape(t)}($|\W)", token_norm):
            return True
        if t in token_norm and len(t) >= 5:
            return True
    return False

def _candidate_idxs_from_triggers(keyword: str) -> list[int]:
    cands = []
    k = _norm(keyword)
    for i, name in enumerate(INTELLIGENCES):
        if name == "Linguistic" and not linguistic_is_allowed(keyword):
            continue
        langs = TRIGGERS.get(name, {})
        trigs = [_norm(x) for x in langs.get("pt", [])] + [_norm(x) for x in langs.get("en", [])]
        if _matches_any(k, trigs):
            cands.append(i)
    return cands

def map_keyword_to_intelligences_heuristic(keyword: str) -> List[str]:
    k = _norm(keyword)
    hits: List[str] = []
    for intel in INTELLIGENCES:
        if intel == "Linguistic" and not linguistic_is_allowed(keyword):
            continue
        langs = TRIGGERS.get(intel, {})
        trigs = [_norm(x) for x in langs.get("pt", [])] + [_norm(x) for x in langs.get("en", [])]
        if _matches_any(k, trigs):
            hits.append(intel)
    return hits

# ---------- Intelligence helpers & validators ----------
def ensure_nonempty_intelligences(text: str) -> list[str]:
    t = _norm(text)

    if re.search(r"compar|orden|classific|quant|tamanh|menor|maior", t):
        return ["Logical-Mathematical"]

    if re.search(r"encaix|forma|fileir|dispos|visual", t):
        return ["Spatial"]

    if re.search(r"manipul|moviment|coordena|cortar", t):
        return ["Bodily-Kinesthetic"]

    # Intrapersonal SOMENTE com evidência explícita
    if re.search(r"sentiment|emoc|autoavalia|reflexao pessoal|consci[eê]ncia de si", t):
        return ["Intrapersonal"]

    return []

def idxs_to_labels_safe(idxs: list[int]) -> list[str]:
    labs = []
    for x in idxs or []:
        try:
            xi = int(x)
            if 0 <= xi < len(INTELLIGENCES):
                labs.append(INTELLIGENCES[xi])
        except Exception:
            continue
    return labs

def normalize_items_with_intelligence(items: list[dict], lang: str = "pt") -> list[dict]:
    """
    "Explode" itens para 1 registro por (role,text,intelligence),
    garantindo que TODOS carreguem:
      - intelligence (obrigatório)
      - confidence = c(e) em [0,1]
      - relevance  = r(e) em [0,1]

    Observação: confidence/relevance ficam no payload (JSON). S3 não persiste isso em RDF.
    """
    out_map: dict[tuple, dict] = {}  # key -> item

    for it in items or []:
        role = it.get("role")
        text = (it.get("text") or "").strip()
        if not role or not text:
            continue

        base_conf = _clip01(it.get("confidence", 0.75))
        base_rel  = _clip01(it.get("relevance", 0.75))

        # 1) preferir mi_idx -> labels
        labels = idxs_to_labels_safe(it.get("mi_idx"))
        assign_src = "llm" if labels else ""

        # --- hard gate para Linguistic (por texto do item) ---
        if labels:
            labels = [lab for lab in labels if (lab != "Linguistic" or linguistic_is_allowed(text))]

        # 2) se veio 'intelligence' direto, incorpora
        if "intelligence" in it and isinstance(it["intelligence"], str):
            if it["intelligence"] in INTELLIGENCES and it["intelligence"] not in labels:
                if it["intelligence"] != "Linguistic" or linguistic_is_allowed(text):
                    labels.append(it["intelligence"])
                    if not assign_src:
                        assign_src = "llm"

        # 3) se ainda vazio, heurística obrigatória
        if not labels:
            labels = ensure_nonempty_intelligences(text)
            assign_src = "forced" if labels else ""

        if not labels:
            continue

        assign_conf = _assign_confidence(assign_src or "trigger")
        conf = _clip01(base_conf * assign_conf)

        for lab in labels[:2]:
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
                # merge: manter o mais confiável/relevante
                rec["confidence"] = max(float(rec.get("confidence", 0.0)), conf)
                rec["relevance"]  = max(float(rec.get("relevance", 0.0)), base_rel)

    return list(out_map.values())

# ------------------------------------------------------------
# LLM: discourse + context
# ------------------------------------------------------------
def call_llm_discourse_and_context(
    paragraph: str,
    lang_hint: str = "pt",
    use_trigger_candidates: bool = True,
    allow_expand_beyond_candidates: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retorna itens crus:
      {"role":"DiscursiveStrategy","text":"..."}
      {"role":"ContextObject","text":"...","mi_idx":[0]}
    """
    cands_note = ""
    if use_trigger_candidates:
        cands_note = (
            "- Para cada ContextObject, priorize inteligências sugeridas pela semântica do próprio texto do objeto "
            "(por exemplo, 'gráfico', 'equação' → Logical-Mathematical; 'mapa', 'figura' → Spatial). "
            "Se não houver evidência, deixe mi_idx vazio.\n"
        )

    sys_msg = (
        "Você é um anotador para um sistema educacional baseado na Teoria das Inteligências Múltiplas. "
        "Extraia do parágrafo estratégias discursivas e objetos de contexto. Responda APENAS com JSON válido."
    )

    contract = {
        "items": [
            {"role": "DiscursiveStrategy", "text": "exemplo de estratégia"},
            {"role": "ContextObject", "text": "exemplo de objeto", "mi_idx": [0]}
        ]
    }

    rules = f"""REGRAS COGNITIVAS IMPORTANTES:
- NÃO utilize Intrapersonal a menos que o texto trate explicitamente
  de sentimentos, autorregulação, consciência de si ou reflexão pessoal.
- Atividades de comparação, ordenação, agrupamento e quantificação
  NÃO são intrapessoais.
- NÃO confunda meio de instrução com objetivo cognitivo.
- Linguistic só deve ser atribuída quando a linguagem for o OBJETIVO
  do aprendizado (ex.: leitura, escrita, argumentação, interpretação textual).
- Se o texto descreve comparação, ordenação, classificação, seriação,
  noções de quantidade, relação ou estrutura, priorize Logical-Mathematical,
  mesmo que tudo esteja descrito verbalmente.
- Se o texto descreve organização visual, proporção, encaixe, forma,
  disposição espacial ou percepção visual concreta, priorize Spatial.
- Se o texto descreve manipulação física, movimento ou coordenação,
  use Bodily-Kinesthetic apenas como apoio, a menos que a habilidade
  motora seja o foco explícito da atividade.
- Sempre considere: qual inteligência está sendo DESENVOLVIDA,
  e não apenas UTILIZADA.
{cands_note}
"""

    user_msg = f"""
IDIOMA DO PARÁGRAFO: {lang_hint}

PARÁGRAFO:
{paragraph}

OPÇÕES DE INTELIGÊNCIAS (use os índices):
{json.dumps([{"idx": i, "label": lab} for i, lab in enumerate(INTELLIGENCES)], ensure_ascii=False, indent=2)}

CONTRACT (responda SOMENTE neste JSON):
{json.dumps(contract, ensure_ascii=False, indent=2)}

{rules}
"""

    if not _ollama_up(OLLAMA_HOST):
        return []

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": DEFAULT_MODEL,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": user_msg}
                ],
                "options": {"temperature": 0.0}
            },
            timeout=180
        )
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content", "").strip()
    except RequestException:
        return []

    # Parse robusto
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        cleaned = _sanitize_json_guess(content)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            rec_items = _recover_items_from_partial_json(content)
            if not rec_items:
                print("[WARN] LLM retornou JSON inválido em discourse/context. Conteúdo bruto (até 2k chars):", file=sys.stderr)
                print(content[:2000], file=sys.stderr)
                print(f"[WARN] Motivo do parse: {e}", file=sys.stderr)
                return []
            parsed = {"items": rec_items}

    items = parsed.get("items", []) if isinstance(parsed, dict) else []
    out: List[Dict[str, Any]] = []
    for it in items:
        role = str(it.get("role", "")).strip()
        text = str(it.get("text", "")).strip()
        if role not in ("DiscursiveStrategy", "ContextObject") or not text:
            continue

        mi_idx = it.get("mi_idx", [])
        fixed_idx: List[int] = []
        if isinstance(mi_idx, list):
            for x in mi_idx:
                try:
                    xi = int(x)
                except Exception:
                    continue
                if 0 <= xi < len(INTELLIGENCES):
                    fixed_idx.append(xi)

        if use_trigger_candidates and not allow_expand_beyond_candidates and role == "ContextObject":
            cand_names = map_keyword_to_intelligences_heuristic(text)
            cand_idx = {INTELLIGENCES.index(mi) for mi in cand_names} if cand_names else set()
            fixed_idx = [xi for xi in fixed_idx if xi in cand_idx]

        out.append({"role": role, "text": text, "mi_idx": fixed_idx})
    return out

# ------------------------------------------------------------
# Extrator de keywords (YAKE)
# ------------------------------------------------------------
def extract_keywords(text: str, n: int = 1, lang: str = "pt") -> List[tuple]:
    kw_extractor = yake.KeywordExtractor(lan=lang, n=n)
    return kw_extractor.extract_keywords(text)

def extract_keywords_1_3(text: str, lang: str = "pt") -> List[tuple]:
    """
    Retorna lista de (term, score, n) com o MELHOR (menor) score encontrado para o termo.
    """
    bag: Dict[str, tuple[float,int]] = {}  # term -> (best_score, best_n)
    for n in (1, 2, 3):
        for term, score in extract_keywords(text, n=n, lang=lang):
            t = term.strip()
            if not t:
                continue
            prev = bag.get(t)
            if prev is None or score < prev[0]:
                bag[t] = (float(score), int(n))
    return sorted([(t, s, n) for t, (s, n) in bag.items()], key=lambda x: x[1])

# ------------------------------------------------------------
# Helpers: subir Ollama e garantir modelo
# ------------------------------------------------------------
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
            {"role": "user",   "content": '{"ping": true}'}
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

def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

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

def start_ollama_daemon_if_needed(host: str, poll_sec: float = 1.0) -> None:
    if _ollama_up(host):
        return

    if not _which("ollama"):
        print("[WARN] Ollama não está acessível via PATH; não vou iniciar daemon automaticamente.", file=sys.stderr)
        return

    print("[INFO] Iniciando 'ollama serve'…", flush=True)
    popen_kwargs = {}
    if platform.system() == "Windows":
        popen_kwargs.update({"creationflags": 0x08000000})  # CREATE_NO_WINDOW
    else:
        popen_kwargs.update({"start_new_session": True})

    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **popen_kwargs)
    except Exception as e:
        raise SystemExit(f"[ERRO] Não foi possível iniciar 'ollama serve': {e}")

    while not _ollama_up(host):
        time.sleep(poll_sec)

# ------------------------------------------------------------
# Sanitização de JSON vindo da LLM (caso ela escape)
# ------------------------------------------------------------
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
                return s[start:i+1]
    return None

def _recover_items_from_partial_json(text: str) -> list[dict]:
    s = text
    m = re.search(r'\"items\"\s*:\s*\[', s)
    if not m:
        return []
    i = m.end()

    items = []
    n = len(s)
    while i < n:
        while i < n and s[i] in " \r\n\t,":
            i += 1
        if i >= n or s[i] == ']':
            break
        if s[i] != '{':
            j = i + 1
            while j < n and s[j] not in '{]':
                j += 1
            if j < n and s[j] == '{':
                i = j
            else:
                break

        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        chunk = s[i:j+1]
                        try:
                            obj = json.loads(chunk)
                            items.append(obj)
                        except Exception:
                            pass
                        i = j + 1
                        break
                elif ch == ']':
                    i = j
                    break
            j += 1
        else:
            break
    return items

def _sanitize_json_guess(s: str) -> str:
    if "```" in s:
        s = s.replace("```json", "```").strip()
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
        else:
            s = parts[0]
    s = s.strip()
    guess = _extract_balanced_json(s) or s
    guess = re.sub(r",\s*([}\]])", r"\1", guess)
    return guess

# ------------------------------------------------------------
# LLM: atribuição de inteligências por LISTA DE KEYWORDS
# ------------------------------------------------------------
def call_llm_assign_for_keywords(
    keywords: list[str],
    use_trigger_candidates: bool = True,
    allow_expand_beyond_candidates: bool = False,
    lang_hint: str = "pt"
) -> dict[str, list[int]]:
    opts = [{"idx": i, "label": lab} for i, lab in enumerate(INTELLIGENCES)]

    kw2cands: dict[str, list[int]] = {}
    if use_trigger_candidates:
        for kw in keywords:
            kw2cands[kw] = _candidate_idxs_from_triggers(kw)

    sys_msg = (
        "Você é um anotador de palavras-chave segundo as inteligências de Gardner. "
        "Responda APENAS com JSON válido conforme o contrato."
    )
    contract = {"items": [{"text": "keyword_aqui", "mi_idx": [0, 2]}]}

    if use_trigger_candidates and not allow_expand_beyond_candidates:
        rule_line = "- Para cada keyword, `mi_idx` deve ser SUBCONJUNTO EXATO de CANDIDATES[keyword]. SE CANDIDATES estiver vazio, ESCOLHA a melhor MI e retorne um array com 1 índice."
    elif use_trigger_candidates and allow_expand_beyond_candidates:
        rule_line = "- Priorize CANDIDATES[keyword]; você PODE incluir outra MI apenas se houver forte evidência. SEMPRE retorne pelo menos 1 índice."
    else:
        rule_line = "- SEMPRE retorne pelo menos 1 índice para cada keyword."

    user_msg = f"""
IDIOMA DOS TERMOS: {lang_hint}

KEYWORDS:
{json.dumps(keywords, ensure_ascii=False, indent=2)}

OPTIONS (use os índices):
{json.dumps(opts, ensure_ascii=False, indent=2)}

CANDIDATES (por keyword; índices permitidos derivados de TRIGGERS):
{json.dumps(kw2cands, ensure_ascii=False, indent=2)}

CONTRACT (responda SOMENTE neste JSON):
{json.dumps(contract, ensure_ascii=False, indent=2)}

REGRAS:
{rule_line}
- O campo "text" deve ser EXATAMENTE igual à keyword fornecida (mesma grafia e espaçamento).
- Limite "mi_idx" a no MÁXIMO 2 índices por keyword (escolha as melhores).
- Não repita keywords no retorno; 1 entrada por keyword.
- Não inclua comentários nem markdown.
"""

    if not _ollama_up(OLLAMA_HOST):
        return {kw: [] for kw in keywords}

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": DEFAULT_MODEL,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": user_msg}
                ],
                "options": {"temperature": 0.0}
            },
            timeout=180
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
        except json.JSONDecodeError as e:
            rec_items = _recover_items_from_partial_json(content)
            if not rec_items:
                print("[WARN] LLM retornou JSON inválido em assign_for_keywords. Conteúdo bruto (até 2k chars):", file=sys.stderr)
                print(content[:2000], file=sys.stderr)
                print(f"[WARN] Motivo do parse: {e}", file=sys.stderr)
                return {kw: [] for kw in keywords}
            parsed = {"items": rec_items}

    items = parsed.get("items", []) if isinstance(parsed, dict) else []

    out: dict[str, list[int]] = {kw: [] for kw in keywords}
    tmp: dict[str, list[int]] = {}

    for it in items:
        text = str(it.get("text", "")).strip()
        idxs = it.get("mi_idx", [])
        if not text:
            continue
        labs = idxs_to_labels_safe(idxs)
        if not labs:
            labs = ensure_nonempty_intelligences(text)
        fixed_idxs = [INTELLIGENCES.index(l) for l in labs if l in INTELLIGENCES]
        if use_trigger_candidates and not allow_expand_beyond_candidates:
            cset = set(_candidate_idxs_from_triggers(text))
            fixed_idxs = [i for i in fixed_idxs if i in cset] or fixed_idxs[:1]
        tmp[text] = fixed_idxs[:2]

    for kw in keywords:
        if kw in tmp and tmp[kw]:
            out[kw] = tmp[kw]
        else:
            labs = ensure_nonempty_intelligences(kw)
            out[kw] = [INTELLIGENCES.index(l) for l in labs if l in INTELLIGENCES][:2]

    return out

# ------------------------------------------------------------
# Pipeline por PARÁGRAFO
# ------------------------------------------------------------
def annotate_one_paragraph_keywords_with_llm(
    text: str,
    lang: str = "pt",
    use_n_1_3: bool = True,
    min_score: float = 0.35,          # YAKE: mantém termos com score <= min_score
    use_heuristic_fallback: bool = True,
    use_llm: bool = True
) -> List[Dict[str, Any]]:
    """
    Retorna lista de evidências com:
      role, text, lang, intelligence, confidence=c(e), relevance=r(e)
    """
    llm_available = use_llm and _ollama_up(OLLAMA_HOST)

    items: List[Dict[str, Any]] = []
    seen = set()  # (role, text.lower(), intelligence)

    # ------------ 1) KEYWORDS (YAKE) ------------
    kws_scored = extract_keywords_1_3(text, lang=lang) if use_n_1_3 else [
        (t, float(s), 1) for (t, s) in extract_keywords(text, n=1, lang=lang)
    ]

    # YAKE: menor = melhor => manter score <= limiar
    keywords_meta = [(t, s, n) for (t, s, n) in kws_scored if float(s) <= float(min_score)]
    keywords = [t for (t, _, _) in keywords_meta]

    llm_map: dict[str, list[int]] = {}
    if keywords and llm_available:
        llm_map = call_llm_assign_for_keywords(
            keywords,
            use_trigger_candidates=True,
            allow_expand_beyond_candidates=True,
            lang_hint=lang
        )
    else:
        llm_map = {kw: [] for kw in keywords}

    for kw, kw_score, kw_n in keywords_meta:
        idxs = llm_map.get(kw, []) or []
        assign_src = "llm" if idxs else ""

        if not idxs and use_heuristic_fallback:
            mis = map_keyword_to_intelligences_heuristic(kw)
            idxs = [INTELLIGENCES.index(mi) for mi in mis] if mis else []
            if idxs:
                assign_src = "trigger"

        if not idxs:
            # rede de segurança: pelo menos 1 MI
            labs = ensure_nonempty_intelligences(kw)[:2]
            assign_src = "forced" if labs else "forced"
            idxs = [INTELLIGENCES.index(l) for l in labs if l in INTELLIGENCES][:2]

        # confidence/relevance (por evidência)
        c_extract = _yake_confidence(kw_score)
        c_assign  = _assign_confidence(assign_src or "trigger")
        conf = _clip01(c_extract * c_assign)
        rel  = _pos_relevance(text, kw)

        for idx_mi in idxs[:2]:
            if 0 <= idx_mi < len(INTELLIGENCES):
                entry = {
                    "role": "Keyword",
                    "text": kw,
                    "lang": lang,
                    "intelligence": INTELLIGENCES[idx_mi],
                    "confidence": conf,
                    "relevance": rel,
                }
                key = (entry["role"], entry["text"].lower(), entry["intelligence"])
                if key not in seen:
                    items.append(entry); seen.add(key)

    # ------------ 2) DISCURSIVE STRATEGY / CONTEXT OBJECT (LLM) ------------
    dc_raw = call_llm_discourse_and_context(
        paragraph=text,
        lang_hint=lang,
        use_trigger_candidates=True,
        allow_expand_beyond_candidates=False
    ) if llm_available else []

    # injeta c(e), r(e) base no item cru (antes de expandir)
    dc_raw_aug = []
    for it in dc_raw:
        role = (it.get("role") or "").strip()
        t = (it.get("text") or "").strip()
        if role not in ("DiscursiveStrategy", "ContextObject") or not t:
            continue
        dc_raw_aug.append({
            "role": role,
            "text": t,
            "lang": lang,
            "mi_idx": it.get("mi_idx"),
            "confidence": 0.85,               # confiança do extrator (LLM)
            "relevance": _pos_relevance(text, t),
        })

    dc_items_norm = normalize_items_with_intelligence(dc_raw_aug, lang=lang)
    for entry in dc_items_norm:
        key = (entry["role"], entry["text"].lower(), entry["intelligence"])
        if key not in seen:
            items.append(entry); seen.add(key)

    # ------------ 3) Safety net final ------------
    # Garante que todos tenham intelligence/confidence/relevance e remove duplicatas residuais.
    items = normalize_items_with_intelligence(items, lang=lang)

    # ------------ 3.5) Linguistic hard-gate (por parágrafo) ------------
    if not linguistic_is_allowed(text):
        items = [it for it in items if it.get("intelligence") != "Linguistic"]

    return items

def annotate_doc_keywords_with_llm(
    doc: Dict[str, Any],
    use_n_1_3: bool = True,
    min_score: float = 0.35,          # YAKE: score <= min_score
    use_heuristic_fallback: bool = True,
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
            lang=lang if lang in ("pt", "en") else "pt",
            use_n_1_3=use_n_1_3,
            min_score=min_score,
            use_heuristic_fallback=use_heuristic_fallback,
            use_llm=use_llm
        )
        out[int(k)] = items

    return out

def annotate_chunks_keywords_with_llm(
    chunks: List[str],
    lang: str = "pt",
    use_n_1_3: bool = True,
    min_score: float = 0.35,
    use_heuristic_fallback: bool = True,
    use_llm: bool = True
) -> Dict[int, List[Dict[str, Any]]]:
    result: Dict[int, List[Dict[str, Any]]] = {}
    llm_available = use_llm and _ollama_up(OLLAMA_HOST)

    for i, chunk in enumerate(chunks):
        kws_scored = extract_keywords_1_3(chunk, lang=lang) if use_n_1_3 else [
            (t, float(s), 1) for (t, s) in extract_keywords(chunk, n=1, lang=lang)
        ]
        keywords_meta = [(t, s, n) for (t, s, n) in kws_scored if float(s) <= float(min_score)]
        keywords = [t for (t, _, _) in keywords_meta]

        if not keywords:
            result[i] = []
            continue

        llm_map = call_llm_assign_for_keywords(
            keywords,
            use_trigger_candidates=True,
            allow_expand_beyond_candidates=True,
            lang_hint=lang
        ) if llm_available else {kw: [] for kw in keywords}

        items: List[Dict[str, Any]] = []
        seen = set()

        for kw, kw_score, _kw_n in keywords_meta:
            idxs = llm_map.get(kw, []) or []
            assign_src = "llm" if idxs else ""

            if not idxs and use_heuristic_fallback:
                mis = map_keyword_to_intelligences_heuristic(kw)
                idxs = [INTELLIGENCES.index(mi) for mi in mis] if mis else []
                if idxs:
                    assign_src = "trigger"

            if not idxs:
                labs = ensure_nonempty_intelligences(kw)[:2]
                idxs = [INTELLIGENCES.index(l) for l in labs if l in INTELLIGENCES][:2]
                assign_src = "forced"

            c_extract = _yake_confidence(kw_score)
            c_assign = _assign_confidence(assign_src or "trigger")
            conf = _clip01(c_extract * c_assign)
            rel = _pos_relevance(chunk, kw)

            for idx_mi in idxs[:2]:
                if 0 <= idx_mi < len(INTELLIGENCES):
                    entry = {
                        "role": "Keyword",
                        "text": kw,
                        "lang": lang,
                        "intelligence": INTELLIGENCES[idx_mi],
                        "confidence": conf,
                        "relevance": rel,
                    }
                    key = (entry["text"].lower(), entry["intelligence"])
                    if key not in seen:
                        items.append(entry); seen.add(key)
        normed = normalize_items_with_intelligence(items, lang=lang)
        if not linguistic_is_allowed(chunk):
            normed = [it for it in normed if it.get("intelligence") != "Linguistic"]
        result[i] = normed

    return result

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import io

    ap = argparse.ArgumentParser(
        description="Annotate evidences -> Multiple Intelligences por parágrafo (LLM + TRIGGERS fallback + YAKE)."
    )
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Caminho do JSON de entrada (doc com {doc_id, paragraphs[]})")
    ap.add_argument("--out", dest="out_path", default="-",
                    help="Caminho do JSON de saída (ou '-' para stdout).")
    ap.add_argument("--offline", action="store_true",
                    help="Força modo offline (não usa LLM; apenas TRIGGERS).")
    ap.add_argument("--n13", action="store_true",
                    help="Extrair keywords com n=1..3 (YAKE). Se omitido, usa n=1.")
    ap.add_argument("--min-score", type=float, default=0.35,
                    help="YAKE score máximo permitido (menor = melhor). Mantém termos com score <= este valor.")
    args = ap.parse_args()

    # 1) Ler JSON de entrada
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
        use_n_1_3=args.n13,
        min_score=args.min_score,
        use_heuristic_fallback=True,
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
