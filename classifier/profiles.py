
"""
PT-BR: Perfis sintéticos de referência (PoC) no mesmo espaço vetorial do MIProfile.
EN: Synthetic reference profiles (PoC) in the same MIProfile vector space.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence

# Ordem CANÔNICA do vetor (v_MI) usada pelo classificador e pelos perfis.
# Ajuste aqui se você decidir mudar o contrato.
MI_ORDER: List[str] = [
    "Linguistic",             # 0
    "Logical-Mathematical",   # 1
    "Spatial",                # 2
    "Musical",                # 3
    "Bodily-Kinesthetic",     # 4
    "Interpersonal",          # 5
    "Intrapersonal",          # 6
    "Naturalist",             # 7
]

@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    vector: List[float]

def _assert_vec(v: Sequence[float]) -> List[float]:
    v = [float(x) for x in v]
    if len(v) != 8:
        raise ValueError(f"Perfil inválido: esperado 8 dimensões, veio {len(v)}")
    if any(x < 0 for x in v):
        raise ValueError("Perfil inválido: valores negativos não são permitidos")
    s = sum(v)
    if s <= 0:
        raise ValueError("Perfil inválido: soma deve ser > 0")
    # normaliza L1 por padrão para facilitar leitura/comparação
    return [x / s for x in v]

# Perfis (com base no seu Quadro de perfis)
PROFILES: Dict[str, Profile] = {
    "P1-LING": Profile("P1-LING", "Predominante Linguística", _assert_vec([0.70,0.05,0.05,0.05,0.05,0.04,0.04,0.02])),
    "P2-LOG":  Profile("P2-LOG",  "Predominante Lógico-matemática", _assert_vec([0.05,0.70,0.05,0.05,0.05,0.04,0.04,0.02])),
    "P3-ESP":  Profile("P3-ESP",  "Predominante Espacial", _assert_vec([0.05,0.10,0.60,0.10,0.05,0.04,0.04,0.02])),
    "P4-CORP": Profile("P4-CORP", "Predominante Corporal-cinestésica", _assert_vec([0.05,0.05,0.10,0.05,0.60,0.08,0.05,0.02])),
    "P5-MUS":  Profile("P5-MUS",  "Predominante Musical", _assert_vec([0.05,0.05,0.05,0.60,0.05,0.08,0.08,0.04])),
    "P6-INTER":Profile("P6-INTER","Predominante Interpessoal", _assert_vec([0.10,0.05,0.05,0.05,0.05,0.50,0.15,0.05])),
    "P7-INTRA":Profile("P7-INTRA","Predominante Intrapessoal", _assert_vec([0.10,0.05,0.05,0.05,0.05,0.15,0.50,0.05])),
    "P8-NAT":  Profile("P8-NAT",  "Predominante Naturalista", _assert_vec([0.05,0.10,0.05,0.05,0.05,0.05,0.05,0.60])),
    "P9-HUM":  Profile("P9-HUM",  "Equilibrado – humanidades", _assert_vec([0.22,0.08,0.08,0.08,0.06,0.22,0.22,0.04])),
    "P10-TEC": Profile("P10-TEC", "Equilibrado – ciência e tecnologia", _assert_vec([0.08,0.24,0.24,0.06,0.06,0.08,0.08,0.16])),
    "P11-GLOB":Profile("P11-GLOB","Equilibrado – global", _assert_vec([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125])),
}

def list_profiles() -> List[str]:
    return list(PROFILES.keys())

def get_profiles(keys: List[str] | None) -> List[Profile]:
    if not keys or keys == ["all"]:
        return [PROFILES[k] for k in list_profiles()]
    out: List[Profile] = []
    for k in keys:
        kk = k.strip()
        if not kk:
            continue
        if kk not in PROFILES:
            raise KeyError(f"Perfil desconhecido: {kk}. Disponíveis: {', '.join(list_profiles())}")
        out.append(PROFILES[kk])
    return out
