from evidences_api import annotate_doc_keywords_with_llm

def annotate_with_llm_default(**overrides):
    """
    Retorna um callable(doc) -> Dict[int, List[dict]] com defaults,
    permitindo override por kwargs.
    """
    defaults = dict(
        use_n_1_3=True,
        min_score=0.0,
        use_heuristic_fallback=True,
        use_llm=True,
    )
    cfg = {**defaults, **overrides}

    def _runner(doc):
        return annotate_doc_keywords_with_llm(
            doc,
            use_n_1_3=cfg["use_n_1_3"],
            min_score=cfg["min_score"],
            use_heuristic_fallback=cfg["use_heuristic_fallback"],
            use_llm=cfg["use_llm"],
        )
    return _runner
