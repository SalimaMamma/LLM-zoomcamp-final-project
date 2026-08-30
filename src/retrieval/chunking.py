"""
Découpage des abstracts en chunks par section logique (contexte, méthode,
résultats, conclusion) plutôt qu'un découpage naïf par taille fixe.

Les abstracts scientifiques suivent souvent une structure implicite ou
explicite (IMRaD). On utilise des mots-clés heuristiques pour repérer les
frontières de section ; à défaut de structure détectée, l'abstract entier
est conservé comme un seul chunk "abstract".
"""
import re

SECTION_MARKERS = {
    "background": r"\b(background|introduction|purpose|objective)s?\b",
    "method": r"\b(method|methods|design|participants|subjects)s?\b",
    "result": r"\b(result|results|findings)s?\b",
    "conclusion": r"\b(conclusion|conclusions|discussion|implications)s?\b",
}


def split_abstract_into_sections(abstract: str) -> list[tuple[str, str]]:
    """Retourne une liste de (section, texte). Fallback: [("abstract", texte)]."""
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
    if len(sentences) < 3:
        return [("abstract", abstract.strip())]

    tagged: list[tuple[str, str]] = []
    current_section = "background"
    buffer = []

    for sentence in sentences:
        lowered = sentence.lower()
        matched_section = None
        for section, pattern in SECTION_MARKERS.items():
            if re.search(pattern, lowered):
                matched_section = section
                break

        if matched_section and matched_section != current_section and buffer:
            tagged.append((current_section, " ".join(buffer)))
            buffer = []
            current_section = matched_section

        buffer.append(sentence)

    if buffer:
        tagged.append((current_section, " ".join(buffer)))

    return tagged


def make_chunk_id(paper_id: str, section: str, idx: int) -> str:
    return f"{paper_id}::{section}::{idx}"


def chunk_paper(paper_id: str, abstract: str) -> list[dict]:
    sections = split_abstract_into_sections(abstract)
    chunks = []
    for idx, (section, text) in enumerate(sections):
        if len(text.strip()) < 20:
            continue
        chunks.append(
            {
                "chunk_id": make_chunk_id(paper_id, section, idx),
                "paper_id": paper_id,
                "section": section,
                "content": text.strip(),
            }
        )
    return chunks
