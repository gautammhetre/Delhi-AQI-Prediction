"""
Retrieval for the assistant (the "R" in RAG).

Knowledge base: short Markdown notes in knowledge/, each summarising one official
source (CPCB, CAQM, WHO, the Health Ministry) with its URL in the front matter.

Chunking: one chunk per "##" section. The notes are written so each section answers
one question, which keeps chunks small (40-120 words) and on one topic. Each chunk
carries its document title and source URL, so every answer can cite where it came from.

Retrieval: TF-IDF over word unigrams and bigrams, cosine similarity, top-k.
Why not neural embeddings by default? The corpus is ~40 chunks, TF-IDF needs no
model download, runs in milliseconds and fits a free hosting tier (sentence-transformers
pulls in PyTorch, which is ~700 MB). Set RAG_BACKEND=embeddings to use
all-MiniLM-L6-v2 instead if sentence-transformers is installed; the interface is the same.

Refusal: if the best score is below MIN_SCORE the retriever returns nothing, and the
assistant says it does not know rather than letting the LLM guess. The threshold was
chosen with the evaluation set in eval/rag_questions.json (see evaluate()).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
EVAL_PATH = ROOT / "eval" / "rag_questions.json"

MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.14"))

# A few everyday words mapped to the vocabulary the official documents use.
# TF-IDF only matches exact words, so "kids" would otherwise never find "children".
SYNONYMS = {
    "kids": "children", "kid": "children", "child": "children", "son": "children", "daughter": "children",
    "run": "running jogging exercise", "running": "jogging exercise", "jog": "jogging exercise", "gym": "exercise",
    "walk": "walking exercise", "cycling": "exercise", "play": "play outdoor",
    "mask": "masks n95", "masks": "n95", "respirator": "n95 masks",
    "window": "windows ventilation", "windows": "ventilation",
    "pregnant": "pregnant women vulnerable", "elderly": "older adults vulnerable", "old": "older adults",
    "asthma": "lung disease asthma", "asthmatic": "asthma lung disease",
    "firecrackers": "firecrackers diwali burning", "crackers": "firecrackers diwali",
    "stubble": "stubble crop-residue burning", "construction": "construction demolition",
    "truck": "trucks", "trucks": "truck entry", "diesel": "diesel vehicles", "school": "schools online classes",
    "schools": "online classes", "wfh": "work from home", "doctor": "symptoms doctor",
    "cough": "symptoms cough", "breathless": "breathlessness symptoms",
}


@dataclass
class Chunk:
    id: str
    doc: str
    title: str
    heading: str
    text: str
    source: str
    url: str

    def cite(self) -> dict:
        return {"id": self.id, "title": self.title, "section": self.heading, "source": self.source, "url": self.url}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


def _parse(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = m.group(2)
    chunks = []
    for section in re.split(r"^## ", body, flags=re.M)[1:]:
        heading, _, text = section.partition("\n")
        text = " ".join(text.split())
        if text:
            chunks.append(Chunk(id=f"{path.stem}#{_slug(heading)}", doc=path.stem, title=meta.get("title", path.stem),
                                heading=heading.strip(), text=text, source=meta.get("source", ""), url=meta.get("url", "")))
    return chunks


def load_chunks(directory: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    return [c for p in sorted(directory.glob("*.md")) for c in _parse(p)]


_ROMAN = {"1": "i", "one": "i", "2": "ii", "two": "ii", "3": "iii", "three": "iii", "4": "iv", "four": "iv"}


def expand_query(q: str) -> str:
    # "stage 3" -> "stage iii", matching how GRAP stages are written in the documents.
    q = re.sub(r"\bstage\s*(1|2|3|4|one|two|three|four)\b", lambda m: f"stage {_ROMAN[m.group(1).lower()]}", q, flags=re.I)
    words = re.findall(r"[a-z0-9.+]+", q.lower())
    extra = [SYNONYMS[w] for w in words if w in SYNONYMS]
    return " ".join([q] + extra)


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Extra stop words found through the evaluation set: everyday verbs that matched headings by accident
# ("suggest a movie to watch" matched "Symptoms to watch"). They carry no topic on their own.
# "very" is taken OUT of the standard list so that "very poor" is not treated the same as "poor".
_STOP = (ENGLISH_STOP_WORDS - {"not", "no", "very"}) | {"watch", "suggest", "tell", "tonight", "please"}


def _stem(t: str) -> str:
    # Tiny suffix stripper so "generators"/"generator" and "jogging"/"jog" match.
    for suf, min_len in (("ing", 6), ("ies", 5), ("ed", 5), ("es", 5), ("s", 4)):
        if len(t) >= min_len and t.endswith(suf) and not t.endswith("ss"):
            return t[: -len(suf)] + ("y" if suf == "ies" else "")
    return t


def _analyze(text: str) -> list[str]:
    """Lower-case, keep tokens like "pm2.5" and "n95" intact, drop stop words, stem, add bigrams."""
    toks = [_stem(t) for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in _STOP and len(t) > 1]
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vec = TfidfVectorizer(analyzer=_analyze, sublinear_tf=True)
        # Heading is included twice: it says what the section is about.
        self.matrix = self.vec.fit_transform([f"{c.heading} {c.heading} {c.title} {c.text}" for c in chunks])

    def scores(self, query: str):
        return cosine_similarity(self.vec.transform([expand_query(query)]), self.matrix)[0]


class EmbeddingRetriever:  # optional, needs `pip install sentence-transformers`
    name = "embeddings"

    def __init__(self, chunks: list[Chunk], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported lazily on purpose
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.matrix = self.model.encode([f"{c.heading}. {c.text}" for c in chunks], normalize_embeddings=True)

    def scores(self, query: str):
        q = self.model.encode([query], normalize_embeddings=True)
        return (self.matrix @ q.T).ravel()


@lru_cache(maxsize=1)
def get_retriever():
    chunks = load_chunks()
    if os.environ.get("RAG_BACKEND", "tfidf").lower() == "embeddings":
        return EmbeddingRetriever(chunks)
    return TfidfRetriever(chunks)


# Topic gate: a question must mention at least one air-quality or health word before retrieval runs.
# Similarity alone isn't enough for short off-topic questions: "a good movie to watch" scores well
# against the "Good AQI" section just because of the word "good".
TOPIC_WORDS = re.compile(
    r"\b(aqi|air|pollut\w*|smog|haze|dust|smoke|pm ?2\.?5|pm ?10|particul\w*|ozone|o3|no2|so2|nh3|co\b|"
    r"carbon|nitrogen|sulphur|sulfur|ammonia|grap|graded|stage|cpcb|caqm|who\b|guideline\w*|breakpoint\w*|"
    r"mask\w*|n95|n99|respirator\w*|purifier\w*|cleaner\w*|hepa|window\w*|ventilat\w*|indoor\w*|outdoor\w*|"
    r"outside|jog\w*|run\w*|walk\w*|exercis\w*|play\w*|sport\w*|school\w*|construction|diesel|truck\w*|"
    r"vehicle\w*|firecracker\w*|cracker\w*|diwali|stubble|burn\w*|symptom\w*|cough\w*|breath\w*|asthma\w*|"
    r"lung\w*|heart|doctor\w*|health\w*|safe|risk\w*|vulnerable|children|kids?|elderly|pregnan\w*|"
    r"delhi|ncr|month\w*|hour\w*|time of day|season\w*|winter|monsoon|forecast\w*|model\w*|accura\w*|predict\w*|"
    r"measure\w*|precaution\w*|restriction\w*)\b", re.I)


def on_topic(query: str) -> bool:
    return bool(TOPIC_WORDS.search(query or ""))


def retrieve(query: str, k: int = 3, min_score: float | None = None, retriever=None) -> list[dict]:
    """Top-k chunks above the threshold, best first. An empty list means "not in the knowledge base"."""
    if not on_topic(query):
        return []
    r = retriever or get_retriever()
    threshold = MIN_SCORE if min_score is None else min_score
    scores = r.scores(query)
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [{**r.chunks[i].cite(), "text": r.chunks[i].text, "score": round(float(scores[i]), 3)}
            for i in ranked if scores[i] >= threshold]


def evaluate(path: Path = EVAL_PATH, k: int = 3, min_score: float | None = None, retriever=None) -> dict:
    """Retrieval evaluation.
    hit@k     share of answerable questions whose expected section is in the top k
    refusal   share of out-of-scope questions for which nothing passes the threshold
    """
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    answerable = [c for c in cases if c.get("expected")]
    unanswerable = [c for c in cases if not c.get("expected")]
    hits, misses, wrong_answers = 0, [], []
    for c in answerable:
        got = [h["id"] for h in retrieve(c["question"], k, min_score, retriever)]
        if any(e in got for e in c["expected"]):
            hits += 1
        else:
            misses.append({"question": c["question"], "expected": c["expected"], "got": got})
    refused = 0
    for c in unanswerable:
        got = retrieve(c["question"], k, min_score, retriever)
        if not got:
            refused += 1
        else:
            wrong_answers.append({"question": c["question"], "got": [h["id"] for h in got]})
    return {
        "answerable": len(answerable),
        f"hit_at_{k}": round(hits / len(answerable), 3) if answerable else None,
        "out_of_scope": len(unanswerable),
        "refusal_rate": round(refused / len(unanswerable), 3) if unanswerable else None,
        "threshold": MIN_SCORE if min_score is None else min_score,
        "misses": misses,
        "answered_out_of_scope": wrong_answers,
    }
