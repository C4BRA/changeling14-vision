"""Text side of stage 1: match sprites to their entity's name and description.

A small pretrained sentence model (MiniLM by default, frozen) turns every
entity's "name. description" into a vector once, before training. Training
then pulls each sprite's embedding towards its own entity's text and away
from the other texts in the batch (SigLIP's pairwise sigmoid loss), through a
learned linear map from the text space into the encoder's 256-wide space.

The table is cached as text.npz in the model folder:
  texts     unique "name. description" strings
  vectors   (N, D) float16, unit length
  entities  entity id per row of entity_text (several entities can share a text)
  entity_text  index into texts for each entity
  model     the sentence model's name
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def entity_text(entity: dict) -> str | None:
    name = (entity.get("name") or "").strip()
    if not name:
        return None
    desc = (entity.get("description") or "").strip()
    return f"{name}. {desc}" if desc else name


def load_entity_texts(dataset: Path) -> dict[str, str]:
    texts = {}
    path = dataset / "catalogue" / "entities.jsonl"
    if not path.is_file():
        return texts
    for line in path.open(encoding="utf-8"):
        e = json.loads(line)
        if e.get("abstract"):
            continue
        t = entity_text(e)
        if t:
            texts[e["id"]] = t
    return texts


def sentence_embedder(model_name: str = DEFAULT_TEXT_MODEL, device: str | None = None):
    """A function list[str] -> (N, D) unit vectors, from sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def embed(texts: list[str], batch: int = 256) -> np.ndarray:
        return model.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False)

    return embed


def build_text_table(dataset: Path, out: Path, embed=None, model_name: str = DEFAULT_TEXT_MODEL,
                     progress=None) -> Path:
    """Embed every named entity's text once and save text.npz in the model folder."""
    path = out / "text.npz"
    if path.is_file():
        return path
    by_entity = load_entity_texts(dataset)
    texts = sorted(set(by_entity.values()))
    if not texts:
        raise ValueError(f"no entity names in {dataset}/catalogue/entities.jsonl")
    embed = embed or sentence_embedder(model_name)
    vectors = []
    chunk = 512
    for i in range(0, len(texts), chunk):
        if progress:
            progress.update("embed text", i, len(texts), f"{model_name}: entity names and descriptions")
        vectors.append(np.asarray(embed(texts[i:i + chunk]), dtype=np.float32))
    vec = np.concatenate(vectors)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
    row = {t: i for i, t in enumerate(texts)}
    entities = sorted(by_entity)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "text.tmp.npz"
    np.savez(tmp, texts=np.array(texts), vectors=vec.astype(np.float16), entities=np.array(entities),
             entity_text=np.array([row[by_entity[e]] for e in entities], dtype=np.int32),
             model=np.array(model_name))
    tmp.replace(path)
    return path


class TextTable:
    def __init__(self, path: Path):
        data = np.load(path)
        self.texts = [str(t) for t in data["texts"]]
        self.vectors = data["vectors"].astype(np.float32)
        self.model = str(data["model"])
        self.row_of_entity = {str(e): int(r) for e, r in zip(data["entities"], data["entity_text"])}

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def rows_for(self, entity_ids: list[str]) -> list[int]:
        return sorted({self.row_of_entity[e] for e in entity_ids if e in self.row_of_entity})
