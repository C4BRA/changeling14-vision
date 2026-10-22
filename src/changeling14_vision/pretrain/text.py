"""Text side of stage 1: match sprites to their entity's name and description.

A small pretrained sentence model (MiniLM by default, frozen) turns every
entity's "name. description" into a vector once, before training. Training
then pulls each sprite's embedding towards its own entity's text and away
from the other texts in the batch (SigLIP's pairwise sigmoid loss), through a
learned linear map from the text space into the encoder's 256-wide space.

With rich=True each entity's vector is the average of a few phrases built
only from catalogue facts, so items with no description still say what they
are: "name. description", the name with its editor suffix ("bucket, full"),
the prototype's folders ("bucket, objects, tools"), and what its components
make it ("bucket: an item you can pick up, holds liquid that can spill").
Components become words only through COMPONENT_WORDS below; anything not in
that table is left out rather than guessed.

The table is cached as text.npz in the model folder:
  texts     unique "name. description" strings
  vectors   (N, D) float16, unit length
  entities  entity id per row of entity_text (several entities can share a text)
  entity_text  index into texts for each entity
  model     the sentence model's name
  rich      whether vectors average the extra phrases
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def entity_text(entity: dict) -> str | None:
    name = (entity.get("name") or "").strip()
    if not name:
        return None
    desc = (entity.get("description") or "").strip()
    return f"{name}. {desc}" if desc else name


# Component -> words, in the order phrases list them. Each was checked
# against a sample of the entities that carry it, and components whose
# plain meaning misleads were left out: Clothing (PDAs and cigars have it),
# Edible (moths eat clothes), Explosive (power cells), Seed (anomaly cores),
# Armor, MeleeWeapon, Tool, Lock, PointLight and others. These are the only
# hand-written facts.
COMPONENT_WORDS = {
    "Gun": "can shoot",
    "Projectile": "a projectile",
    "Produce": "harvested from a plant",
    "Storage": "holds items inside",
    "EntityStorage": "opens and closes to hold things inside",
    "Stack": "a stack that can be split",
    "FloorTile": "can be placed as flooring",
    "Circuitboard": "an electronics board",
    "MachineBoard": "a machine board",
    "Door": "a door",
    "Airlock": "an airlock",
    "Wall": "a wall",
    "Machine": "a machine",
    "Computer": "a computer console",
    "VendingMachine": "a vending machine",
    "Strap": "can be buckled to",
    "Organ": "a body organ",
    "Pda": "a PDA",
    "IdCard": "an ID card",
    "Instrument": "can play music",
    "BorgModule": "a cyborg module",
    "Item": "an item you can pick up",
}
MAX_COMPONENT_WORDS = 6


def _words(part: str) -> str:
    """'OuterClothing' -> 'outer clothing', 'base_solution.yml' -> 'base solution'."""
    part = re.sub(r"\.ya?ml$", "", part).replace("_", " ").replace("-", " ")
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", part).lower().strip()


def entity_phrases(entity: dict) -> list[str]:
    """The texts averaged into an entity's vector; the first is the one shown."""
    main = entity_text(entity)
    if not main:
        return []
    name = entity["name"].strip()
    phrases = [main]
    suffix = (entity.get("suffix") or "").strip()
    if suffix:
        phrases.append(f"{name}, {suffix.lower()}")
    folders = [_words(p) for p in (entity.get("source") or "").split("/")[1:-1]]
    folders = [f for f in folders if f]
    if folders:
        phrases.append(f"{name}, {', '.join(folders)}")
    have = set(entity.get("components") or ())
    words = [w for c, w in COMPONENT_WORDS.items() if c in have][:MAX_COMPONENT_WORDS]
    if words:
        phrases.append(f"{name}: {', '.join(words)}")
    return phrases


def load_entity_texts(dataset: Path, rich: bool = False) -> dict[str, list[str]]:
    """entity id -> phrases (just ["name. description"] unless rich)."""
    texts = {}
    path = dataset / "catalogue" / "entities.jsonl"
    if not path.is_file():
        return texts
    for line in path.open(encoding="utf-8"):
        e = json.loads(line)
        if e.get("abstract"):
            continue
        phrases = entity_phrases(e) if rich else [t for t in [entity_text(e)] if t]
        if phrases:
            texts[e["id"]] = phrases
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
                     progress=None, rich: bool = False) -> Path:
    """Embed every named entity's text once and save text.npz in the model folder.

    Entities that share a shown text ("name. description") share a row; with
    rich, that row is the mean over those entities of their phrase averages."""
    path = out / "text.npz"
    if path.is_file():
        return path
    by_entity = load_entity_texts(dataset, rich)
    texts = sorted({p[0] for p in by_entity.values()})
    if not texts:
        raise ValueError(f"no entity names in {dataset}/catalogue/entities.jsonl")
    phrases = sorted({t for p in by_entity.values() for t in p})
    embed = embed or sentence_embedder(model_name)
    vectors = []
    chunk = 512
    for i in range(0, len(phrases), chunk):
        if progress:
            progress.update("embed text", i, len(phrases), f"{model_name}: entity names and descriptions")
        vectors.append(np.asarray(embed(phrases[i:i + chunk]), dtype=np.float32))
    pvec = np.concatenate(vectors)
    pvec /= np.linalg.norm(pvec, axis=1, keepdims=True) + 1e-8
    prow = {t: i for i, t in enumerate(phrases)}
    row = {t: i for i, t in enumerate(texts)}
    vec = np.zeros((len(texts), pvec.shape[1]), dtype=np.float32)
    for p in by_entity.values():
        vec[row[p[0]]] += pvec[[prow[t] for t in p]].mean(0)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
    entities = sorted(by_entity)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "text.tmp.npz"
    np.savez(tmp, texts=np.array(texts), vectors=vec.astype(np.float16), entities=np.array(entities),
             entity_text=np.array([row[by_entity[e][0]] for e in entities], dtype=np.int32),
             model=np.array(model_name), rich=np.array(rich))
    tmp.replace(path)
    return path


class TextTable:
    def __init__(self, path: Path):
        data = np.load(path)
        self.texts = [str(t) for t in data["texts"]]
        self.vectors = data["vectors"].astype(np.float32)
        self.model = str(data["model"])
        self.rich = bool(data["rich"]) if "rich" in data else False
        self.row_of_entity = {str(e): int(r) for e, r in zip(data["entities"], data["entity_text"])}

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def rows_for(self, entity_ids: list[str]) -> list[int]:
        return sorted({self.row_of_entity[e] for e in entity_ids if e in self.row_of_entity})
