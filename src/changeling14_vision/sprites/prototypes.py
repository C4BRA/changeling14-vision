"""Read entity prototypes and work out which sprites each entity draws.

Inheritance follows SS14's rules closely enough for sprite lookup: a child
starts from its parents (in listed order), components merge by type, and a
child's component fields override the parent's field by field. Lists such as
Sprite layers are replaced whole. `abstract` is never inherited.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class PrototypeLoader(_Loader):  # type: ignore[misc, valid-type]
    """Safe loader that reads SS14's custom tags (!type:Foo) as plain data.

    Every scalar stays a string, as SS14's own YAML reader sees it: a state
    named `on` or `0` must not turn into True or 0.
    """


PrototypeLoader.yaml_implicit_resolvers = {}


def truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


VARIANTS = "__variants__"


def _construct_tagged(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        data = loader.construct_mapping(node, deep=True)
        if suffix == "type:CreateVariants":
            # One prototype entry that stands for several: field i of variant i.
            return {VARIANTS: list(data.get("values") or data.get("sequences") or [])}
        return data
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


PrototypeLoader.add_multi_constructor("!", _construct_tagged)


@dataclass
class Entity:
    id: str
    parents: tuple[str, ...]
    abstract: bool
    source: str
    raw: dict
    # Filled in by resolve():
    name: str | None = None
    description: str | None = None
    suffix: str | None = None
    chain: tuple[str, ...] = ()
    components: dict[str, dict] = field(default_factory=dict)


def load_entities(prototypes_dir: Path) -> tuple[dict[str, Entity], list[str]]:
    entities: dict[str, Entity] = {}
    problems: list[str] = []
    for path in sorted(prototypes_dir.rglob("*.yml")):
        rel = path.relative_to(prototypes_dir).as_posix()
        try:
            with open(path, encoding="utf-8-sig") as f:
                docs = yaml.load(f, Loader=PrototypeLoader)
        except yaml.YAMLError as exc:
            problems.append(f"{rel}: YAML error: {str(exc).splitlines()[0]}")
            continue
        if not isinstance(docs, list):
            continue
        for doc in (variant for d in docs for variant in expand_variants(d)):
            if not isinstance(doc, dict) or doc.get("type") != "entity" or "id" not in doc:
                continue
            parents = doc.get("parent") or ()
            if isinstance(parents, str):
                parents = (parents,)
            entity_id = str(doc["id"])
            if entity_id in entities:
                problems.append(f"{rel}: duplicate entity id {entity_id}")
            entities[entity_id] = Entity(entity_id, tuple(str(p) for p in parents),
                                         truthy(doc.get("abstract")), rel, doc)
    return entities, problems


def _variant_count(node: object) -> int:
    if isinstance(node, dict):
        if VARIANTS in node:
            return len(node[VARIANTS])
        return max((_variant_count(v) for v in node.values()), default=0)
    if isinstance(node, list):
        return max((_variant_count(v) for v in node), default=0)
    return 0


def _pick_variant(node: object, i: int) -> object:
    if isinstance(node, dict):
        if VARIANTS in node:
            values = node[VARIANTS]
            return values[i] if i < len(values) else None
        return {k: _pick_variant(v, i) for k, v in node.items()}
    if isinstance(node, list):
        return [_pick_variant(v, i) for v in node]
    return node


def expand_variants(doc: object) -> list:
    """Expand a prototype using !type:CreateVariants (anywhere in it) into one per variant."""
    count = _variant_count(doc)
    if not count:
        return [doc]
    return [_pick_variant(doc, i) for i in range(count)]


def _merge_component(base: dict, child: dict) -> dict:
    merged = dict(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_component(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve(entities: dict[str, Entity]) -> list[str]:
    """Fill name, description, chain and merged components on every entity."""
    problems: list[str] = []
    done: set[str] = set()

    def visit(entity: Entity, stack: tuple[str, ...]) -> None:
        if entity.id in done:
            return
        if entity.id in stack:
            problems.append(f"inheritance loop: {' -> '.join(stack + (entity.id,))}")
            return
        name = description = suffix = None
        components: dict[str, dict] = {}
        chain: list[str] = []
        for parent_id in entity.parents:
            parent = entities.get(parent_id)
            if parent is None:
                problems.append(f"{entity.id}: unknown parent {parent_id}")
                continue
            visit(parent, stack + (entity.id,))
            name = parent.name if parent.name is not None else name
            description = parent.description if parent.description is not None else description
            suffix = parent.suffix if parent.suffix is not None else suffix
            for ctype, data in parent.components.items():
                components[ctype] = _merge_component(components.get(ctype, {}), data)
            for ancestor in (parent.id,) + parent.chain:
                if ancestor not in chain:
                    chain.append(ancestor)
        raw = entity.raw
        entity.name = raw.get("name", name)
        entity.description = raw.get("description", description)
        entity.suffix = raw.get("suffix", suffix)
        for comp in raw.get("components") or []:
            if isinstance(comp, dict) and "type" in comp:
                data = {k: v for k, v in comp.items() if k != "type"}
                components[comp["type"]] = _merge_component(components.get(comp["type"], {}), data)
        entity.components = components
        entity.chain = tuple(chain)
        done.add(entity.id)

    for entity in entities.values():
        visit(entity, ())
    return problems


@dataclass(frozen=True)
class SpriteRef:
    role: str            # sprite, layer, icon, item, clothing
    rsi: str | None      # relative to Textures/, e.g. "Objects/Tools/wrench.rsi"
    state: str | None    # None when the state is chosen at runtime or the role uses the whole sheet
    layer: int | None = None
    map: tuple[str, ...] = ()
    visible: bool = True
    texture: str | None = None
    color: str | None = None         # the layer's tint, e.g. "#3f3f3f" or "red"
    sprite_color: str | None = None  # the whole Sprite's tint, applied to every layer


def _rsi_path(value: object) -> str | None:
    if not isinstance(value, str) or value.strip() in ("", "null", "~"):
        return None
    value = value.strip().lstrip("/")
    if value.startswith("Textures/"):
        value = value[len("Textures/"):]
    return value


def _color(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def sprite_refs(entity: Entity) -> list[SpriteRef]:
    refs: list[SpriteRef] = []
    sprite = entity.components.get("Sprite")
    base_rsi = None
    if sprite is not None:
        base_rsi = _rsi_path(sprite.get("sprite") or sprite.get("rsi"))
        sprite_color = _color(sprite.get("color"))
        layers = sprite.get("layers")
        if isinstance(layers, list) and layers:
            for i, layer in enumerate(layers):
                if not isinstance(layer, dict):
                    continue
                rsi = _rsi_path(layer.get("sprite") or layer.get("rsi")) or base_rsi
                mapping = layer.get("map") or ()
                if isinstance(mapping, str):
                    mapping = (mapping,)
                refs.append(SpriteRef(
                    "layer", rsi, layer.get("state"), i, tuple(str(m) for m in mapping),
                    str(layer.get("visible", "true")).lower() != "false", _rsi_path(layer.get("texture")),
                    _color(layer.get("color")), sprite_color))
        elif base_rsi or sprite.get("state"):
            # A top-level state is only drawn when there are no layers; with layers,
            # an inherited `state: icon` is ignored by the game too.
            refs.append(SpriteRef("sprite", base_rsi, sprite.get("state"), sprite_color=sprite_color))
    icon = entity.components.get("Icon")
    if icon is not None:
        refs.append(SpriteRef("icon", _rsi_path(icon.get("sprite")), icon.get("state")))
    for role, ctype in (("item", "Item"), ("clothing", "Clothing")):
        comp = entity.components.get(ctype)
        if comp is None:
            continue
        # In-hand and worn sprites fall back to the Sprite's own sheet, as in the game.
        rsi = _rsi_path(comp.get("sprite")) or base_rsi
        if rsi:
            refs.append(SpriteRef(role, rsi, None))
    return refs
