"""Corpus model: timestamped facts, change injection, and the document set over (logical) time.

The benchmark owns the ground truth and the clock. A `Scenario` is a set of `Fact`s (each with
an old value, a new value, and the tick it changes) plus distractor documents. `documents_at(eff)`
returns what a fully-fresh index would hold as of effective time `eff` — the refresh policy (in
runner.py) decides which `eff` a system actually sees at each tick.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    text: str
    doc_id: str = ""


@dataclass
class Fact:
    entity: str
    attribute: str
    old: str
    new: str
    change_tick: int

    @property
    def id(self) -> str:
        return f"{self.entity}:{self.attribute}"

    def query(self) -> str:
        return f"Who is the current {self.attribute} of {self.entity}?"

    def statement(self, value: str) -> str:
        return f"The {self.attribute} of {self.entity} is {value}."

    def truth(self, tick: int) -> str:
        return self.new if tick >= self.change_tick else self.old


@dataclass
class Scenario:
    facts: list[Fact]
    distractors: list[Document] = field(default_factory=list)
    model: str = "append"  # "append" (old doc kept) or "replace" (old removed)

    def documents_at(self, eff_tick: int) -> list[Document]:
        """Corpus as of effective time `eff_tick` (what a fully-fresh index would hold)."""
        docs = list(self.distractors)
        for f in self.facts:
            changed = f.change_tick <= eff_tick
            if self.model == "replace":
                v = f.new if changed else f.old
                docs.append(Document(f.statement(v), f.id))
            else:  # append: old stays; new added once changed -> realistic old+new conflict
                docs.append(Document(f.statement(f.old), f"{f.id}:old"))
                if changed:
                    docs.append(Document(f.statement(f.new), f"{f.id}:new"))
        return docs


# --- scenario generator: made-up entities (no parametric knowledge), order-neutral values ---
_FIRST = ["Maria", "David", "Sara", "John", "Ahmed", "Lily", "Lena", "Carl", "Yuki", "Zoe",
          "Omar", "Ivan", "Nina", "Eve", "Paul", "Greg", "Rita", "Hana", "Sven", "Liu",
          "Tara", "Mark", "Uma", "Noor", "Ken", "Ana"]
_LAST = ["Lopez", "Park", "Chen", "Diaz", "Hassan", "Wong", "Berg", "Mata", "Tanaka", "Ray",
         "Said", "Petr", "Roy", "Lin", "Adams", "Ono", "Gomez", "Kim", "Holm", "Wei",
         "Singh", "Bauer", "Devi", "Ali", "Sato", "Cruz"]
_ATTRS = [("director", "Org"), ("CEO", "Corp"), ("lead", "Project"),
          ("manager", "Team"), ("chair", "Board")]


def make_scenario(n_facts: int = 24, seed: int = 0, n_distractors: int = 5,
                  change_spread: int = 5, change_offset: int = 3, model: str = "append") -> Scenario:
    """Generate a reproducible scenario. Values are random distinct names with no inherent
    ordering (so the current value cannot be guessed without the document), and entities are
    made-up so answers must come from the retrieved context, not the model's prior knowledge."""
    rng = random.Random(seed)
    used: set[str] = set()

    def name() -> str:
        while True:
            nm = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
            if nm not in used:
                used.add(nm)
                return nm

    facts = []
    for i in range(n_facts):
        attr, pre = _ATTRS[i % len(_ATTRS)]
        facts.append(Fact(f"{pre}{i}", attr, name(), name(), change_offset + (i % change_spread)))
    distractors = [Document(f"The director of FillerOrg{j} is {name()}.", f"distractor:{j}")
                   for j in range(n_distractors)]
    return Scenario(facts, distractors, model=model)
