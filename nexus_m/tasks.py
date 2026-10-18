"""Synthetic family-disjoint algorithmic tasks; no real-world data needed."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, random
from .grammar import execute

@dataclass(frozen=True)
class Task:
    family: str
    examples: tuple[tuple[int,int], ...]
    heldout: tuple[tuple[int,int], ...]
    oracle: tuple[str, ...] | None = None

    @property
    def fingerprint(self) -> str:
        data=json.dumps({"family":self.family,"examples":self.examples},sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:24]

    def public(self) -> "Task":
        """Drop hidden labels and oracle before passing to a solver."""
        return Task("unlabeled", self.examples, (), None)


def generate_task(rng:random.Random, family:str, n_examples=5, n_test=20) -> Task:
    a=rng.choice((-3,-2,-1,1,2,3))
    b=rng.choice((-3,-2,-1,0,1,2,3))
    c=rng.choice((-3,-2,-1,1,2,3))
    x=['X']; ca=[f'C{a}']; cb=[f'C{b}']; cc=[f'C{c}']
    templates={
       'linear': x+ca+['MUL']+cb+['ADD'],
       'shift': x+ca+['ADD'],
       'square': x+['SQUARE']+cb+['ADD'],
       'absolute': x+['ABS']+cb+['ADD'],
       'parity': x+['MOD2']+ca+['ADD'],
       'negate': x+['NEG']+cb+['ADD'],
       'halved': x+['HALF']+cb+['ADD'],
       'shift_absolute': x+ca+['ADD','ABS']+cb+['ADD'],
       'quadratic': x+['SQUARE']+x+ca+['MUL','ADD']+cb+['ADD'],
       'parity_scaled': x+['MOD2']+ca+['MUL']+cb+['ADD'],
       'product_shift': x+ca+['ADD']+x+cc+['ADD','MUL'],
       'abs_square': x+['SQUARE']+ca+['SUB','ABS'],
    }
    if family not in templates: raise ValueError(f'Unknown family {family}')
    program=tuple(templates[family]+['HALT'])
    # The example and evaluation domains differ to measure extrapolation.
    candidates=list(range(-9,10))
    rng.shuffle(candidates)
    xs=candidates[:min(n_examples,len(candidates))]
    test_x=list(range(-35,-10))+list(range(11,36))
    rng.shuffle(test_x)
    test_x=test_x[:n_test]
    return Task(family,tuple((v,execute(program,v)) for v in xs),
                tuple((v,execute(program,v)) for v in test_x), program)

TRAIN_FAMILIES=('shift','linear','square','absolute','parity','negate','halved')
OOD_FAMILIES=('shift_absolute','quadratic','parity_scaled','product_shift','abs_square')


def sample_task(rng:random.Random, split='train', n_examples=5, n_test=20)->Task:
    families=TRAIN_FAMILIES if split=='train' else OOD_FAMILIES
    return generate_task(rng,rng.choice(families),n_examples,n_test)
