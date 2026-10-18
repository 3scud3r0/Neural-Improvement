"""A second module for coordinated algebraic changes and dependency analysis."""
from .arithmetic import redundant_identity


def redundant_double(x: int) -> int:
    return (((x+x)*1)+0)*1+0


def pipeline(x):
    return redundant_double(redundant_identity(x))
