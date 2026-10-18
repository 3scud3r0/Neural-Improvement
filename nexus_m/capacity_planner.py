"""Declarative hardware planning for distributed AST search (NO cloud calls)."""
from dataclasses import dataclass,asdict
import math

@dataclass(frozen=True)
class CapacityPlan:
    agents:int
    neurons_each:int
    tree_nodes:int
    simulations:int
    worker_count:int
    total_neurons:int
    activation_megabytes_fp32:float
    estimated_tree_megabytes:float
    simulations_per_worker:int
    cloud_provisioned:bool
    approval_required:bool
    def as_dict(self):return asdict(self)

def plan(agents:int,neurons_each:int,tree_nodes:int,
         simulations:int,workers:int=4)->CapacityPlan:
    if not (1<=agents<=1000 and 1<=neurons_each<=10000 and
            1<=tree_nodes<=10_000_000 and 1<=simulations<=1_000_000 and
            1<=workers<=64):
        raise ValueError('capacity request exceeds bounded planning envelope')
    total=agents*neurons_each
    return CapacityPlan(agents,neurons_each,tree_nodes,simulations,workers,
        total,round(total*4/1e6,3),round(tree_nodes*160/1e6,3),
        math.ceil(simulations/workers),False,True)
