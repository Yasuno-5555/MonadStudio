from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class Node(BaseModel):
    id: str
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    pos: Optional[List[float]] = None

class Edge(BaseModel):
    from_: str = Field(alias="from")
    to: str
    port_out: str
    port_in: str

    class Config:
        populate_by_name = True

class DAG(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class Shock(BaseModel):
    target: str
    size: float
    persistence: float = 0.8

class SimulationConfig(BaseModel):
    shock: Shock
    horizon: int = 200

class Meta(BaseModel):
    version: str = "1.0"
    engine: str = "monad_core_v1"

class Scenario(BaseModel):
    meta: Meta
    dag: DAG
    simulation: SimulationConfig
