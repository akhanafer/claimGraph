from typing import List

from pydantic import BaseModel


class ClaimEdge(BaseModel):
    id: str
    text: str


class ResourceNode(BaseModel):
    id: str
    url: str
    title: str
    source_node: bool
    claims: List[ClaimEdge]
