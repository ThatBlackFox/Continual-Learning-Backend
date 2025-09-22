from pydantic import BaseModel

class Smiles(BaseModel):
    smiles: str