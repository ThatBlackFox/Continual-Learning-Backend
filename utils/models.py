from pydantic import BaseModel

class Smiles(BaseModel):
    text: str
    dataset: str