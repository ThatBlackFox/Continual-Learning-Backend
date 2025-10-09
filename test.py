from fastapi import FastAPI
from pydantic import BaseModel

# Create FastAPI app instance
app = FastAPI()

# Define a request body model
class Item(BaseModel):
    name: str
    description: str

# POST endpoint that accepts an Item
@app.post("/items/")
async def create_item(item: Item):
    return {"message": f"Item {item.name} created!", "data": item}

# Run the server using `uvicorn server:app --reload`