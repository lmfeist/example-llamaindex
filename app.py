import os.path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# Initialize FastAPI app
app = FastAPI(
    title="LlamaIndex Query API",
    description="Query the 'Gift of the Magi' story using LlamaIndex",
    version="1.0.0"
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    response: str
    query: str

class HealthResponse(BaseModel):
    status: str
    message: str

# Initialize index on startup
@app.on_event("startup")
async def startup_event():
    global query_engine
    
    # check if storage already exists
    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    
    query_engine = index.as_query_engine()

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="LlamaIndex Query API is running"
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the LlamaIndex Query API",
        "description": "Ask questions about 'The Gift of the Magi' story",
        "endpoints": {
            "POST /query": "Submit a query about the story",
            "GET /health": "Health check endpoint",
            "GET /docs": "API documentation"
        }
    }

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_story(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response = query_engine.query(request.query)
        return QueryResponse(
            response=str(response),
            query=request.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
