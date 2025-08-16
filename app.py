import os
import asyncio
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from contextlib import asynccontextmanager

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader
import requests
from bs4 import BeautifulSoup


# Events for the workflow
class URLProvided(Event):
    url: str


class ContentFetched(Event):
    content: str
    url: str


class SummaryGenerated(Event):
    summary: str
    url: str


# Website Summarization Workflow
class WebsiteSummarizationWorkflow(Workflow):
    def __init__(self, llm: Optional[Any] = None):
        super().__init__()
        self.llm = llm or OpenAI(model="gpt-4o-mini")

    @step
    async def fetch_website_content(self, ev: StartEvent) -> ContentFetched:
        """Fetch the content of the website."""
        url = ev.data.get("url")
        if not url:
            raise ValueError("URL is required")
        
        try:
            # Use SimpleWebPageReader for better content extraction
            reader = SimpleWebPageReader(html_to_text=True)
            documents = reader.load_data([url])
            
            if documents:
                content = documents[0].text
            else:
                # Fallback to direct web scraping
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract text content
                content = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
            
            return ContentFetched(content=content, url=url)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch content from {url}: {str(e)}")

    @step
    async def generate_summary(self, ev: ContentFetched) -> StopEvent:
        """Generate a summary of the website content."""
        try:
            # Limit content length to avoid token limits
            max_content_length = 8000  # Adjust based on your model's context window
            content = ev.content[:max_content_length] if len(ev.content) > max_content_length else ev.content
            
            prompt = f"""
            Please provide a comprehensive yet concise summary of the following website content.
            Focus on the main topics, key points, and important information.
            
            Website URL: {ev.url}
            
            Content:
            {content}
            
            Summary:
            """
            
            response = await self.llm.acomplete(prompt)
            summary = str(response)
            
            return StopEvent(result={"summary": summary, "url": ev.url})
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


# Global variables
query_engine = None
summarization_workflow = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global query_engine, summarization_workflow
    
    # Initialize the original document query functionality
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
    
    # Initialize the website summarization workflow
    summarization_workflow = WebsiteSummarizationWorkflow()
    
    yield
    
    # Cleanup if needed
    pass


# Initialize FastAPI app
app = FastAPI(
    title="LlamaIndex Query API with Website Summarization",
    description="Query documents and summarize website content using LlamaIndex",
    version="2.0.0",
    lifespan=lifespan
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str
    query: str


class URLRequest(BaseModel):
    url: HttpUrl


class SummaryResponse(BaseModel):
    summary: str
    url: str


class HealthResponse(BaseModel):
    status: str
    message: str


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="LlamaIndex Query API with Website Summarization is running"
    )


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the LlamaIndex Query API with Website Summarization",
        "description": "Ask questions about documents and get website summaries",
        "endpoints": {
            "POST /query": "Submit a query about the documents",
            "POST /summarize": "Get a summary of a website",
            "GET /health": "Health check endpoint",
            "GET /docs": "API documentation"
        }
    }


# Original query endpoint
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


# New website summarization endpoint
@app.post("/summarize", response_model=SummaryResponse)
async def summarize_website(request: URLRequest):
    """
    Summarize the content of a website using LlamaIndex Workflow.
    
    Args:
        request: URLRequest containing the website URL to summarize
        
    Returns:
        SummaryResponse containing the summary and original URL
    """
    try:
        url_str = str(request.url)
        
        # Run the workflow
        result = await summarization_workflow.run(url=url_str)
        
        return SummaryResponse(
            summary=result["summary"],
            url=result["url"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while summarizing the website: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
