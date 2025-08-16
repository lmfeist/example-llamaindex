import os
import traceback
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from contextlib import asynccontextmanager


from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
# Removed unused imports - FunctionTool and FunctionCallingAgent not needed for this workflow
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader
import requests
from bs4 import BeautifulSoup


# Events for the workflow
class ContentFetched(Event):
    content: str
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
            # Re-raise the original exception to let the custom exception handler show the full stack trace
            raise e

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
            # Re-raise the original exception to let the custom exception handler show the full stack trace
            raise e


# Global variables
summarization_workflow = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global summarization_workflow
    
    # Initialize the website summarization workflow
    summarization_workflow = WebsiteSummarizationWorkflow()
    
    yield
    
    # Cleanup if needed
    pass


# Initialize FastAPI app
app = FastAPI(
    title="Website Summarization API",
    description="Summarize website content using LlamaIndex",
    version="2.0.0",
    lifespan=lifespan,
    debug=True  # Enable debug mode for detailed error responses
)


# Custom exception handler to show stack traces
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle all unhandled exceptions and return detailed error information including stack trace.
    """
    error_detail = {
        "error": str(exc),
        "type": type(exc).__name__,
        "traceback": traceback.format_exc(),
        "path": str(request.url),
        "method": request.method
    }
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error with full stack trace",
            "error_info": error_detail
        }
    )


# Enhanced HTTP exception handler to show more details
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with additional context.
    """
    error_detail = {
        "status_code": exc.status_code,
        "detail": exc.detail,
        "path": str(request.url),
        "method": request.method
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_detail
    )


# Request/Response models
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
        message="Website Summarization API is running"
    )


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Website Summarization API",
        "description": "Get AI-powered summaries of website content",
        "endpoints": {
            "POST /summarize": "Get a summary of a website",
            "OPTIONS /summarize": "Get allowed methods for summarize endpoint",
            "GET /health": "Health check endpoint",
            "GET /docs": "API documentation"
        }
    }



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
        # Re-raise the original exception to let the custom exception handler show the full stack trace
        raise e


@app.options("/summarize")
async def summarize_options():
    """
    Handle OPTIONS request for the summarize endpoint.
    Returns allowed methods and CORS headers.
    """
    response = Response()
    response.headers["Allow"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Origin"] = "*"
    
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
