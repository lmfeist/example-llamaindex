import os
import traceback
import logging
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
        # Get URL from the event data or kwargs passed to run()
        url = getattr(ev, 'url', None) or (hasattr(ev, 'data') and ev.data.get("url"))
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
            # Log the full exception with stack trace for debugging
            logger.error(f"Failed to fetch content from {url}:\n{traceback.format_exc()}")
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
            # Log the full exception with stack trace for debugging
            logger.error(f"Failed to generate summary:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


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


# Configure logging to show detailed error information
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Override any existing logging configuration
)
logger = logging.getLogger(__name__)

# Also set uvicorn access logger to show more details
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.setLevel(logging.DEBUG)


# Custom exception handler to log stack traces to server console
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle all unhandled exceptions, log full stack trace to server console,
    and return a clean error response to the client.
    """
    # Log the full stack trace to server console
    logger.error(
        f"Unhandled exception in {request.method} {request.url}:\n"
        f"Exception type: {type(exc).__name__}\n"
        f"Exception message: {str(exc)}\n"
        f"Full traceback:\n{traceback.format_exc()}"
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "path": str(request.url),
            "method": request.method
        }
    )


# Enhanced HTTP exception handler to log HTTP exceptions
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with logging for server-side debugging.
    """
    # Log HTTP exceptions for debugging
    logger.warning(
        f"HTTP exception in {request.method} {request.url}:\n"
        f"Status code: {exc.status_code}\n"
        f"Detail: {exc.detail}"
    )
    
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
        # Log the full exception with stack trace for debugging
        logger.error(f"Error in summarize_website endpoint:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while summarizing the website: {str(e)}"
        )


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
    
    # Configure uvicorn to show detailed logs and stack traces
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="debug",  # Enable debug logging
        access_log=True,    # Show access logs
        use_colors=True     # Enable colored output for better readability
    )
