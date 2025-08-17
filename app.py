import os
import traceback
import logging
import tempfile
import shutil
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException, Response, Request, UploadFile, File
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from contextlib import asynccontextmanager

# Disable LlamaIndex instrumentation to prevent context token errors
os.environ["LLAMA_INDEX_DISABLE_TELEMETRY"] = "true"

# Also disable OpenTelemetry instrumentation if it's enabled
os.environ["OTEL_SDK_DISABLED"] = "true"

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
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
import requests
from bs4 import BeautifulSoup


# Events for the workflow
class ContentFetched(Event):
    content: str
    url: str


class PDFsProcessed(Event):
    documents: List[Any]
    file_names: List[str]


# Website Summarization Workflow
class WebsiteSummarizationWorkflow(Workflow):
    def __init__(self, llm: Optional[Any] = None):
        super().__init__()
        self.llm = llm or OpenAI(model="gpt-4o-mini")

    @step
    async def fetch_website_content(self, ev: StartEvent) -> ContentFetched:
        """Fetch the content of the website."""
        url = getattr(ev, 'url', None)
                      
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


# PDF to Neo4j Property Graph Workflow
class PDFToGraphWorkflow(Workflow):
    def __init__(self, llm: Optional[Any] = None):
        super().__init__()
        self.llm = llm or OpenAI(model="gpt-4o-mini", temperature=0.0)

    @step
    async def process_pdfs(self, ev: StartEvent) -> PDFsProcessed:
        """Process uploaded PDF files and extract documents."""
        pdf_files = getattr(ev, 'pdf_files', None)
        
        if not pdf_files:
            raise ValueError("PDF files are required")
        
        try:
            # Create temporary directory for PDF processing
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            file_names = []
            
            # Save uploaded files temporarily
            for pdf_file in pdf_files:
                file_path = os.path.join(temp_dir, pdf_file.filename)
                with open(file_path, "wb") as f:
                    content = await pdf_file.read()
                    f.write(content)
                file_paths.append(file_path)
                file_names.append(pdf_file.filename)
            
            # Process PDFs with LlamaIndex
            documents = SimpleDirectoryReader(temp_dir).load_data()
            
            # Clean up temporary files
            shutil.rmtree(temp_dir)
            
            return PDFsProcessed(documents=documents, file_names=file_names)
            
        except Exception as e:
            # Clean up temporary directory if it exists
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            logger.error(f"Failed to process PDF files:\n{traceback.format_exc()}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF files: {str(e)}")

    @step
    async def create_property_graph(self, ev: PDFsProcessed) -> StopEvent:
        """Create property graph index in Neo4j from processed documents."""
        try:
            # Define biomedical entities and relations for knowledge extraction
            entities = Literal["GENE", "PATHWAY", "DISEASE", "TREATMENT", "TREATMENT_OUTCOME"]
            relations = Literal[
                "REGULATES", "INTERACTS_WITH", "CAUSES", "TREATS", "RESULTS_IN",
                "ASSOCIATED_WITH", "TARGETS", "INHIBITS", "ACTIVATES", "PART_OF",
                "RESPONDS_TO", "MODULATES", "INVOLVED_IN", "CAUSED_BY", "TREATED_BY"
            ]
            
            # Create biomedical validation schema
            validation_schema = {
                "GENE": [
                    "REGULATES", "INTERACTS_WITH", "CAUSES", "ASSOCIATED_WITH", 
                    "TARGETS", "MODULATES", "INVOLVED_IN", "PART_OF"
                ],
                "PATHWAY": [
                    "REGULATES", "INTERACTS_WITH", "ASSOCIATED_WITH", 
                    "MODULATES", "INVOLVED_IN", "PART_OF", "TARGETS"
                ],
                "DISEASE": [
                    "CAUSED_BY", "ASSOCIATED_WITH", "RESPONDS_TO", "TREATED_BY",
                    "RESULTS_IN", "INVOLVED_IN", "TARGETS"
                ],
                "TREATMENT": [
                    "TREATS", "TARGETS", "INHIBITS", "ACTIVATES", "RESULTS_IN",
                    "MODULATES", "INTERACTS_WITH", "ASSOCIATED_WITH"
                ],
                "TREATMENT_OUTCOME": [
                    "RESULTS_IN", "ASSOCIATED_WITH", "CAUSED_BY", "INVOLVED_IN",
                    "RESPONDS_TO", "MODULATES"
                ],
            }

            # Define entity properties for schema generation
            entity_props = {
                "GENE": {"description": "A gene or genetic element"},
                "PATHWAY": {"description": "A biological pathway"},
                "DISEASE": {"description": "A disease or medical condition"},
                "TREATMENT": {"description": "A treatment or therapeutic intervention"},
                "TREATMENT_OUTCOME": {"description": "An outcome or result of treatment"}
            }

            # Initialize knowledge graph extractor
            kg_extractor = SchemaLLMPathExtractor(
                llm=self.llm,
                possible_entities=entities,
                possible_relations=relations,
                kg_validation_schema=validation_schema,
                entity_props=entity_props,
                strict=False,  # Set to False to avoid Pydantic schema generation issues
            )

            # Connect to Neo4j using environment variables
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD")
            
            if not neo4j_password:
                raise ValueError("NEO4J_PASSWORD environment variable is required")

            graph_store = Neo4jPropertyGraphStore(
                username=neo4j_username,
                password=neo4j_password,
                url=neo4j_uri,
            )

            # Create PropertyGraphIndex
            index = PropertyGraphIndex.from_documents(
                ev.documents,
                kg_extractors=[kg_extractor],
                embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
                property_graph_store=graph_store,
                show_progress=True,
            )

            return StopEvent(result={
                "message": f"Successfully processed {len(ev.documents)} documents and created property graph index",
                "files_processed": ev.file_names,
                "documents_count": len(ev.documents),
                "graph_store_type": "Neo4j"
            })
            
        except Exception as e:
            logger.error(f"Failed to create property graph:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to create property graph: {str(e)}")


# Global variables
summarization_workflow = None
pdf_workflow = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global summarization_workflow, pdf_workflow
    
    # Initialize the website summarization workflow
    summarization_workflow = WebsiteSummarizationWorkflow()
    
    # Initialize the PDF to graph workflow
    pdf_workflow = PDFToGraphWorkflow()
    
    yield
    
    # Cleanup if needed
    pass


# Initialize FastAPI app
app = FastAPI(
    title="LlamaIndex Processing API",
    description="Website summarization and PDF to Neo4j property graph processing using LlamaIndex",
    version="2.0.0",
    lifespan=lifespan,
    debug=True  # Enable debug mode for detailed error responses
)

# Add CORS middleware to ensure CORS headers are always added
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=86400,  # Cache preflight for 24 hours
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


class PDFProcessResponse(BaseModel):
    message: str
    files_processed: List[str]
    documents_count: int
    graph_store_type: str


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="LlamaIndex Processing API is running"
    )


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the LlamaIndex Processing API",
        "description": "Get AI-powered summaries of website content and process PDFs into Neo4j property graphs",
        "endpoints": {
            "POST /summarize": "Get a summary of a website",
            "OPTIONS /summarize": "Get allowed methods for summarize endpoint",
            "POST /upload_pdfs": "Upload PDF files and create Neo4j property graph index",
            "OPTIONS /upload_pdfs": "Get allowed methods for upload_pdfs endpoint",
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
    Returns allowed methods. CORS headers are handled by middleware.
    """
    response = Response()
    response.headers["Allow"] = "POST, OPTIONS"
    
    return response


# PDF to Neo4j Property Graph endpoint
@app.post("/upload_pdfs", response_model=PDFProcessResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload PDF files and process them into a Neo4j property graph index.
    
    Args:
        files: List of PDF files to upload and process
        
    Returns:
        PDFProcessResponse containing processing results and metadata
    """
    try:
        # Validate file types
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a PDF file. Only PDF files are allowed."
                )
        
        # Run the PDF processing workflow
        result = await pdf_workflow.run(pdf_files=files)
        
        return PDFProcessResponse(
            message=result["message"],
            files_processed=result["files_processed"],
            documents_count=result["documents_count"],
            graph_store_type=result["graph_store_type"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the full exception with stack trace for debugging
        logger.error(f"Error in upload_pdfs endpoint:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing PDF files: {str(e)}"
        )


@app.options("/upload_pdfs")
async def upload_pdfs_options():
    """
    Handle OPTIONS request for the upload_pdfs endpoint.
    Returns allowed methods. CORS headers are handled by middleware.
    """
    response = Response()
    response.headers["Allow"] = "POST, OPTIONS"
    
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
