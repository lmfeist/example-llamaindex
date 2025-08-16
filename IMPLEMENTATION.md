# Website Summarization with LlamaIndex Workflow

## Overview

This implementation provides a complete API service that accepts website URLs and returns AI-generated summaries using LlamaIndex's Workflow system. The solution includes both the original document querying functionality and the new website summarization feature.

## Key Features

- **LlamaIndex Workflow**: Uses the latest LlamaIndex Workflow architecture for building agent-like systems
- **Website Content Extraction**: Supports multiple methods for extracting website content
- **AI Summarization**: Uses OpenAI GPT models to generate comprehensive summaries
- **Robust Error Handling**: Includes fallback mechanisms and proper error responses
- **FastAPI Integration**: RESTful API with automatic documentation
- **Production Ready**: Includes proper async handling and response models

## Architecture

### Workflow Components

1. **WebsiteSummarizationWorkflow**: Main workflow class that orchestrates the summarization process
2. **ContentFetched Event**: Custom event that carries website content between workflow steps
3. **Two-Step Process**:
   - `fetch_website_content`: Extracts text content from the provided URL
   - `generate_summary`: Uses LLM to create a summary of the content

### API Endpoints

- `POST /summarize`: New endpoint for website summarization
- `POST /query`: Original document querying endpoint
- `GET /health`: Health check endpoint
- `GET /docs`: Interactive API documentation

## Implementation Details

### Website Content Extraction

The workflow uses a dual approach for content extraction:

1. **Primary**: LlamaIndex `SimpleWebPageReader` with HTML-to-text conversion
2. **Fallback**: Direct web scraping with BeautifulSoup for cleaning and text extraction

### Workflow Steps

```python
@step
async def fetch_website_content(self, ev: StartEvent) -> ContentFetched:
    # Extracts content from URL and returns ContentFetched event

@step  
async def generate_summary(self, ev: ContentFetched) -> StopEvent:
    # Takes content and generates summary using LLM
```

### Request/Response Models

```python
class URLRequest(BaseModel):
    url: HttpUrl  # Validates URL format

class SummaryResponse(BaseModel):
    summary: str
    url: str
```

## Setup and Configuration

### Requirements

- Python 3.8+
- OpenAI API key
- Required packages (see requirements.txt)

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export PORT=8000  # Optional, defaults to 8000
```

### Installation

```bash
pip install -r requirements.txt
```

### Running the Service

```bash
python app.py
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Usage Examples

### Summarize a Website

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

### Response

```json
{
  "summary": "This article discusses the main concepts of...",
  "url": "https://example.com/article"
}
```

## Error Handling

The implementation includes comprehensive error handling:

- **URL Validation**: Pydantic validates URL format
- **HTTP Errors**: Proper status codes for different failure scenarios
- **Content Extraction Failures**: Fallback mechanisms and informative error messages
- **LLM Failures**: Timeout and token limit handling

## Content Processing

- **Token Limits**: Content is truncated to 8000 characters to stay within LLM context windows
- **Content Cleaning**: Removes scripts, styles, and excessive whitespace
- **Encoding Handling**: Properly handles different character encodings

## Scalability Considerations

- **Async Processing**: Full async/await implementation for better concurrency
- **Connection Pooling**: Uses aiohttp for efficient HTTP requests
- **Resource Management**: Proper cleanup and context management
- **Timeout Handling**: Prevents hanging requests

## Testing

Run the basic structure test:

```bash
python simple_test.py
```

For full workflow testing (requires OpenAI API key):

```bash
python test_workflow.py
```

## Integration with Existing Functionality

The implementation preserves all existing functionality:
- Original document querying still works via `/query` endpoint
- Health checks remain functional
- All existing response models are maintained
- Backward compatibility is ensured

## Production Deployment

The service is designed for deployment on platforms like:
- Koyeb (as per original README)
- Heroku
- Docker containers
- AWS Lambda (with modifications)

The FastAPI application includes proper lifespan management for initialization and cleanup of resources.