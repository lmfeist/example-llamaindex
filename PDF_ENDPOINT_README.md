# PDF to Neo4j Property Graph Endpoint

This document describes the new PDF processing endpoint that uploads PDF files and creates a Neo4j property graph index using LlamaIndex workflows.

## Overview

The `/upload_pdfs` endpoint allows you to:
1. Upload multiple PDF files via POST request
2. Extract text content from PDFs using LlamaIndex
3. Process the content to identify entities and relationships
4. Store the extracted knowledge graph in Neo4j as a property graph

## Endpoint Details

### POST /upload_pdfs

**Description**: Upload PDF files and process them into a Neo4j property graph index.

**Request Format**: `multipart/form-data`

**Parameters**:
- `files`: List of PDF files (required)
  - Content-Type: `application/pdf`
  - Multiple files supported

**Response Format**: JSON

**Response Model**:
```json
{
  "message": "Successfully processed X documents and created property graph index",
  "files_processed": ["file1.pdf", "file2.pdf"],
  "documents_count": 2,
  "graph_store_type": "Neo4j"
}
```

## Environment Variables

The following environment variables must be set for the endpoint to work:

### Required
- `NEO4J_PASSWORD`: Password for Neo4j database connection
- `OPENAI_API_KEY`: OpenAI API key for LLM and embedding operations

### Optional (with defaults)
- `NEO4J_URI`: Neo4j connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USERNAME`: Neo4j username (default: `neo4j`)

## Knowledge Graph Schema

The endpoint extracts the following types of entities and relationships:

### Entities
- **PERSON**: People mentioned in documents
- **ORGANIZATION**: Companies, institutions, groups
- **LOCATION**: Places, addresses, geographical locations
- **CONCEPT**: Abstract concepts, ideas, topics
- **TECHNOLOGY**: Technologies, tools, software
- **PRODUCT**: Products, services, offerings

### Relations
- **WORKS_AT**: Person works at organization
- **LOCATED_IN**: Entity is located in a place
- **RELATED_TO**: General relationship between entities
- **USES**: Entity uses technology/product
- **DEVELOPS**: Entity develops technology/product
- **MANAGES**: Person manages entity
- **PART_OF**: Entity is part of another entity

## Usage Examples

### Using curl
```bash
# Upload a single PDF
curl -X POST http://localhost:8000/upload_pdfs \
  -F "files=@document1.pdf"

# Upload multiple PDFs
curl -X POST http://localhost:8000/upload_pdfs \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

### Using Python requests
```python
import requests

# Upload single PDF
with open('document.pdf', 'rb') as f:
    files = {'files': ('document.pdf', f, 'application/pdf')}
    response = requests.post('http://localhost:8000/upload_pdfs', files=files)
    print(response.json())

# Upload multiple PDFs
files = [
    ('files', ('doc1.pdf', open('doc1.pdf', 'rb'), 'application/pdf')),
    ('files', ('doc2.pdf', open('doc2.pdf', 'rb'), 'application/pdf'))
]
response = requests.post('http://localhost:8000/upload_pdfs', files=files)
print(response.json())

# Close files
for _, (_, file_obj, _) in files:
    file_obj.close()
```

### Using JavaScript/Fetch
```javascript
const formData = new FormData();
formData.append('files', pdfFile1, 'document1.pdf');
formData.append('files', pdfFile2, 'document2.pdf');

fetch('http://localhost:8000/upload_pdfs', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

## Error Handling

The endpoint handles various error conditions:

### 400 Bad Request
- Non-PDF files uploaded
- Invalid file format
- File processing errors

### 500 Internal Server Error
- Neo4j connection issues
- OpenAI API errors
- LlamaIndex processing errors
- Missing required environment variables

### Example Error Response
```json
{
  "status_code": 400,
  "detail": "File document.txt is not a PDF file. Only PDF files are allowed.",
  "path": "/upload_pdfs",
  "method": "POST"
}
```

## Neo4j Integration

The endpoint uses LlamaIndex's `PropertyGraphIndex` with:
- **Graph Store**: Neo4jPropertyGraphStore
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: GPT-4o-mini for entity/relation extraction
- **Extractor**: SchemaLLMPathExtractor with predefined schema

### Neo4j Setup

Ensure Neo4j is running and accessible:

1. **Install Neo4j**: Follow [Neo4j installation guide](https://neo4j.com/docs/operations-manual/current/installation/)
2. **Start Neo4j**: `neo4j start`
3. **Set credentials**: Use Neo4j Browser to set username/password
4. **Configure environment variables** as described above

## Testing

Use the provided test script to verify functionality:

```bash
# Install test dependencies
pip install requests

# Run the test
python test_pdf_endpoint.py
```

## Performance Considerations

- **File Size**: Large PDFs may take longer to process
- **Batch Size**: Processing multiple files increases memory usage
- **Neo4j Performance**: Ensure Neo4j has adequate resources
- **OpenAI Rate Limits**: Monitor API usage for large document sets

## Security Notes

- Files are temporarily stored during processing and cleaned up automatically
- Ensure Neo4j credentials are properly secured
- OpenAI API key should be kept confidential
- Consider implementing file size limits for production use

## Troubleshooting

### Common Issues

1. **"NEO4J_PASSWORD environment variable is required"**
   - Set the NEO4J_PASSWORD environment variable

2. **"Failed to connect to Neo4j"**
   - Verify Neo4j is running
   - Check connection URI and credentials
   - Ensure network connectivity

3. **"OpenAI API error"**
   - Verify OPENAI_API_KEY is set correctly
   - Check API quota and rate limits
   - Ensure account has access to required models

4. **"Failed to process PDF files"**
   - Verify files are valid PDFs
   - Check file permissions
   - Ensure sufficient disk space for temporary files

### Debug Mode

The application runs in debug mode by default, providing detailed error messages and stack traces in the logs.

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`