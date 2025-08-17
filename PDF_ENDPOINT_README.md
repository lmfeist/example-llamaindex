# PDF to Neo4j Property Graph Endpoint

This document describes the new PDF processing endpoint that uploads biomedical research PDF files and creates a Neo4j property graph index using LlamaIndex workflows.

## Overview

The `/upload_pdfs` endpoint allows you to:
1. Upload multiple biomedical research PDF files via POST request
2. Extract text content from PDFs using LlamaIndex
3. Process the content to identify biomedical entities (genes, pathways, diseases, treatments, outcomes) and their relationships
4. Store the extracted biomedical knowledge graph in Neo4j as a property graph for research analysis

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
  "message": "Successfully processed 2 documents and created property graph index",
  "files_processed": ["gene_therapy_study.pdf", "pathway_analysis.pdf"],
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

The endpoint extracts biomedical entities and relationships from research documents:

### Entities
- **GENE**: Genes, genetic markers, DNA sequences
- **PATHWAY**: Biological pathways, metabolic processes, signaling cascades
- **DISEASE**: Diseases, conditions, disorders, syndromes
- **TREATMENT**: Treatments, therapies, drugs, interventions
- **TREATMENT_OUTCOME**: Treatment results, outcomes, responses, side effects

### Relations
- **REGULATES**: Gene/pathway regulates another entity
- **INTERACTS_WITH**: Direct interaction between entities
- **CAUSES**: Entity causes disease or outcome
- **TREATS**: Treatment addresses disease
- **RESULTS_IN**: Action leads to specific outcome
- **ASSOCIATED_WITH**: Statistical or observed association
- **TARGETS**: Treatment targets specific gene/pathway
- **INHIBITS**: Entity inhibits function of another
- **ACTIVATES**: Entity activates function of another
- **PART_OF**: Entity is component of larger system
- **RESPONDS_TO**: Entity responds to treatment/stimulus
- **MODULATES**: Entity modifies function of another
- **INVOLVED_IN**: Entity participates in process
- **CAUSED_BY**: Disease/outcome caused by entity
- **TREATED_BY**: Disease addressed by treatment

## Usage Examples

### Using curl
```bash
# Upload a single biomedical research PDF
curl -X POST http://localhost:8000/upload_pdfs \
  -F "files=@cancer_research_paper.pdf"

# Upload multiple research PDFs
curl -X POST http://localhost:8000/upload_pdfs \
  -F "files=@gene_therapy_study.pdf" \
  -F "files=@pathway_analysis.pdf"
```

### Using Python requests
```python
import requests

# Upload single biomedical research PDF
with open('alzheimers_research.pdf', 'rb') as f:
    files = {'files': ('alzheimers_research.pdf', f, 'application/pdf')}
    response = requests.post('http://localhost:8000/upload_pdfs', files=files)
    print(response.json())

# Upload multiple research PDFs
files = [
    ('files', ('genomics_study.pdf', open('genomics_study.pdf', 'rb'), 'application/pdf')),
    ('files', ('drug_trial_results.pdf', open('drug_trial_results.pdf', 'rb'), 'application/pdf'))
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
formData.append('files', pdfFile1, 'clinical_trial_report.pdf');
formData.append('files', pdfFile2, 'biomarker_analysis.pdf');

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
  "detail": "File research_data.txt is not a PDF file. Only PDF files are allowed.",
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

### Biomedical Domain Optimization

This endpoint is specifically optimized for biomedical and life sciences research:

- **Entity Recognition**: Trained to identify genes, pathways, diseases, treatments, and outcomes
- **Relationship Extraction**: Focuses on biomedical relationships like gene regulation, drug targets, and treatment outcomes
- **Research Paper Processing**: Optimized for scientific literature format and terminology
- **Clinical Data**: Capable of processing clinical trial reports and treatment outcome studies

## Testing

Use the provided test script to verify functionality:

```bash
# Install test dependencies
pip install requests

# Run the test
python test_pdf_endpoint.py
```

## Performance Considerations

- **File Size**: Large research papers and clinical studies may take longer to process
- **Batch Size**: Processing multiple research documents increases memory usage
- **Neo4j Performance**: Ensure Neo4j has adequate resources for complex biomedical graphs
- **OpenAI Rate Limits**: Monitor API usage for large research document collections
- **Entity Complexity**: Biomedical texts with dense terminology may require more processing time

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
   - Verify files are valid research PDFs (not scanned images)
   - Check file permissions
   - Ensure sufficient disk space for temporary files
   - Some research PDFs may have complex formatting that affects text extraction

### Debug Mode

The application runs in debug mode by default, providing detailed error messages and stack traces in the logs.

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`