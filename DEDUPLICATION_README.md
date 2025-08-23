# Entity Deduplication for Property Graphs

This document describes the entity deduplication functionality implemented for Neo4j property graphs in the LlamaIndex Processing API.

## Overview

The entity deduplication system automatically identifies and merges duplicate entities in a Neo4j property graph using:
- **Vector Similarity**: Cosine similarity on entity embeddings
- **String Matching**: Edit distance and substring matching on entity names
- **Label Consistency**: Only merges entities with identical labels

## How It Works

### 1. Vector Index Creation
The system creates a vector index on entity embeddings:
```cypher
CREATE VECTOR INDEX entity IF NOT EXISTS
FOR (m:`__Entity__`)
ON m.embedding
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
```

### 2. Duplicate Detection
For each entity, the system:
- Finds similar entities using vector similarity (default threshold: 0.9)
- Applies string matching filters:
  - Substring containment (case-insensitive)
  - Edit distance matching (default max distance: 5)
- Ensures entities have identical labels

### 3. Entity Merging
When duplicates are found:
- The first entity in alphabetical order becomes the canonical entity
- All relationships from duplicate entities are transferred to the canonical entity
- Properties from duplicate entities are merged into the canonical entity
- Duplicate entities are deleted

## Usage

### Automatic Deduplication
Entity deduplication runs automatically after property graph creation when processing PDF files:

```python
# Deduplication happens automatically after graph creation
POST /upload_pdfs
```

### Manual Deduplication
You can also run deduplication on an existing graph:

```python
# Manual deduplication endpoint
POST /deduplicate
{
    "similarity_threshold": 0.9,    # Optional, default: 0.9
    "word_edit_distance": 5         # Optional, default: 5
}
```

### Response Format
Both endpoints return information about merged entities:

```json
{
    "message": "Successfully processed 3 documents and created property graph index. Merged 5 duplicate entities.",
    "files_processed": ["file1.pdf", "file2.pdf"],
    "documents_count": 3,
    "entities_merged": 5,
    "graph_store_type": "Neo4j"
}
```

## Configuration Parameters

### Similarity Threshold (0.0 - 1.0)
- **Default**: 0.9
- **Description**: Minimum cosine similarity for entities to be considered duplicates
- **Higher values**: More strict matching (fewer false positives)
- **Lower values**: More lenient matching (may create false positives)

### Word Edit Distance (integer)
- **Default**: 5
- **Description**: Maximum Levenshtein distance for string matching
- **Higher values**: More lenient string matching
- **Lower values**: More strict string matching

## Example Scenarios

### Before Deduplication
```
- Entity: "TP53" (GENE)
- Entity: "tp53" (GENE)  
- Entity: "P53" (GENE)
- Entity: "tumor protein p53" (GENE)
```

### After Deduplication
```
- Entity: "P53" (GENE) [canonical entity with merged relationships]
```

## Prerequisites

### Neo4j Setup
- Neo4j database with APOC plugin installed
- Vector indexing capability (Neo4j 5.11+ recommended)
- Sufficient memory for vector operations

### Environment Variables
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

## Error Handling

The deduplication process includes comprehensive error handling:
- Graceful fallback if vector index creation fails
- Detailed logging of merge operations
- Transaction safety for merge operations
- Returns count of successfully merged entities

## Performance Considerations

- **Vector Index**: May take time to build on large graphs
- **Memory Usage**: Vector similarity operations are memory-intensive
- **Batch Processing**: Large numbers of duplicates are processed in batches
- **Logging**: Detailed logs help monitor progress and debug issues

## Troubleshooting

### Common Issues

1. **APOC Plugin Missing**
   - Install APOC plugin in Neo4j
   - Restart Neo4j service

2. **Insufficient Memory**
   - Increase Neo4j heap size
   - Consider processing smaller batches

3. **Vector Index Errors**
   - Ensure Neo4j version supports vector indexing
   - Check that embeddings are properly formatted

### Logs to Check
```bash
# Application logs show deduplication progress
logger.info("Starting entity deduplication process...")
logger.info("Merging entities: ['tp53', 'TP53'] -> P53")
logger.info("Successfully merged 5 duplicate entities")
```

## Best Practices

1. **Run deduplication after initial graph creation**
2. **Monitor merge counts to validate results**
3. **Adjust thresholds based on domain-specific needs**
4. **Regular deduplication for frequently updated graphs**
5. **Backup graphs before running deduplication on production data**