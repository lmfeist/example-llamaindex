# Entity and Node Deduplication for Property Graphs

This document describes the entity and node deduplication functionality implemented for Neo4j property graphs in the LlamaIndex Processing API.

## Overview

The deduplication system automatically identifies and merges duplicate entities and nodes in a Neo4j property graph using:
- **Vector Similarity**: Cosine similarity on embeddings
- **String Matching**: Edit distance and substring matching on names
- **Label Consistency**: Only merges nodes with identical labels
- **Dual Processing**: Handles both `__Entity__` and `__Node__` labels

## How It Works

### 1. Vector Index Creation
The system creates vector indexes for both entity and node embeddings:
```cypher
-- For entities
CREATE VECTOR INDEX entity IF NOT EXISTS
FOR (m:`__Entity__`)
ON m.embedding
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}

-- For nodes
CREATE VECTOR INDEX node IF NOT EXISTS
FOR (m:`__Node__`)
ON m.embedding
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
```

### 2. Duplicate Detection
For each entity and node type (`__Entity__` and `__Node__`), the system:
- Finds similar nodes using vector similarity (default threshold: 0.9)
- Applies string matching filters:
  - Substring containment (case-insensitive)
  - Edit distance matching (default max distance: 5)
- Ensures nodes have identical labels

### 3. Node/Entity Merging
When duplicates are found:
- The first node in alphabetical order becomes the canonical node
- All relationships from duplicate nodes are transferred to the canonical node
- Properties from duplicate nodes are merged into the canonical node
- Duplicate nodes are deleted

## Usage

### Automatic Deduplication
Entity and node deduplication runs automatically after property graph creation when processing PDF files. There is no separate endpoint - deduplication is integrated into the document processing workflow:

```bash
# Deduplication happens automatically after graph creation
POST /upload_pdfs
```

The deduplication process uses default parameters:
- **Similarity Threshold**: 0.9 (cosine similarity)
- **Word Edit Distance**: 5 (maximum Levenshtein distance)
- **Node Types**: Both `__Entity__` and `__Node__` labels are processed

### Response Format
The upload endpoint returns information about merged entities and nodes:

```json
{
    "message": "Successfully processed 3 documents and created property graph index. Merged 5 duplicate entities.",
    "files_processed": ["file1.pdf", "file2.pdf"],
    "documents_count": 3,
    "entities_merged": 5,
    "graph_store_type": "Neo4j"
}
```

Note: The `entities_merged` count includes both `__Entity__` and `__Node__` duplicates that were merged.

## Configuration Parameters

The deduplication process uses hardcoded parameters optimized for biomedical entities and nodes:

### Similarity Threshold: 0.9
- **Description**: Minimum cosine similarity for entities/nodes to be considered duplicates
- **Rationale**: High threshold reduces false positives in biomedical entity and node matching

### Word Edit Distance: 5
- **Description**: Maximum Levenshtein distance for string matching
- **Rationale**: Allows for common variations in biomedical entity/node naming (e.g., "TP53" vs "tp53")

### Node Types Processed
- **`__Entity__`**: Biomedical entities (genes, diseases, treatments, etc.)
- **`__Node__`**: General nodes in the property graph
- **Processing**: Both types are processed sequentially with the same parameters

## Example Scenarios

### Before Deduplication
```
-- __Entity__ nodes
- Entity: "TP53" (GENE)
- Entity: "tp53" (GENE)  
- Entity: "P53" (GENE)
- Entity: "tumor protein p53" (GENE)

-- __Node__ nodes  
- Node: "DNA repair pathway"
- Node: "dna repair pathway"
- Node: "DNA_repair_pathway"
```

### After Deduplication
```
-- __Entity__ nodes
- Entity: "P53" (GENE) [canonical entity with merged relationships]

-- __Node__ nodes
- Node: "DNA repair pathway" [canonical node with merged relationships]
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
- Non-blocking: if deduplication fails, the graph creation still succeeds

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
   - Consider processing smaller document batches

3. **Vector Index Errors**
   - Ensure Neo4j version supports vector indexing
   - Check that embeddings are properly formatted

### Logs to Check
```bash
# Application logs show deduplication progress
logger.info("Starting entity and node deduplication process...")
logger.info("Processing duplicates for __Entity__ nodes...")
logger.info("Merging __Entity__ nodes: ['tp53', 'TP53'] -> P53")
logger.info("Processing duplicates for __Node__ nodes...")
logger.info("Merging __Node__ nodes: ['dna repair pathway'] -> DNA repair pathway")
logger.info("Successfully merged 5 duplicate entities and nodes")
```

## Best Practices

1. **Monitor merge counts** in response messages to validate results
2. **Check logs** for deduplication progress and any errors
3. **Ensure sufficient Neo4j memory** for large document sets
4. **Backup graphs** before processing large document collections
5. **Verify APOC plugin** is installed and configured correctly