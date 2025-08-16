# Summarize Workflow Tests

This document explains how to run and understand the tests for the website summarization workflow.

## Test Overview

The test suite (`test_summarize.py`) provides comprehensive coverage for the website summarization workflow, including:

- **Unit tests** for individual workflow steps
- **Integration tests** for the complete workflow
- **API endpoint tests** for the FastAPI routes
- **Model validation tests** for Pydantic models

## Key Features

### Mocking Strategy

Since the tests don't have access to an OpenAI API key, all external dependencies are mocked:

- **OpenAI LLM calls** are mocked using `AsyncMock`
- **Web requests** are mocked to avoid external network calls
- **SimpleWebPageReader** is mocked for consistent test behavior

### Test Categories

1. **WebsiteSummarizationWorkflow Tests**:
   - Content fetching with SimpleWebPageReader
   - Fallback to requests when SimpleWebPageReader fails
   - Summary generation with content truncation
   - Error handling for missing URLs and network failures

2. **API Endpoint Tests**:
   - Successful summarization requests
   - Invalid URL handling
   - Workflow error propagation
   - CORS OPTIONS endpoint

3. **Model Tests**:
   - URL validation in URLRequest
   - Response model structure

4. **Integration Tests**:
   - End-to-end workflow execution with all mocks

## Running the Tests

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
python3 -m pytest test_summarize.py -v
```

### Run Specific Test Categories

```bash
# Run only workflow tests
python3 -m pytest test_summarize.py::TestWebsiteSummarizationWorkflow -v

# Run only API tests
python3 -m pytest test_summarize.py::TestSummarizeAPI -v

# Run only integration tests
python3 -m pytest test_summarize.py::TestIntegration -v
```

### Test Coverage

The tests cover:

- ✅ Content fetching from websites
- ✅ HTML parsing and cleaning
- ✅ Summary generation (mocked)
- ✅ Error handling and edge cases
- ✅ API endpoint functionality
- ✅ Request/response validation
- ✅ CORS support

## Test Architecture

The tests use several pytest features:

- **Fixtures**: For creating mock objects and test data
- **AsyncMock**: For mocking async function calls
- **Patch decorators**: For mocking external dependencies
- **HTTPX ASGITransport**: For testing FastAPI endpoints

## Example Test Output

```
============================= test session starts ==============================
test_summarize.py::TestWebsiteSummarizationWorkflow::test_fetch_website_content_with_simple_web_reader PASSED [  6%]
test_summarize.py::TestWebsiteSummarizationWorkflow::test_fetch_website_content_fallback_to_requests PASSED [ 13%]
...
======================== 15 passed, 9 warnings in 1.24s ========================
```

## Notes

- All external API calls are mocked, so no API keys are required
- Tests run quickly since no actual network requests are made
- The test suite validates both happy path and error scenarios
- Warnings about deprecated Pydantic features are expected and don't affect functionality