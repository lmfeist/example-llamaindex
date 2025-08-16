import pytest
import asyncio
from unittest.mock import AsyncMock, Mock

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = AsyncMock()
    llm.acomplete.return_value = "This is a mocked summary of the website content."
    return llm

@pytest.fixture
def sample_content():
    """Sample website content for testing."""
    return "This is sample website content with multiple paragraphs and useful information about testing workflows."

@pytest.fixture
def sample_url():
    """Sample URL for testing."""
    return "https://example.com"

@pytest.fixture
def mock_document(sample_content):
    """Mock document object from SimpleWebPageReader."""
    doc = Mock()
    doc.text = sample_content
    return doc