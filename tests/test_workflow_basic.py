import pytest
import requests
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException

from app import WebsiteSummarizationWorkflow, ContentFetched
from llama_index.core.workflow import StartEvent, StopEvent

@pytest.fixture
def workflow(mock_llm):
    """Workflow instance with mocked LLM."""
    return WebsiteSummarizationWorkflow(llm=mock_llm)

class TestFetchWebsiteContent:
    """Test the fetch_website_content step."""
    
    @pytest.mark.asyncio
    async def test_fetch_content_success_with_web_reader(self, workflow, sample_url, mock_document):
        """Test successful content fetching using SimpleWebPageReader."""
        
        with patch('app.SimpleWebPageReader') as mock_reader_class:
            mock_reader = Mock()
            mock_reader.load_data.return_value = [mock_document]
            mock_reader_class.return_value = mock_reader
            
            start_event = StartEvent(url=sample_url)
            result = await workflow.fetch_website_content(start_event)
            
            assert isinstance(result, ContentFetched)
            assert result.content == mock_document.text
            assert result.url == sample_url
            mock_reader_class.assert_called_once_with(html_to_text=True)
            mock_reader.load_data.assert_called_once_with([sample_url])

    @pytest.mark.asyncio
    async def test_fetch_content_fallback_to_requests(self, workflow, sample_url, sample_content):
        """Test fallback to requests when SimpleWebPageReader returns no documents."""
        
        html_content = f"<html><body><h1>Test Title</h1><p>{sample_content}</p></body></html>"
        
        with patch('app.SimpleWebPageReader') as mock_reader_class, \
             patch('app.requests.get') as mock_requests:
            
            # Mock SimpleWebPageReader to return empty list
            mock_reader = Mock()
            mock_reader.load_data.return_value = []
            mock_reader_class.return_value = mock_reader
            
            # Mock requests response
            mock_response = Mock()
            mock_response.text = html_content
            mock_response.raise_for_status.return_value = None
            mock_requests.return_value = mock_response
            
            start_event = StartEvent(url=sample_url)
            result = await workflow.fetch_website_content(start_event)
            
            assert isinstance(result, ContentFetched)
            assert "Test Title" in result.content
            assert sample_content in result.content
            assert result.url == sample_url
            mock_requests.assert_called_once_with(sample_url, timeout=10)

    @pytest.mark.asyncio
    async def test_fetch_content_missing_url(self, workflow):
        """Test error handling when URL is missing."""
        
        start_event = StartEvent()  # No URL provided
        
        with pytest.raises(ValueError, match="URL is required"):
            await workflow.fetch_website_content(start_event)

    @pytest.mark.asyncio
    async def test_fetch_content_network_error(self, workflow, sample_url):
        """Test error handling for network failures."""
        
        with patch('app.SimpleWebPageReader') as mock_reader_class, \
             patch('app.requests.get') as mock_requests:
            
            # Mock SimpleWebPageReader to return empty list (triggering fallback)
            mock_reader = Mock()
            mock_reader.load_data.return_value = []
            mock_reader_class.return_value = mock_reader
            
            # Mock requests to raise an exception
            mock_requests.side_effect = requests.exceptions.RequestException("Connection failed")
            
            start_event = StartEvent(url=sample_url)
            
            with pytest.raises(HTTPException) as exc_info:
                await workflow.fetch_website_content(start_event)
            
            assert exc_info.value.status_code == 400
            assert "Failed to fetch content" in str(exc_info.value.detail)

class TestGenerateSummary:
    """Test the generate_summary step."""
    
    @pytest.mark.asyncio
    async def test_generate_summary_success(self, workflow, mock_llm, sample_url, sample_content):
        """Test successful summary generation."""
        
        content_event = ContentFetched(content=sample_content, url=sample_url)
        
        result = await workflow.generate_summary(content_event)
        
        assert isinstance(result, StopEvent)
        assert "summary" in result.result
        assert "url" in result.result
        assert result.result["url"] == sample_url
        assert result.result["summary"] == "This is a mocked summary of the website content."
        
        # Verify LLM was called with correct prompt
        mock_llm.acomplete.assert_called_once()
        call_args = mock_llm.acomplete.call_args[0][0]
        assert sample_url in call_args
        assert sample_content in call_args
        assert "comprehensive yet concise summary" in call_args

    @pytest.mark.asyncio
    async def test_generate_summary_content_truncation(self, workflow, mock_llm, sample_url):
        """Test content truncation for very long content."""
        
        # Create content longer than max_content_length (8000 chars)
        long_content = "x" * 10000
        content_event = ContentFetched(content=long_content, url=sample_url)
        
        await workflow.generate_summary(content_event)
        
        # Verify content was truncated
        call_args = mock_llm.acomplete.call_args[0][0]
        # The prompt includes the content, so check that it's reasonable length
        assert len(call_args) < 9000  # Should be less than original + prompt overhead

    @pytest.mark.asyncio
    async def test_generate_summary_llm_failure(self, workflow, mock_llm, sample_url, sample_content):
        """Test error handling when LLM fails."""
        
        mock_llm.acomplete.side_effect = Exception("LLM API failed")
        content_event = ContentFetched(content=sample_content, url=sample_url)
        
        with pytest.raises(HTTPException) as exc_info:
            await workflow.generate_summary(content_event)
        
        assert exc_info.value.status_code == 500
        assert "Failed to generate summary" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_generate_summary_empty_content(self, workflow, sample_url):
        """Test summary generation with empty content."""
        
        content_event = ContentFetched(content="", url=sample_url)
        
        result = await workflow.generate_summary(content_event)
        
        assert isinstance(result, StopEvent)
        assert result.result["url"] == sample_url

class TestWebsiteSummarizationWorkflow:
    """Test complete workflow integration."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_success(self, workflow, sample_url, mock_document):
        """Test complete workflow execution from start to finish."""
        
        with patch('app.SimpleWebPageReader') as mock_reader_class:
            mock_reader = Mock()
            mock_reader.load_data.return_value = [mock_document]
            mock_reader_class.return_value = mock_reader
            
            # Run complete workflow
            result = await workflow.run(url=sample_url)
            
            # Verify final result
            assert "summary" in result
            assert "url" in result
            assert result["url"] == sample_url
            assert isinstance(result["summary"], str)

    @pytest.mark.asyncio
    async def test_workflow_with_default_llm(self, sample_url, mock_document):
        """Test workflow initialization with default LLM."""
        
        with patch('app.OpenAI') as mock_openai_class, \
             patch('app.SimpleWebPageReader') as mock_reader_class:
            
            mock_llm = AsyncMock()
            mock_openai_class.return_value = mock_llm
            mock_llm.acomplete.return_value = "Default LLM summary"
            
            mock_reader = Mock()
            mock_reader.load_data.return_value = [mock_document]
            mock_reader_class.return_value = mock_reader
            
            workflow = WebsiteSummarizationWorkflow()
            
            # Verify default LLM initialization
            mock_openai_class.assert_called_once_with(model="gpt-4o-mini")
            assert workflow.llm == mock_llm
            
            # Test workflow execution
            result = await workflow.run(url=sample_url)
            assert result["summary"] == "Default LLM summary"