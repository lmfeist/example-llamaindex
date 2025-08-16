#!/usr/bin/env python3

import asyncio
import os
from app import WebsiteSummarizationWorkflow

async def test_workflow():
    """Test the website summarization workflow with a simple URL."""
    
    # Set a dummy OpenAI API key for testing (you would need a real one for actual use)
    os.environ.setdefault('OPENAI_API_KEY', 'your-openai-api-key-here')
    
    # Create workflow instance
    workflow = WebsiteSummarizationWorkflow()
    
    # Test with a simple, reliable website
    test_url = "https://httpbin.org/html"
    
    try:
        print(f"Testing website summarization with URL: {test_url}")
        result = await workflow.run(url=test_url)
        
        print("✅ Workflow completed successfully!")
        print(f"URL: {result['url']}")
        print(f"Summary: {result['summary'][:200]}...")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Website Summarization Workflow")
    success = asyncio.run(test_workflow())
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")