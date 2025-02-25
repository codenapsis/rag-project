import unittest
import logging
from src.llm.llm_service import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestLLMService(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up LLMService test...")
        self.llm_service = LLMService()
        logger.info("LLMService initialized")
        
    def test_configure_llm_settings(self):
        logger.info("Testing LLM settings configuration...")
        settings = {
            "model": "test-model",
            "temperature": 0.7
        }
        logger.info(f"Test settings: {settings}")
        
        self.llm_service.configure_llm_settings(settings)
        logger.info("LLM settings configured")

    def tearDown(self):
        logger.info("Cleaning up after LLMService test...")
