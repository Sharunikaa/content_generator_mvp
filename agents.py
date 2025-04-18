from crewai import Agent
from tools import (
    ValidateInputTool, GenerateContentIdeasTool, GenerateSEOKeywordsTool,
    CreateOutlineTool, GenerateContentDraftTool, CheckGrammarToneTool,
    SaveContentTool, FormatContentTool, SearchKeywordTool
)
import google.generativeai as genai
import os
import logging

# Setup logging to debug issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom LLM Wrapper for CrewAI
class GeminiLLM:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY is required")
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("GeminiLLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise

    def __call__(self, prompt, **kwargs):
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_output_tokens": kwargs.get("max_tokens", 300)
                },
                timeout=10
            )
            return response.text if hasattr(response, "text") else "No response"
        except Exception as e:
            logger.error(f"Error calling Gemini model: {str(e)}")
            return f"Error: {str(e)}"

    def supports_stop_words(self):
        # Return False to indicate Gemini doesn't support stop words
        return False

    def bind(self, **kwargs):
        # Return self to maintain chainability while keeping the same instance
        return self

    def predict(self, prompt, **kwargs):
        # Alias for __call__ as some LLM interfaces expect this method
        return self.__call__(prompt, **kwargs)

    def get_num_tokens(self, text):
        # Mock implementation since Gemini doesn't expose token counting
        return len(text.split())

# Initialize LLM
try:
    llm = GeminiLLM()
except Exception as e:
    logger.error(f"Failed to create GeminiLLM: {str(e)}")
    raise

# Agents
try:
    idea_intake_agent = Agent(
        role="Idea Intake Agent",
        goal="Validate and structure user inputs for content creation",
        backstory="Expert in parsing and validating content creation inputs",
        tools=[ValidateInputTool()],
        llm=llm,
        verbose=True
    )

    content_strategy_agent = Agent(
        role="Content Strategy Agent",
        goal="Create a content calendar based on user inputs",
        backstory="Specialist in scheduling and planning content",
        tools=[FormatContentTool()],
        llm=llm,
        verbose=True
    )

    idea_generation_agent = Agent(
        role="Idea Generation Agent",
        goal="Generate creative content ideas",
        backstory="Creative thinker with expertise in brainstorming",
        tools=[GenerateContentIdeasTool()],
        llm=llm,
        verbose=True
    )

    outline_creation_agent = Agent(
        role="Outline Creation Agent",
        goal="Create detailed content outlines with SEO integration",
        backstory="Expert in structuring content with SEO optimization",
        tools=[GenerateSEOKeywordsTool(), CreateOutlineTool(), SearchKeywordTool()],
        llm=llm,
        verbose=True
    )

    content_writing_agent = Agent(
        role="Content Writing Agent",
        goal="Generate full content drafts based on outlines",
        backstory="Skilled writer adapting to various styles and tones",
        tools=[GenerateContentDraftTool()],
        llm=llm,
        verbose=True
    )

    review_approval_agent = Agent(
        role="Review & Approval Agent",
        goal="Ensure content quality through grammar and tone checks",
        backstory="Meticulous editor with a focus on clarity and correctness",
        tools=[CheckGrammarToneTool()],
        llm=llm,
        verbose=True
    )

    save_schedule_agent = Agent(
        role="Save & Schedule Agent",
        goal="Save content and schedule it in a calendar",
        backstory="Organized manager of content storage and scheduling",
        tools=[SaveContentTool(), FormatContentTool()],
        llm=llm,
        verbose=True
    )
except Exception as e:
    logger.error(f"Failed to initialize agents: {str(e)}")
    raise