from crewai.tools import BaseTool
from typing import Dict, List, Any
from datetime import datetime
import os
import json
import google.generativeai as genai
from spellchecker import SpellChecker
import streamlit as st
import requests

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Grammar Check Tool
class CheckGrammarToneTool(BaseTool):
    name: str = "Check Grammar Tone Tool"
    description: str = "Checks grammar and tone, providing corrections."

    def _run(self, content: str) -> Dict[str, Any]:
        if not content or not isinstance(content, str):
            return {"content": "Empty content", "suggestions": [], "corrected": "Empty content"}

        spell = SpellChecker()
        words = content.split()
        misspelled = spell.unknown(words) or set()
        suggestions = []
        corrected_content = content

        for word in misspelled:
            replacements = spell.candidates(word) or []
            suggestions.append({
                "message": f"Possible misspelling: {word}",
                "replacements": list(replacements)
            })
            # Auto-correct with the first replacement, if available
            if replacements:
                corrected_content = corrected_content.replace(word, list(replacements)[0])

        # Basic tone check (simplified, using LLM for tone suggestions)
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            tone_prompt = f"Analyze the tone of this text: '{content}'. Suggest improvements if it seems overly formal, casual, or inconsistent."
            response = model.generate_content(tone_prompt)
            tone_suggestion = response.text if hasattr(response, "text") else "No tone suggestions."
        except Exception as e:
            tone_suggestion = f"Tone check failed: {str(e)}"

        return {
            "content": content,
            "suggestions": suggestions,
            "corrected": corrected_content,
            "tone_suggestion": tone_suggestion
        }

# Input Validation Tool
class ValidateInputTool(BaseTool):
    name: str = "Validate Input Tool"
    description: str = "Validates user inputs for required fields and content type."

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ["topic", "target_audience", "tone"]
        for field in required_fields:
            if not inputs.get(field):
                raise ValueError(f"{field} is required.")
        if inputs.get("content_type") not in ["Blog", "Social Post"]:
            raise ValueError("Invalid content type.")
        if "num_pieces" not in inputs or not isinstance(inputs["num_pieces"], int):
            inputs["num_pieces"] = 1
        return inputs

# Content Ideas Tool
class GenerateContentIdeasTool(BaseTool):
    name: str = "Generate Content Ideas Tool"
    description: str = "Generates 3–5 content ideas using Gemini API."

    def _run(self, inputs: Dict[str, Any]) -> List[str]:
        if not inputs:
            return ["No valid inputs provided"]

        prompt = f"""
        Generate 3–5 ideas for a {inputs.get('tone', 'casual').lower()} {inputs.get('content_type', 'social post').lower()}
        about {inputs.get('topic', 'general topic')} for {inputs.get('target_audience', 'general audience')}.
        Each idea should be a clear, single sentence suitable for all audiences.
        """
        for attempt in range(3):
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt, generation_config={"temperature": 0.7, "max_output_tokens": 150})
                if not hasattr(response, "text"):
                    continue
                ideas = [idea.strip() for idea in response.text.strip().split("\n") if idea.strip() and not idea.startswith(("-", "*", "#"))]
                return ideas[:5] or ["No valid ideas generated"]
            except Exception as e:
                st.warning(f"Error generating ideas (attempt {attempt+1}): {str(e)}")
        return ["Error generating ideas after retries"]

# SEO Keywords Tool
class GenerateSEOKeywordsTool(BaseTool):
    name: str = "Generate SEO Keywords Tool"
    description: str = "Generates mock SEO keywords for a topic."

    def _run(self, topic: str) -> List[str]:
        if not topic or not isinstance(topic, str):
            return [f"default_keyword_{i}" for i in range(1, 4)]
        return [f"{topic.lower().replace(' ', '_')}_keyword_{i}" for i in range(1, 4)]

# Keyword Search Tool (Mock SEO Search)
class SearchKeywordTool(BaseTool):
    name: str = "Search Keyword Tool"
    description: str = "Searches for keyword meanings or related terms."

    def _run(self, keyword: str) -> Dict[str, Any]:
        if not keyword or not isinstance(keyword, str):
            return {"keyword": "unknown", "meaning": "No keyword provided", "related": []}
        try:
            # Mock API call to a dictionary or SEO service (replace with real API like Google Trends)
            response = {
                "keyword": keyword,
                "meaning": f"Definition of {keyword}: A term related to the topic.",
                "related": [f"{keyword}_variant_{i}" for i in range(1, 3)]
            }
            return response
        except Exception as e:
            return {"keyword": keyword, "meaning": f"Error searching: {str(e)}", "related": []}

# Outline Creation Tool
class CreateOutlineTool(BaseTool):
    name: str = "Create Outline Tool"
    description: str = "Creates a detailed H1-H3 outline with meaningful content."

    def _run(self, idea: str, keywords: List[str]) -> Dict[str, Any]:
        if not idea or not isinstance(idea, str):
            idea = "Default idea"
        if not keywords or not isinstance(keywords, list):
            keywords = ["default_keyword_1", "default_keyword_2"]

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            Create a detailed outline for a content piece based on the idea: '{idea}'.
            Include:
            - H1: Main title (based on the idea).
            - H2: Two subsections, each with a title and 1-2 sentences of meaningful content related to {', '.join(keywords)}.
            - H3: One sub-subsection per H2 with a title and a brief description.
            """
            response = model.generate_content(prompt, generation_config={"temperature": 0.6, "max_output_tokens": 300})
            if not hasattr(response, "text"):
                raise ValueError("No valid outline generated")

            # Parse response into structured outline
            lines = response.text.strip().split("\n")
            outline = {"H1": idea.split('.')[0] if '.' in idea else idea, "H2": [], "H3": []}
            current_h2 = None
            for line in lines:
                line = line.strip()
                if line.startswith("# ") or line.startswith("H1"):
                    outline["H1"] = line.replace("# ", "").replace("H1", "").strip()
                elif line.startswith("## ") or line.startswith("H2"):
                    current_h2 = {"title": line.replace("## ", "").replace("H2", "").strip(), "content": ""}
                    outline["H2"].append(current_h2)
                elif line.startswith("### ") or line.startswith("H3"):
                    outline["H3"].append({
                        "title": line.replace("### ", "").replace("H3", "").strip(),
                        "content": "Brief description of this subsection."
                    })
                elif current_h2 and line:
                    current_h2["content"] += line + " "
            
            # Ensure at least two H2s
            if len(outline["H2"]) < 2:
                outline["H2"].extend([
                    {"title": f"Section {i+1}", "content": f"Discuss {keywords[i % len(keywords)]}"} 
                    for i in range(len(outline["H2"]), 2)
                ])
            return outline
        except Exception as e:
            return {
                "H1": idea,
                "H2": [{"title": f"Section {i+1}", "content": f"Discuss {keywords[i % len(keywords)]}"} for i in range(2)],
                "H3": [{"title": "Subsection", "content": "Details"}]
            }

# Content Draft Tool
class GenerateContentDraftTool(BaseTool):
    name: str = "Generate Content Draft Tool"
    description: str = "Generates a full content draft based on an outline."

    def _run(self, outline: Dict[str, Any], tone: str, content_type: str) -> str:
        if not outline or not isinstance(outline, dict):
            return "Error: Invalid outline"
        if not tone or not isinstance(tone, str):
            tone = "casual"
        if not content_type or content_type not in ["Blog", "Social Post"]:
            content_type = "Social Post"

        prompt = f"""
        Write a {tone.lower()} {content_type.lower()} based on:
        - Title: {outline.get('H1', 'Default Title')}
        - Sections: {', '.join([h2.get('title', 'Section') for h2 in outline.get('H2', [])])}
        - Content: {'. '.join([h2.get('content', '') for h2 in outline.get('H2', [])])}
        {'Keep it under 280 characters with 1–2 emojis for social posts.' if content_type == 'Social Post' else 'Write 200–300 words for a blog.'}
        """
        for attempt in range(3):
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt, generation_config={"temperature": 0.8, "max_output_tokens": 300})
                if not hasattr(response, "text"):
                    continue
                return response.text.strip()
            except Exception as e:
                st.warning(f"Error generating draft (attempt {attempt+1}): {str(e)}")
        return "Error generating draft after retries"

# Save Content Tool
class SaveContentTool(BaseTool):
    name: str = "Save Content Tool"
    description: str = "Saves content to a project folder."

    def _run(self, content: Dict[str, Any], timeline: str) -> str:
        if not content or not isinstance(content, dict):
            content = {"error": "Invalid content"}
        if not timeline or not isinstance(timeline, str):
            timeline = str(datetime.now().date())

        project_folder = "content_projects"
        os.makedirs(project_folder, exist_ok=True)
        filename = f"{project_folder}/content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump({"content": content, "timeline": timeline}, f)
        return filename

# Format Content Tool
class FormatContentTool(BaseTool):
    name: str = "Format Content Tool"
    description: str = "Formats content for publishing and calendar display."

    def _run(self, ideas: List[str], content_type: str, timeline: str, outlines: List[Dict[str, Any]], drafts: List[str]) -> Dict[str, Any]:
        if not ideas or not isinstance(ideas, list):
            ideas = ["Default idea"]
        if not content_type or content_type not in ["Blog", "Social Post"]:
            content_type = "Social Post"
        if not timeline or not isinstance(timeline, str):
            timeline = str(datetime.now().date())
        if not outlines or not isinstance(outlines, list):
            outlines = [{"H1": "Default title", "H2": [{"title": "Default section", "content": "Default content"}]}]
        if not drafts or not isinstance(drafts, list):
            drafts = ["Default draft"]

        formatted = {
            "type": content_type,
            "title": outlines[0]["H1"] if outlines else ideas[0].split('.')[0],
            "ideas": ideas,
            "outlines": outlines,
            "drafts": drafts,
            "timeline": timeline,
            "calendar_entry": {
                "title": outlines[0]["H1"] if outlines else "Untitled",
                "date": timeline,
                "type": content_type,
                "draft": drafts[0] if drafts else "No draft"
            }
        }
        return formatted