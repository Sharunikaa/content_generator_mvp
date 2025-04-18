import os
import streamlit as st
try:
    from crewai.tools import BaseTool
except ImportError:
    st.error("CrewAI is not installed. Please run 'pip install crewai==0.114.0'.")
    st.stop()
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
from datetime import datetime
import google.generativeai as genai
import requests
import pandas as pd
import glob

# Grammar Check with LanguageTool (unchanged)
def check_grammar_tone(content: str) -> Dict[str, Any]:
    try:
        if not isinstance(content, str) or not content.strip():
            return {
                "content": str(content),
                "suggestions": [],
                "corrected": str(content)
            }
        suggestions = []
        corrected_content = content
        try:
            response = requests.post(
                "https://api.languagetool.org/v2/check",
                data={"text": content, "language": "en-US"},
                timeout=5
            )
            response.raise_for_status()
            result = response.json()
            matches = result.get("matches", [])
            suggestions = [
                {
                    "message": m["message"],
                    "replacements": [r["value"] for r in m.get("replacements", [])],
                    "context": m["context"]["text"]
                }
                for m in matches
            ]
            for match in sorted(matches, key=lambda x: x["offset"], reverse=True):
                replacements = match.get("replacements", [])
                if replacements:
                    start = match["offset"]
                    end = start + match["length"]
                    corrected_content = corrected_content[:start] + replacements[0]["value"] + corrected_content[end:]
        except Exception as e:
            st.warning(f"LanguageTool API error: {str(e)}")
            suggestions.append({"message": "Grammar check failed", "replacements": [], "context": content[:50]})
        return {
            "content": content,
            "suggestions": suggestions,
            "corrected": corrected_content
        }
    except Exception as e:
        st.warning(f"Error checking grammar: {str(e)}")
        return {
            "content": content if isinstance(content, str) else str(content),
            "suggestions": [],
            "corrected": content if isinstance(content, str) else str(content)
        }

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env file.")
    st.stop()

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {str(e)}")
    st.stop()

# Custom Tools
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

class GenerateContentIdeasTool(BaseTool):
    name: str = "Generate Content Ideas Tool"
    description: str = "Generates 3–5 content ideas using Gemini API based on user input topic."

    def _run(self, inputs: Dict[str, Any]) -> List[str]:
        if not inputs or not inputs.get("topic"):
            return ["No valid topic provided"]
        prompt = f"""
        Generate 3–5 ideas for a {inputs.get('tone', 'formal').lower()} social media post about {inputs.get('topic')} for {inputs.get('target_audience', 'general audience')}.
        Each idea should be a single sentence, relevant to the topic, and appropriate for all audiences.
        """
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(
                    "gemini-1.5-flash",
                    safety_settings={
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                    }
                )
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.7, "max_output_tokens": 150}
                )
                if not hasattr(response, "text") or not response.text:
                    st.warning(f"Empty response from Gemini (attempt {attempt+1})")
                    continue
                result = response.text.strip().split("\n")
                ideas = [idea.strip() for idea in result if idea.strip() and not idea.startswith(("-", "*", "#"))]
                return ideas[:5] if ideas else ["No valid ideas generated"]
            except Exception as e:
                st.warning(f"Error generating ideas (attempt {attempt+1}): {str(e)}")
                continue
        topic = inputs.get('topic', 'general topic')
        return [
            f"Learn {topic} to develop essential skills for {inputs.get('target_audience', 'general audience')}.",
            f"Master {topic} to create innovative projects.",
            f"Use {topic} to advance your career opportunities."
        ]

class GenerateSEOKeywordsTool(BaseTool):
    name: str = "Generate SEO Keywords Tool"
    description: str = "Generates SEO keywords using Gemini API based on the topic."

    def _run(self, topic: str) -> List[Dict[str, Any]]:
        if not topic or not isinstance(topic, str):
            return [{"keyword": f"keyword_{i}", "source": "generated"} for i in range(1, 4)]
        
        prompt = f"""
        Generate 5–10 SEO keywords for content about '{topic}'.
        Each keyword should be a short phrase, relevant to the topic, and optimized for search engines.
        List one keyword per line.
        """
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(
                    "gemini-1.5-flash",
                    safety_settings={
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                    }
                )
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.7, "max_output_tokens": 150}
                )
                if not hasattr(response, "text") or not response.text:
                    st.warning(f"Empty response from Gemini (attempt {attempt+1})")
                    continue
                keywords = [
                    {"keyword": keyword.strip(), "source": "Gemini"}
                    for keyword in response.text.strip().split("\n")
                    if keyword.strip()
                ]
                return keywords[:10] if keywords else [{"keyword": f"{topic.lower()} basics", "source": "fallback"}]
            except Exception as e:
                st.warning(f"Error generating keywords (attempt {attempt+1}): {str(e)}")
                continue
        return [{"keyword": f"{topic.lower()} basics", "source": "fallback"}]

class CreateOutlineTool(BaseTool):
    name: str = "Create Outline Tool"
    description: str = "Creates a professionally formatted outline for a content idea using keywords."

    def _run(self, idea: str, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not idea or not isinstance(idea, str):
            idea = "Default idea"
        if not keywords or not isinstance(keywords, list):
            keywords = [{"keyword": "keyword_1", "source": "generated"}]
        
        outline = {
            "title": idea.strip(),
            "sections": []
        }
        
        for i, kw in enumerate(keywords[:3]):
            keyword = kw.get("keyword", f"Section {i+1}")
            outline["sections"].append({
                "title": f"Introduction to {keyword.title()}",
                "content": f"Explore {keyword} and its relevance to {idea.lower()}."
            })
        
        if outline["sections"]:
            for i in range(2):
                outline["sections"].append({
                    "title": f"Key Aspect {i+1}",
                    "content": f"Detailed insights into {keywords[0].get('keyword', 'aspect')}."
                })
        
        return outline

class GenerateContentDraftTool(BaseTool):
    name: str = "Generate Content Draft Tool"
    description: str = "Generates a 200–250-word social media post based on an outline, topic, and SEO keywords."

    def _run(self, outline: Dict[str, Any], tone: str, topic: str) -> str:
        if not outline or not isinstance(outline, dict):
            return "Error: Invalid outline"
        if not tone or not isinstance(tone, str):
            tone = "formal"
        if not topic or not isinstance(topic, str):
            topic = "general topic"
        
        # Extract keywords from outline or fallback
        keywords = [section.get("title", "").lower() for section in outline.get("sections", [])]
        keywords_str = ", ".join(keywords[:3]) if keywords else topic.lower()
        
        prompt = f"""
        Write a {tone.lower()} social media post for students titled '{outline.get('title', 'Default Title')}'.
        Focus on the topic '{topic}' and incorporate these SEO keywords: {keywords_str}.
        Keep it professional, 200–250 words, and relevant to students' learning and career goals.
        Avoid hashtags, slang, and casual phrases.
        """
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(
                    "gemini-1.5-flash",
                    safety_settings={
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                    }
                )
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.8, "max_output_tokens": 300}
                )
                if not hasattr(response, "text") or not response.text:
                    st.warning(f"Empty response from Gemini (attempt {attempt+1})")
                    continue
                draft = response.text.strip()
                word_count = len(draft.split())
                if word_count < 200:
                    draft += f" Engaging with {topic} equips students with versatile skills, fostering opportunities for innovation and professional growth." * ((200 - word_count) // 20)
                elif word_count > 250:
                    draft = " ".join(draft.split()[:250])
                return draft
            except Exception as e:
                st.warning(f"Error generating draft (attempt {attempt+1}): {str(e)}")
                continue
        return f"{outline.get('title')} for students. Engaging with {topic} fosters essential skills for career advancement. " * 10

class CheckGrammarToneTool(BaseTool):
    name: str = "Check Grammar Tone Tool"
    description: str = "Checks grammar and spelling of content using LanguageTool."

    def _run(self, content: str) -> Dict[str, Any]:
        if not content or not isinstance(content, str):
            return {
                "content": "Empty content",
                "suggestions": [],
                "corrected": "Empty content"
            }
        return check_grammar_tone(content)

class SaveContentTool(BaseTool):
    name: str = "Save Content Tool"
    description: str = "Saves content to a project folder with topic, sub_heading, and timeline."

    def _run(self, content: Dict[str, Any], timeline: str, topic: str, sub_heading: str) -> str:
        if not content or not isinstance(content, dict):
            content = {"error": "Invalid content"}
        if not timeline or not isinstance(timeline, str):
            timeline = str(datetime.now().date())
        if not topic or not isinstance(topic, str) or topic.strip().lower() == "unknown":
            topic = "Default Topic"
        if not sub_heading or not isinstance(sub_heading, str):
            sub_heading = "Default Sub-Heading"
        project_folder = "content_projects"
        os.makedirs(project_folder, exist_ok=True)
        filename = f"{project_folder}/content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        content["topic"] = topic.strip()
        content["sub_heading"] = sub_heading.strip()
        with open(filename, 'w') as f:
            json.dump({"content": content, "timeline": timeline}, f)
        return filename

# Instantiate Tools
validate_input_tool = ValidateInputTool()
generate_content_ideas_tool = GenerateContentIdeasTool()
generate_seo_keywords_tool = GenerateSEOKeywordsTool()
create_outline_tool = CreateOutlineTool()
generate_content_draft_tool = GenerateContentDraftTool()
check_grammar_tone_tool = CheckGrammarToneTool()
save_content_tool = SaveContentTool()

# Load Calendar Data
def load_calendar_data() -> List[Dict[str, str]]:
    project_folder = "content_projects"
    calendar_data = []
    try:
        json_files = glob.glob(f"{project_folder}/*.json")
        for file in json_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    topic = data.get("content", {}).get("topic", "Unknown").strip()
                    sub_heading = data.get("content", {}).get("sub_heading", "N/A").strip()
                    content_type = data.get("content", {}).get("type", "Unknown").strip()
                    timeline = data.get("timeline", "Unknown").strip()
                    if not topic or topic.lower() == "unknown" or not timeline or timeline.lower() == "unknown":
                        continue
                    topic = topic.replace("\n", " ").replace("\r", " ")
                    sub_heading = sub_heading.replace("\n", " ").replace("\r", " ")
                    content_type = content_type.replace("\n", " ").replace("\r", " ")
                    calendar_data.append({
                        "Topic": topic,
                        "Sub-Heading": sub_heading,
                        "Content Type": content_type,
                        "Scheduled Date": timeline
                    })
            except Exception as e:
                st.warning(f"Error reading {file}: {str(e)}")
        seen = set()
        unique_data = []
        for item in calendar_data:
            key = (item["Topic"], item["Sub-Heading"], item["Content Type"], item["Scheduled Date"])
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        unique_data.sort(key=lambda x: x["Scheduled Date"])
        return unique_data
    except Exception as e:
        st.warning(f"Error loading calendar data: {str(e)}")
        return []

# Workflow State
class WorkflowState:
    def __init__(self):
        self.inputs = {}
        self.validated_inputs = {}
        self.ideas = []
        self.selected_idea = None
        self.keywords = []
        self.outline = {}
        self.draft = ""
        self.reviewed_draft = {}
        self.saved_file = ""
        self.error = None
        self.current_step = 1

# Workflow Steps
def run_step_1(state, inputs):
    try:
        state.inputs = inputs
        state.validated_inputs = validate_input_tool._run(inputs)
        state.current_step = 2
        return True
    except Exception as e:
        state.error = f"Step 1 Error: {str(e)}"
        return False

def run_step_2(state):
    try:
        state.ideas = generate_content_ideas_tool._run(state.validated_inputs)
        if not state.ideas or not isinstance(state.ideas, list):
            state.ideas = ["Failed to generate ideas"]
        state.current_step = 3
        return True
    except Exception as e:
        state.error = f"Step 2 Error: {str(e)}"
        return False

def run_step_3(state, selected_idea):
    try:
        if not selected_idea or selected_idea not in state.ideas:
            state.error = "No valid idea selected"
            return False
        state.selected_idea = selected_idea
        state.keywords = generate_seo_keywords_tool._run(state.validated_inputs.get("topic"))
        state.outline = create_outline_tool._run(selected_idea, state.keywords)
        state.current_step = 4
        return True
    except Exception as e:
        state.error = f"Step 3 Error: {str(e)}"
        return False

def run_step_4(state):
    try:
        state.draft = generate_content_draft_tool._run(
            state.outline,
            state.validated_inputs.get("tone", "Formal"),
            state.validated_inputs.get("topic", "general topic")
        )
        state.current_step = 5
        return True
    except Exception as e:
        state.error = f"Step 4 Error: {str(e)}"
        return False

def run_step_5(state):
    try:
        state.reviewed_draft = check_grammar_tone_tool._run(state.draft)
        state.current_step = 6
        return True
    except Exception as e:
        state.error = f"Step 5 Error: {str(e)}"
        return False

def run_step_6(state):
    try:
        formatted_content = {
            "type": state.validated_inputs.get("content_type", "Social Post"),
            "posts": [{"post": state.reviewed_draft.get("corrected", state.draft), "schedule": state.validated_inputs.get("timeline")}],
            "ideas": state.ideas,
            "keywords": state.keywords,
            "outline": state.outline,
            "draft": state.reviewed_draft.get("corrected", state.draft),
            "timeline": state.validated_inputs.get("timeline")
        }
        state.saved_file = save_content_tool._run(
            formatted_content,
            state.validated_inputs.get("timeline"),
            state.validated_inputs.get("topic"),
            state.selected_idea
        )
        state.current_step = 7
        return True
    except Exception as e:
        state.error = f"Step 6 Error: {str(e)}"
        return False

# Streamlit Frontend
def main():
    st.title("Content Generation MVP")

    # Sticky Calendar CSS
    st.markdown(
        """
        <style>
        .sticky-calendar {
            position: sticky;
            top: 0;
            z-index: 100;
            background-color: white;
            padding: 5px 0;
            border-bottom: 2px solid #ddd;
        }
        .stDataFrame {
            width: 100%;
            font-size: 14px;
        }
        .stDataFrame table {
            border-collapse: collapse;
        }
        .stDataFrame th, .stDataFrame td {
            border: 1px solid #ddd;
            padding: 6px 8px;
            text-align: left;
        }
        .stDataFrame th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        .stDataFrame tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .stDataFrame tr:hover {
            background-color: #f1f1f1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display Calendar
    calendar_data = load_calendar_data()
    if calendar_data:
        st.markdown('<div class="sticky-calendar">', unsafe_allow_html=True)
        df = pd.DataFrame(calendar_data)
        st.dataframe(
            df,
            column_config={
                "Topic": st.column_config.TextColumn(width=120),
                "Sub-Heading": st.column_config.TextColumn(width=400),
                "Content Type": st.column_config.TextColumn(width=100),
                "Scheduled Date": st.column_config.TextColumn(width=80)
            },
            hide_index=True,
            height=150 if len(calendar_data) > 3 else None,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = WorkflowState()
        st.session_state.selected_idea = None
        st.session_state.edited_draft = None

    state = st.session_state.workflow_state

    # Step 1: Input Form
    st.header("Step 1: Enter Content Details")
    with st.form("content_form"):
        topic = st.text_input("Topic", value="Python")
        target_audience = st.text_input("Target Audience", value="Students")
        tone = st.selectbox("Tone", ["Formal", "Casual", "Inspirational", "Humorous"], index=0)
        anecdotes = st.text_area("Anecdotes", value="story", help="Optional context")
        content_type = st.selectbox("Content Type", ["Blog", "Social Post"], index=1)
        num_pieces = st.slider("Number of Content Pieces", 1, 5, 1)
        timeline = st.date_input("Publishing Timeline", value=datetime(2025, 4, 23))
        submit = st.form_submit_button("Generate Ideas")

    if submit:
        inputs = {
            "topic": topic,
            "target_audience": target_audience,
            "tone": tone,
            "anecdotes": anecdotes,
            "content_type": content_type,
            "num_pieces": num_pieces,
            "timeline": str(timeline)
        }
        with st.spinner("Validating inputs and generating ideas..."):
            if run_step_1(state, inputs) and run_step_2(state):
                st.session_state.selected_idea = None
                st.session_state.edited_draft = None
                st.session_state.workflow_state = state
            else:
                st.error(state.error)

    # Step 2: Select Idea
    if state.current_step >= 2 and state.ideas:
        st.header("Step 2: Select a Content Idea")
        selected_idea = st.selectbox("Choose an idea", state.ideas, key="idea_selector")
        if st.button("Generate Content for Selected Idea"):
            with st.spinner("Generating keywords, creating outline, and drafting..."):
                if run_step_3(state, selected_idea) and run_step_4(state) and run_step_5(state):
                    st.session_state.selected_idea = selected_idea
                    st.session_state.workflow_state = state
                else:
                    st.error(state.error)

    # Step 3: Display Keywords and Outline
    if state.current_step >= 3 and state.keywords and state.outline:
        st.header("Step 3: Review SEO Keywords and Outline")
        
        st.subheader("SEO Keywords")
        keyword_data = [{"Keyword": kw.get("keyword", "N/A"), "Source": kw.get("source", "N/A")} for kw in state.keywords]
        st.table(keyword_data)

        st.subheader("Content Outline")
        st.write(f"**Title**: {state.outline.get('title', 'N/A')}")
        for section in state.outline.get('sections', []):
            st.write(f"**{section.get('title', 'N/A')}**")
            st.write(f"- {section.get('content', 'N/A')}")

    # Step 4: Display and Edit Draft
    if state.current_step >= 4 and state.draft:
        st.header("Step 4: Review and Edit Draft")
        
        st.subheader("Generated Draft")
        st.write(state.draft)

        st.subheader("Grammar and Spelling Check")
        if st.button("Run Grammar Check"):
            with st.spinner("Checking grammar..."):
                state.reviewed_draft = check_grammar_tone_tool._run(state.draft)
                st.session_state.workflow_state = state

        if state.reviewed_draft:
            st.write("**Corrected Draft**:")
            st.write(state.reviewed_draft.get("corrected", state.draft))
            st.write("**Grammar and Spelling Suggestions**:")
            if state.reviewed_draft["suggestions"]:
                for suggestion in state.reviewed_draft["suggestions"]:
                    replacements = suggestion.get("replacements", [])
                    st.write(f"- {suggestion['message']} (in: '{suggestion['context']}'): {', '.join(replacements) if replacements else 'No replacements suggested'}")
            else:
                st.write("No grammar or spelling issues found.")

        st.subheader("Edit Draft")
        edited_draft = st.text_area(
            "Modify the draft",
            value=state.reviewed_draft.get("corrected", state.draft),
            height=200
        )
        if st.checkbox("Check grammar in real-time"):
            with st.spinner("Analyzing..."):
                live_review = check_grammar_tone_tool._run(edited_draft)
                if live_review["suggestions"]:
                    st.write("**Live Suggestions**:")
                    for suggestion in live_review["suggestions"]:
                        replacements = suggestion.get("replacements", [])
                        st.write(f"- {suggestion['message']} (in: '{suggestion['context']}'): {', '.join(replacements) if replacements else 'No replacements'}")
                else:
                    st.write("No live grammar issues detected.")

        if st.button("Save Edited Draft"):
            state.draft = edited_draft
            state.reviewed_draft = check_grammar_tone_tool._run(edited_draft)
            st.session_state.edited_draft = edited_draft
            st.success("Draft updated!")
            st.session_state.workflow_state = state

    # Step 5: Save and Schedule
    if state.current_step >= 5:
        st.header("Step 5: Save and Schedule")
        if st.button("Finalize and Save Content"):
            with st.spinner("Saving content..."):
                if run_step_6(state):
                    st.success(f"Content saved to {state.saved_file} and scheduled for {state.validated_inputs.get('timeline')}")
                    st.session_state.workflow_state = state
                else:
                    st.error(state.error)

if __name__ == "__main__":
    main()