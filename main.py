import streamlit as st
from crewai import Crew
from agents import (
    idea_intake_agent, content_strategy_agent, idea_generation_agent,
    outline_creation_agent, content_writing_agent, review_approval_agent,
    save_schedule_agent
)
from tools import (
    ValidateInputTool, GenerateContentIdeasTool, GenerateSEOKeywordsTool,
    CreateOutlineTool, GenerateContentDraftTool, CheckGrammarToneTool,
    SaveContentTool, FormatContentTool, SearchKeywordTool
)
from tasks import create_tasks
from datetime import datetime, timedelta
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Workflow State
class WorkflowState:
    def __init__(self):
        self.inputs = {}
        self.validated_inputs = {}
        self.ideas = []
        self.outlines = []
        self.selected_outline = None
        self.draft = ""
        self.reviewed_draft = {}
        self.formatted_content = {}
        self.saved_file = ""
        self.calendar = []
        self.error = None
        self.current_step = 1

# Run Workflow
def run_workflow(state, step, inputs=None, selected_outline=None, edit_content=None):
    if step == 1 and inputs:
        # Step 1: Idea Intake
        try:
            state.inputs = inputs
            state.validated_inputs = ValidateInputTool()._run(inputs)
            state.current_step = 2
        except Exception as e:
            state.error = f"Step 1 Error: {str(e)}"
            return

    if step == 2:
        # Step 2: Generate Ideas
        try:
            state.ideas = GenerateContentIdeasTool()._run(state.validated_inputs)
            if not state.ideas or "Error" in state.ideas[0]:
                state.ideas = ["Fallback idea: Discuss the topic generally."]
            state.current_step = 3
        except Exception as e:
            state.error = f"Step 2 Error: {str(e)}"
            return

    if step == 3:
        # Step 3: Create Outlines
        try:
            state.outlines = []
            for idea in state.ideas[:max(2, state.validated_inputs.get("num_pieces", 1))]:
                if "Error" in idea:
                    continue
                keywords = GenerateSEOKeywordsTool()._run(idea)
                outline = CreateOutlineTool()._run(idea, keywords)
                state.outlines.append(outline)
            if not state.outlines:
                state.outlines = [{
                    "H1": "Fallback Outline",
                    "H2": [{"title": "Section 1", "content": "Discuss topic"}, {"title": "Section 2", "content": "Explore details"}],
                    "H3": [{"title": "Subsection", "content": "Details"}]
                }]
            state.current_step = 4
        except Exception as e:
            state.error = f"Step 3 Error: {str(e)}"
            return

    if step == 4 and selected_outline:
        # Step 4: Generate Draft
        try:
            state.selected_outline = selected_outline
            state.draft = GenerateContentDraftTool()._run(
                selected_outline,
                state.validated_inputs.get("tone", "Casual"),
                state.validated_inputs.get("content_type", "Social Post")
            )
            state.current_step = 5
        except Exception as e:
            state.error = f"Step 4 Error: {str(e)}"
            return

    if step == 5:
        # Step 5: Review Draft
        try:
            state.reviewed_draft = CheckGrammarToneTool()._run(state.draft)
            state.current_step = 6
        except Exception as e:
            state.error = f"Step 5 Error: {str(e)}"
            return

    if step == 6:
        # Step 6: Format and Save
        try:
            state.formatted_content = FormatContentTool()._run(
                state.ideas,
                state.validated_inputs.get("content_type", "Social Post"),
                state.validated_inputs.get("timeline", str(datetime.now().date())),
                state.outlines,
                [state.reviewed_draft.get("corrected", state.draft)]
            )
            state.saved_file = SaveContentTool()._run(
                state.formatted_content,
                state.validated_inputs.get("timeline")
            )
            # Update calendar
            state.calendar.append({
                "title": state.formatted_content["title"],
                "date": state.formatted_content["timeline"],
                "type": state.formatted_content["type"],
                "draft": state.reviewed_draft.get("corrected", state.draft),
                "file": state.saved_file
            })
            state.current_step = 7
        except Exception as e:
            state.error = f"Step 6 Error: {str(e)}"
            return

    if step == 7 and edit_content:
        # Step 7: Edit Content
        try:
            # Update draft and resave
            state.draft = edit_content["draft"]
            state.reviewed_draft = CheckGrammarToneTool()._run(state.draft)
            state.formatted_content["drafts"] = [state.reviewed_draft.get("corrected", state.draft)]
            state.saved_file = SaveContentTool()._run(
                state.formatted_content,
                state.validated_inputs.get("timeline")
            )
            # Update calendar entry
            for entry in state.calendar:
                if entry["file"] == state.saved_file:
                    entry["draft"] = state.reviewed_draft.get("corrected", state.draft)
        except Exception as e:
            state.error = f"Step 7 Error: {str(e)}"

# Streamlit Frontend
def main():
    st.title("Content Generation MVP")

    # Initialize session state
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = WorkflowState()

    state = st.session_state.workflow_state

    # Step 1: Idea Intake
    st.header("Step 1: Enter Content Details")
    with st.form("content_form"):
        topic = st.text_input("Topic", value="vue, node, react")
        target_audience = st.text_input("Target Audience", value="Students")
        tone = st.selectbox("Tone", ["Formal", "Casual", "Inspirational", "Humorous"], index=3)
        anecdotes = st.text_area("Anecdotes", value="story")
        content_type = st.selectbox("Content Type", ["Blog", "Social Post"], index=1)
        num_pieces = st.slider("Number of Content Pieces", 1, 5, 1)
        timeline = st.date_input("Publishing Timeline", value=datetime(2025, 4, 16))
        submit = st.form_submit_button("Generate Content")

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
        with st.spinner("Validating inputs..."):
            run_workflow(state, 1, inputs=inputs)
        if state.error:
            st.error(state.error)

    # Step 2: Display Ideas
    if state.current_step > 1 and state.ideas:
        st.header("Step 2: Generated Content Ideas")
        for i, idea in enumerate(state.ideas):
            st.write(f"{i+1}. {idea}")

    # Step 3: Display Outlines
    if state.current_step > 2 and state.outlines:
        st.header("Step 3: Content Outlines")
        for i, outline in enumerate(state.outlines):
            with st.expander(f"Outline {i+1}: {outline.get('H1', 'Untitled')}"):
                st.write(f"**H1**: {outline['H1']}")
                for h2 in outline.get("H2", []):
                    st.write(f"**H2**: {h2['title']}")
                    st.write(f"Content: {h2['content']}")
                for h3 in outline.get("H3", []):
                    st.write(f"**H3**: {h3['title']}")
                    st.write(f"Content: {h3['content']}")
                if st.button(f"Select Outline {i+1}", key=f"select_{i}"):
                    with st.spinner("Generating draft..."):
                        run_workflow(state, 4, selected_outline=outline)
                        if state.error:
                            st.error(state.error)

    # Step 4: Display Draft
    if state.current_step > 4 and state.draft:
        st.header("Step 4: Content Draft")
        st.write(state.draft)

        # SEO Keyword Search
        st.subheader("SEO Keyword Search")
        keyword = st.text_input("Enter a keyword to search")
        if st.button("Search Keyword"):
            result = SearchKeywordTool()._run(keyword)
            st.write(f"**Keyword**: {result['keyword']}")
            st.write(f"**Meaning**: {result['meaning']}")
            st.write(f"**Related Terms**: {', '.join(result['related'])}")

    # Step 5: Display Reviewed Draft
    if state.current_step > 5 and state.reviewed_draft:
        st.header("Step 5: Grammar & Tone Review")
        st.write("**Original Draft**:")
        st.write(state.reviewed_draft["content"])
        st.write("**Corrected Draft**:")
        st.write(state.reviewed_draft["corrected"])
        st.write("**Suggestions**:")
        for suggestion in state.reviewed_draft.get("suggestions", []):
            st.write(f"- {suggestion['message']}: {', '.join(suggestion['replacements'])}")
        st.write("**Tone Suggestion**:")
        st.write(state.reviewed_draft["tone_suggestion"])

    # Step 6: Display Calendar
    if state.current_step > 6 and state.calendar:
        st.header("Step 6: Content Calendar")
        calendar_data = [
            {
                "Title": entry["title"],
                "Date": entry["date"],
                "Type": entry["type"],
                "Draft Preview": entry["draft"][:50] + "..." if len(entry["draft"]) > 50 else entry["draft"],
                "File": entry["file"]
            }
            for entry in state.calendar
        ]
        st.table(calendar_data)

        # Edit Content
        st.subheader("Edit Scheduled Content")
        selected_file = st.selectbox("Select content to edit", [entry["file"] for entry in state.calendar])
        if selected_file:
            selected_entry = next(entry for entry in state.calendar if entry["file"] == selected_file)
            new_draft = st.text_area("Edit Draft", value=selected_entry["draft"])
            if st.button("Save Edited Content"):
                with st.spinner("Saving changes..."):
                    run_workflow(state, 7, edit_content={"draft": new_draft, "file": selected_file})
                    if state.error:
                        st.error(state.error)
                    else:
                        st.success("Content updated successfully!")

if __name__ == "__main__":
    main()