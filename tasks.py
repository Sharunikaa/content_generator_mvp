from crewai import Task
from agents import (
    idea_intake_agent, content_strategy_agent, idea_generation_agent,
    outline_creation_agent, content_writing_agent, review_approval_agent,
    save_schedule_agent
)

# Tasks
def create_tasks(state):
    idea_intake_task = Task(
        description="Validate user inputs for content creation",
        expected_output="Structured JSON with validated inputs",
        agent=idea_intake_agent,
        inputs={"form_data": state.inputs}
    )

    content_strategy_task = Task(
        description="Create a content calendar based on validated inputs",
        expected_output="JSON content calendar",
        agent=content_strategy_agent,
        inputs={"validated_inputs": state.validated_inputs}
    )

    idea_generation_task = Task(
        description="Generate 3–5 content ideas based on inputs",
        expected_output="List of content ideas",
        agent=idea_generation_agent,
        inputs={"validated_inputs": state.validated_inputs}
    )

    outline_creation_task = Task(
        description="Create at least 2 detailed outlines with SEO keywords",
        expected_output="List of outlines with H1, H2, H3, and keywords",
        agent=outline_creation_agent,
        inputs={"ideas": state.ideas}
    )

    content_writing_task = Task(
        description="Generate a draft based on the selected outline",
        expected_output="Full content draft",
        agent=content_writing_agent,
        inputs={"outline": state.selected_outline, "tone": state.validated_inputs.get("tone"), "content_type": state.validated_inputs.get("content_type")}
    )

    review_approval_task = Task(
        description="Check grammar and tone, provide corrected content",
        expected_output="Reviewed draft with corrections",
        agent=review_approval_agent,
        inputs={"draft": state.draft}
    )

    save_schedule_task = Task(
        description="Save content and update calendar",
        expected_output="Saved file path and updated calendar",
        agent=save_schedule_agent,
        inputs={"formatted_content": state.formatted_content, "timeline": state.validated_inputs.get("timeline")}
    )

    return [
        idea_intake_task,
        content_strategy_task,
        idea_generation_task,
        outline_creation_task,
        content_writing_task,
        review_approval_task,
        save_schedule_task
    ]