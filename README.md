# Content Generator MVP with Multi-Agent System

This project demonstrates a sophisticated content generation system using CrewAI's multi-agent architecture, powered by Google's Gemini AI. The system employs multiple specialized agents working together to generate, refine, and manage content efficiently.

## Multi-Agent Systems Overview

Multi-agent systems (MAS) are computational systems where multiple intelligent agents interact to solve problems that are beyond their individual capabilities. In this project:

- Agents work autonomously but collaboratively
- Each agent has specialized roles and responsibilities
- Communication and coordination happen through a well-defined workflow
- Tasks are distributed based on agent expertise
- System maintains coherence through structured data flow

### Benefits of Multi-Agent Architecture

1. **Specialization**: Each agent focuses on specific tasks they excel at
2. **Parallel Processing**: Multiple agents can work simultaneously
3. **Redundancy**: System remains functional even if some agents fail
4. **Scalability**: Easy to add new agents or modify existing ones
5. **Flexibility**: Agents can adapt to different content types and requirements

## CrewAI Components Implementation

### 1. Agents

Our system implements the following specialized agents using CrewAI:

```python
# Agent Structure
agents = {
    'idea_intake_agent': {
        'role': 'Idea Intake Agent',
        'goal': 'Validate and structure user inputs',
        'tools': ['ValidateInputTool']
    },
    'content_strategy_agent': {
        'role': 'Content Strategy Agent',
        'goal': 'Create content calendar',
        'tools': ['FormatContentTool']
    },
    'idea_generation_agent': {
        'role': 'Idea Generation Agent',
        'goal': 'Generate creative content ideas',
        'tools': ['GenerateContentIdeasTool']
    },
    # ... more agents
}
```

### 2. Tasks

Tasks are defined as discrete units of work assigned to agents:

- **Idea Intake**: Validates user inputs and structures data
- **Content Strategy**: Plans content calendar and distribution
- **Idea Generation**: Creates multiple content ideas
- **Outline Creation**: Develops detailed content outlines
- **Content Writing**: Generates full content drafts
- **Review & Approval**: Checks grammar and tone
- **Save & Schedule**: Manages content storage and scheduling

### 3. Tools

Custom tools that agents use to perform their tasks:

1. **ValidateInputTool**: Input validation and sanitization
2. **GenerateContentIdeasTool**: AI-powered idea generation
3. **GenerateSEOKeywordsTool**: Keyword research and optimization
4. **CreateOutlineTool**: Content structure creation
5. **GenerateContentDraftTool**: Draft generation
6. **CheckGrammarToneTool**: Grammar and tone checking
7. **SaveContentTool**: Content persistence
8. **FormatContentTool**: Content formatting and preparation

## Project Structure

```
content_generator_mvp/
├── agents.py           # Agent definitions and configurations
├── app.py             # Main Streamlit application
├── main.py            # Core workflow implementation
├── tasks.py           # Task definitions and management
├── tools.py           # Custom tool implementations
├── test_gemini.py     # Gemini API integration tests
├── requirements.txt    # Project dependencies
└── content_projects/  # Generated content storage
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/Sharunikaa/content_generator_mvp.git
cd content_generator_mvp
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
GEMINI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Key Features

1. **Intelligent Content Generation**
   - AI-powered idea generation
   - SEO optimization
   - Grammar and tone checking
   - Multiple content formats support

2. **Content Calendar Management**
   - Scheduled content planning
   - Content organization
   - Timeline management

3. **Quality Assurance**
   - Grammar checking
   - Tone analysis
   - Content validation

4. **User Interface**
   - Streamlit-based interactive UI
   - Real-time content preview
   - Easy content editing
   - Calendar view

## Technical Implementation

### Gemini AI Integration

The project uses Google's Gemini AI for natural language processing:

```python
class GeminiLLM:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
```

### Workflow State Management

Content generation workflow is managed through a state machine:

```python
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
```

## Dependencies

- streamlit>=1.31,<1.32
- crewai>=0.114,<0.115
- python-dotenv>=1.0,<1.1
- google-generativeai>=0.8,<0.9
- pyspellchecker>=0.8,<0.9
- protobuf==3.20.3
- google-api-core==2.19.2



## Acknowledgments

- CrewAI for the multi-agent framework
- Google Gemini AI for natural language processing
- Streamlit for the user interface framework
