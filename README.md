# MinutesX: Intelligent Multi-Agent Meeting Notes

> **Tagline:** *AI-powered multi-agent system that connects to Google Meet, captures meeting content, generates intelligent summaries, and auto-creates captions & action items in real-time.*

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![ADK](https://img.shields.io/badge/Google%20ADK-1.0-green.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

MinutesX is an intelligent multi-agent system built with **Google's Agent Development Kit (ADK)** and powered by **Gemini 2.5 Flash**. It seamlessly connects to Google Meet sessions to:

- 📝 **Capture meeting transcripts** in real-time
- 📋 **Generate multi-level summaries** (one-liner, executive, detailed)
- ✅ **Extract action items** with owners and due dates
- 💬 **Create shareable captions** for social/business use
- 🏷️ **Classify meeting types** (Sales, Product, Legal, etc.)
- 🧠 **Remember context** across meetings for continuity

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Google Meet Integration                      │
│                    (Meet API / Transcript Tool)                  │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MeetingOrchestratorAgent                      │
│              (Root Agent - Session Manager - ADK)                │
│                      Gemini 2.5 Flash                            │
└────────┬──────────┬──────────┬──────────┬──────────┬────────────┘
         │          │          │          │          │
         ▼          ▼          ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │Summary │ │Action  │ │Caption │ │Classify│ │Memory  │
    │ Agent  │ │ Agent  │ │ Agent  │ │ Agent  │ │ Agent  │
    │(Parallel)│(Parallel)│(Parallel)│(Parallel)│(Sequential)│
    └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘
         │          │          │          │          │
         └──────────┴──────────┴──────────┴──────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ReviewerAgent (Sequential)                  │
│              Merges, Refines, Quality Checks                     │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Memory Bank   │    │  Task Publisher │    │   A2A Protocol  │
│ (Long-term Store)│    │ (Slack/Jira)   │    │  (Agent Comm)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## ✨ Features Demonstrated

### Core Concepts (Required 3+)

| # | Concept | Implementation |
|---|---------|---------------|
| 1 | **Multi-Agent System** | Orchestrator + 5 parallel agents + 1 sequential reviewer |
| 2 | **LLM-Powered Agents** | All agents powered by Gemini 2.5 Flash via ADK |
| 3 | **Parallel Agents** | Summary, Action, Caption, Classify run concurrently |
| 4 | **Sequential Agents** | Reviewer agent processes after parallel completion |
| 5 | **Custom Tools** | MeetTranscriptTool, SpeakerIdentifierTool |
| 6 | **Built-in Tools** | Google Search for context enrichment |
| 7 | **Sessions & State** | InMemorySessionService for meeting state |
| 8 | **Long-term Memory** | MemoryBank for cross-meeting context |
| 9 | **Context Engineering** | Context compaction for long transcripts |
| 10 | **Observability** | Structured logging, tracing, metrics |
| 11 | **A2A Protocol** | Agent-to-agent message passing |
| 12 | **Agent Evaluation** | ROUGE/BERTScore + human eval framework |
| 13 | **Agent Deployment** | Vertex AI Agent Engine configs |

## 📁 Project Structure

```
MinutesX/
├── agents/
│   ├── __init__.py
│   ├── orchestrator_agent.py    # Main ADK orchestrator
│   ├── summary_agent.py         # Multi-level summarization
│   ├── action_agent.py          # Action item extraction
│   ├── caption_agent.py         # Caption generation
│   ├── classifier_agent.py      # Meeting classification
│   ├── memory_agent.py          # Memory management
│   └── reviewer_agent.py        # Quality review & merge
├── tools/
│   ├── __init__.py
│   ├── meet_transcript_tool.py  # Google Meet integration
│   ├── speaker_identifier.py    # Speaker diarization
│   ├── task_publisher.py        # Slack/Jira integration
│   └── search_tool.py           # Google Search tool
├── memory/
│   ├── __init__.py
│   ├── memory_bank.py           # Long-term vector store
│   └── context_compactor.py     # Context compaction
├── session/
│   ├── __init__.py
│   └── session_service.py       # Session management
├── a2a/
│   ├── __init__.py
│   ├── protocol.py              # A2A message protocol
│   └── message_bus.py           # Message routing
├── observability/
│   ├── __init__.py
│   ├── logger.py                # Structured logging
│   ├── tracer.py                # OpenTelemetry tracing
│   └── metrics.py               # Prometheus metrics
├── evaluation/
│   ├── __init__.py
│   ├── rouge_eval.py            # ROUGE scoring
│   ├── bert_eval.py             # BERTScore evaluation
│   └── human_eval.py            # Human evaluation framework
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── agent_engine_config.yaml # Vertex AI config
│   └── cloudbuild.yaml
├── samples/
│   ├── meeting_transcript_1.txt
│   ├── meeting_transcript_2.txt
│   └── expected_outputs/
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   └── test_tools.py
├── app.py                       # FastAPI application
├── main.py                      # Entry point
├── config.py                    # Configuration
├── requirements.txt
├── .env.template
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google Cloud account with Meet API access
- Gemini API key (from Google AI Studio)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MinutesX.git
cd MinutesX

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.template .env  # or 'copy' on Windows

# Edit .env with your API keys
```

### Configuration

Edit `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_CLOUD_PROJECT=your_project_id
```

### Run the Demo

```bash
# Process a sample transcript
python main.py --transcript samples/meeting_transcript_1.txt

# Start the API server
python app.py
```

### API Usage

```bash
# Process a meeting
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"meeting_id": "meet123", "transcript": "..."}'

# Search past meetings
curl "http://localhost:8000/api/v1/search?query=pricing+discussion"
```

## 📖 Usage Examples

### Python SDK

```python
from minutesx import MinutesXClient

client = MinutesXClient()

# Process a meeting
result = client.process_meeting(
    meeting_id="team-standup-2025-12-01",
    transcript="[Meeting transcript text...]"
)

print(result.caption)           # One-line summary
print(result.executive_summary) # 3-sentence summary
print(result.action_items)      # List of action items
print(result.category)          # Meeting classification
```

### Connect to Live Google Meet

```python
from minutesx import MinutesXClient

client = MinutesXClient()

# Connect to a Google Meet session
session = client.connect_meet(
    meeting_code="abc-defg-hij",
    credentials_path="service_account.json"
)

# Start real-time processing
session.start_processing(
    on_summary=lambda s: print(f"Summary: {s}"),
    on_action=lambda a: print(f"Action: {a}")
)
```

## 🧪 Evaluation

```bash
# Run ROUGE evaluation
python -m evaluation.rouge_eval --predictions outputs/ --references samples/expected_outputs/

# Run BERTScore evaluation
python -m evaluation.bert_eval --predictions outputs/ --references samples/expected_outputs/

# Generate evaluation report
python -m evaluation.generate_report
```

## 🚢 Deployment

### Docker

```bash
docker build -t minutesx .
docker run -p 8000:8000 --env-file .env minutesx
```

### Vertex AI Agent Engine

```bash
gcloud ai agent-engines deploy minutesx \
  --config=deploy/agent_engine_config.yaml \
  --project=your-project-id
```

## 📚 Technology Stack

- **Agent Framework**: Google ADK (Agent Development Kit)
- **LLM**: Gemini 2.5 Flash
- **API**: FastAPI
- **Memory**: FAISS / ChromaDB
- **Observability**: OpenTelemetry, Prometheus
- **Deployment**: Vertex AI Agent Engine, Cloud Run

## 🔗 Resources

- [ADK Documentation](https://google.github.io/adk-docs)
- [ADK Python SDK](https://github.com/google/adk-python)
- [A2A Protocol](https://github.com/google/a2a)
- [Vertex AI Agent Engine](https://cloud.google.com/vertex-ai/docs/agents)
- [Google AI Studio](https://aistudio.google.com)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👥 Authors

- Harsh Bopaliya — @yourhandle

---

*Built for the Google AI Agents Hackathon 2025*
