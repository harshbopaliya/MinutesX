# MinutesX: AI-Powered Meeting Notes for Google Meet

> **Intelligent meeting assistant that listens to Google Meet calls, transcribes conversations in real-time, generates summaries, extracts action items, and creates shareable captions — all powered by Google Gemini 2.5 Flash.**

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 What MinutesX Does

MinutesX connects to your Google Meet sessions and automatically:

- 🎧 **Captures meeting audio** in real-time (system audio loopback)
- 📝 **Live transcription** with English captions displayed as people speak
- 📋 **Smart summarization** - executive summary, key points, and decisions
- ✅ **Action item extraction** with owners and priorities
- 💬 **Shareable captions** for LinkedIn, Slack, email, and newsletters
- 🧠 **Meeting memory** - remembers context from past meetings

## 🚀 Quick Start (2 Minutes)

### 1. Install Dependencies

```bash
# Clone and setup
git clone https://github.com/harshbopaliya/MinutesX.git
cd MinutesX

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Install packages
pip install google-generativeai sounddevice numpy python-dotenv rich click
```

### 2. Configure API Key

```bash
# Create .env file
copy .env.template .env

# Edit .env and add your Gemini API key
# Get your key at: https://aistudio.google.com/apikey
```

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run the Demo

```bash
# Run with sample transcript (no audio required)
python demo.py

# Or listen to a real Google Meet call
python demo.py  # Select your audio device when prompted
```

## 📺 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                     Google Meet Call                         │
│                   (Audio playing on PC)                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              System Audio Capture (Loopback)                 │
│         Captures what you hear through speakers              │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Gemini 2.5 Flash - Live Transcription             │
│               Real-time speech-to-text                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
     ┌───────────┐     ┌───────────┐     ┌───────────┐
     │  Caption  │     │  Summary  │     │  Action   │
     │  Agent    │     │  Agent    │     │  Agent    │
     └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Meeting Notes                            │
│  • Live Captions • Summary • Key Points • Action Items      │
└─────────────────────────────────────────────────────────────┘
```

## 🎬 Demo Output Example

When you run MinutesX on a meeting, you get:

### Live Captions (Real-time)
```
>>> Sarah: Let's discuss our Q1 roadmap priorities
>>> John: Mobile performance issues are critical
>>> Mike: I can fix the Android crashes by end of January
```

### Meeting Summary
```
📝 Caption: Q1 planning meeting focused on mobile app fixes and onboarding redesign

📋 Executive Summary:
The team prioritized Android stability fixes, onboarding flow redesign, and 
technical debt reduction for Q1. Key decisions include allocating 20% of sprint 
capacity to code quality and launching a public API beta by end of quarter.

📌 Key Points:
• Android crash rate at 2.5% needs immediate attention
• 40% user drop-off during onboarding at step 3
• Dashboard will be simplified to show only 5 key metrics
• Public API beta planned for late March

✅ Action Items:
🔴 Fix Android stability issues - Mike (Due: Jan 15)
🟡 Deliver onboarding mockups - Lisa (Due: Jan 10)
🟡 Create funnel analysis report - David (Due: Thursday)
🟢 Set up project board - David (Due: Today)
```

## 📁 Project Structure

```
MinutesX/
├── demo.py                 # 🚀 Main demo - run this!
├── demo_transcript.txt     # Sample meeting transcript
├── main.py                 # Full CLI application
├── config.py               # Configuration management
├── .env.template           # Environment template
├── requirements.txt        # Python dependencies
│
├── agents/                 # AI Agents (Gemini-powered)
│   ├── orchestrator_agent.py   # Main coordinator
│   ├── summary_agent.py        # Meeting summarization
│   ├── caption_agent.py        # Caption generation
│   ├── action_agent.py         # Action item extraction
│   ├── classifier_agent.py     # Meeting type classification
│   └── reviewer_agent.py       # Quality review
│
├── tools/                  # Audio & Transcription
│   ├── audio_capture.py        # System audio capture
│   ├── live_transcription.py   # Gemini transcription
│   └── meet_transcript_tool.py # Google Meet integration
│
├── memory/                 # Meeting Memory
│   └── memory_bank.py          # Long-term storage
│
├── observability/          # Logging & Metrics
│   ├── logger.py
│   ├── metrics.py
│   └── tracer.py
│
└── meeting_notes/          # Output directory
    └── meeting_YYYYMMDD_*.md
```

## 🔧 Audio Setup for Google Meet

To capture Google Meet audio, you need to enable system audio capture:

### Windows (Stereo Mix)
1. Right-click speaker icon → Sound Settings
2. Sound Control Panel → Recording tab
3. Right-click → Show Disabled Devices
4. Enable "Stereo Mix" or "What U Hear"

### Alternative: Virtual Audio Cable
- Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)
- Set it as your default playback device
- Select it as input in MinutesX

### Testing Audio
```bash
python demo.py
# Look for "LOOPBACK" devices in the list
# Select the loopback device number
```

## 🤖 Powered by Gemini 2.5 Flash

MinutesX uses Google's latest Gemini 2.5 Flash model for:

| Feature | How Gemini Helps |
|---------|-----------------|
| **Transcription** | Multimodal audio-to-text conversion |
| **Summarization** | Understanding context and extracting key points |
| **Action Items** | Identifying tasks, owners, and deadlines |
| **Captions** | Creating concise, shareable summaries |
| **Classification** | Categorizing meeting types automatically |

## 📊 Output Formats

MinutesX saves meeting notes in multiple formats:

- **Markdown** (.md) - Formatted notes with sections
- **JSON** (.json) - Structured data for integrations
- **Plain Text** (.txt) - Raw transcript

Output files are saved to `./meeting_notes/` directory.

## 🛠 Configuration

Edit `.env` to customize:

```env
# Required
GOOGLE_API_KEY=your_key_here

# Optional - Model settings
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_TEMPERATURE=0.3

# Optional - Output settings  
LOG_LEVEL=INFO
```

## 📋 Commands

```bash
# Run interactive demo
python demo.py

# Process existing transcript file
python main.py process ./my_transcript.txt

# Test audio devices
python main.py test-audio

# Show configuration
python main.py config-info
```

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**Harsh Bopaliya** - [@harshbopaliya](https://github.com/harshbopaliya)

---

*Built with ❤️ using Google Gemini 2.5 Flash*
