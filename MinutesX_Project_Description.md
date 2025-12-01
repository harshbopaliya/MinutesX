# MinutesX: Project Description

MinutesX is a tool we built to solve a simple but common problem: it’s hard to stay focused in a meeting while also trying to take good notes. When we are on Google Meet—especially during fast discussions—it becomes almost impossible to type everything while listening at the same time. Someone asks a question, someone answers, decisions are made on the spot, and before you realize it, the meeting has moved on but you missed writing half of it. MinutesX removes this problem completely by listening to the meeting for you and turning the conversation into clean notes, summaries, and tasks automatically.

The goal of MinutesX is very straightforward: **you talk naturally in your meeting, and the AI writes the notes for you**. You don’t need to type, switch tabs, or remember everything from memory. The system listens to the audio coming from your Google Meet, converts it to text using Gemini, and then several agents work together to organize that information into something useful.

We built MinutesX using Google Gemini 2.5 Flash because it’s fast, accurate, and can handle real-time transcription well. Once the audio is captured, the tool can understand the context of the conversation, recognize important points, and rewrite everything in a structured format that’s easy to read. Instead of giving you a messy transcript full of filler words, MinutesX gives you something meaningful that you can share or store for later.

## Why we built MinutesX

This project started because most of us found ourselves in situations where online meetings were happening back-to-back, and we didn’t have a moment to breathe, let alone write proper notes. Sometimes, in a meeting with a manager, the discussion would be something like:

> “Can you finish the report by Wednesday? We also need to update the client deck. And make sure we check last month’s metrics.”

While trying to listen to the manager, respond to questions, and understand priorities, writing everything down becomes stressful. After the call, we often forget small details like due dates or who said what. It’s not because we’re careless—it’s just too much to juggle.

So the idea was:  
**What if the system could do the “listening and writing” part while we focus on the “talking and thinking” part?**

That is exactly what MinutesX achieves.

## How MinutesX works

When a Google Meet call is happening, MinutesX uses system audio loopback to listen to whatever you hear through your speakers. It then sends that audio to Gemini for live transcription. This means the AI essentially “hears” the meeting the same way a human would.

Once the transcript is being generated, MinutesX uses several AI agents to break down the conversation:

- **Summary Agent** — writes a short summary of what the meeting was about  
- **Caption Agent** — creates a one-line highlight  
- **Action Agent** — finds tasks and who they belong to  
- **Classifier Agent** — figures out the meeting type  
- **Reviewer Agent** — fixes mistakes and improves clarity  

All these agents run together and create a final document that organizes everything clearly.

## A realistic meeting example

Imagine you're in a Google Meet call with your manager and two teammates. The conversation might look like this:

> Manager: “We need to finish the February performance report. Harsh, can you take the lead on that? Also, please reach out to the marketing team about the landing page changes. I want the updated version by Tuesday. And Priya, you handle the analytics section.”

> Teammate: “I’ll send my draft by tonight. But I might need help with the customer retention numbers.”

> Manager: “That’s fine. Harsh, help Priya if she gets stuck. And let’s schedule a follow-up call on Friday.”

After the meeting, you might get something like this:

**Caption:** “Team sync about February report and website updates.”

**Executive Summary:**  
The team discussed preparing the February performance report and coordinating with the marketing team about landing page updates. Tasks were assigned to Harsh and Priya, with deadlines set for Tuesday. A follow-up meeting is planned for Friday to review progress.

**Key Points:**  
• February report needs to be completed  
• Landing page update requests need to be sent  
• Analytics section assigned to Priya  
• Friday follow-up meeting planned  
• Collaboration on retention metrics  

**Action Items:**  
• Lead February report — Harsh (Due Tuesday)  
• Send landing page update request — Harsh (Due Tuesday)  
• Prepare analytics section — Priya (Draft tonight)  
• Assist with retention metrics — Harsh (As needed)  
• Schedule Friday follow-up — Manager  

## Meeting memory

MinutesX can also remember things from previous meetings. If a topic comes up again, the system can reference past discussions and decisions. This helps connect multiple meetings together.

## Real-world uses

MinutesX can help in many different situations:

- Team syncs  
- Planning meetings  
- Manager 1:1s  
- Client calls  
- School group work  
- Project updates  
- Brainstorming sessions  

Basically anywhere people talk and make decisions.

## Conclusion

MinutesX makes meetings easier by doing the listening and writing for you. It creates summaries, action items, and clean notes automatically. This reduces stress, saves time, and helps people stay focused. Whether you’re discussing deadlines with a manager or reviewing progress with a team, MinutesX ensures nothing is forgotten.

