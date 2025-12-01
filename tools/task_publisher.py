"""
TaskPublisherTool - Integration with task management systems.

Publishes action items to:
- Slack
- Jira
- Email (via SMTP)
"""
import json
from typing import Any, Dict, List, Optional

from config import config
from observability.logger import get_logger


logger = get_logger(__name__)


class TaskPublisherTool:
    """
    Tool for publishing action items to external systems.
    
    Integrates with:
    - Slack (via Slack SDK)
    - Jira (via Jira API)
    - Custom webhooks
    """
    
    def __init__(self):
        self._slack_client = None
        self._jira_client = None
        logger.info("TaskPublisherTool initialized")
    
    def _get_slack_client(self):
        """Get or create Slack client."""
        if self._slack_client is not None:
            return self._slack_client
        
        try:
            from slack_sdk import WebClient
            
            token = config.integrations.slack_bot_token
            if token:
                self._slack_client = WebClient(token=token)
                return self._slack_client
            else:
                logger.warning("Slack token not configured")
                return None
        except ImportError:
            logger.warning("slack_sdk not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Slack client: {e}")
            return None
    
    def _get_jira_client(self):
        """Get or create Jira client."""
        if self._jira_client is not None:
            return self._jira_client
        
        try:
            from jira import JIRA
            
            url = config.integrations.jira_url
            email = config.integrations.jira_email
            token = config.integrations.jira_api_token
            
            if url and email and token:
                self._jira_client = JIRA(
                    server=url,
                    basic_auth=(email, token)
                )
                return self._jira_client
            else:
                logger.warning("Jira credentials not configured")
                return None
        except ImportError:
            logger.warning("jira library not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Jira client: {e}")
            return None
    
    def publish_to_slack(
        self,
        meeting_id: str,
        caption: str,
        summary: str,
        action_items: List[Dict[str, Any]],
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Publish meeting notes to Slack.
        
        Args:
            meeting_id: Meeting identifier
            caption: One-line caption
            summary: Executive summary
            action_items: List of action items
            channel: Slack channel (defaults to config)
            
        Returns:
            Result with success status and message details
        """
        channel = channel or config.integrations.slack_channel
        logger.info(f"Publishing to Slack channel: {channel}")
        
        client = self._get_slack_client()
        if client is None:
            return {"success": False, "error": "Slack not configured"}
        
        try:
            # Build message blocks
            blocks = self._build_slack_blocks(
                meeting_id, caption, summary, action_items
            )
            
            response = client.chat_postMessage(
                channel=channel,
                text=f"📝 Meeting Notes: {caption}",
                blocks=blocks,
            )
            
            logger.info(f"Published to Slack: {response['ts']}")
            return {
                "success": True,
                "channel": channel,
                "timestamp": response["ts"],
            }
            
        except Exception as e:
            logger.error(f"Failed to publish to Slack: {e}")
            return {"success": False, "error": str(e)}
    
    def _build_slack_blocks(
        self,
        meeting_id: str,
        caption: str,
        summary: str,
        action_items: List[Dict[str, Any]],
    ) -> List[Dict]:
        """Build Slack Block Kit message."""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📝 {caption}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Summary:*\n{summary}"
                }
            },
            {"type": "divider"},
        ]
        
        if action_items:
            action_text = "*Action Items:*\n"
            for item in action_items[:10]:  # Limit to 10 items
                priority = item.get("priority", "medium")
                emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                owner = item.get("owner", "Unassigned")
                due = f" (Due: {item['due_date']})" if item.get("due_date") else ""
                action_text += f"{emoji} {item['description']} - *{owner}*{due}\n"
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": action_text
                }
            })
        
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Meeting ID: `{meeting_id}` | Generated by MinutesX"
                }
            ]
        })
        
        return blocks
    
    def create_jira_tasks(
        self,
        action_items: List[Dict[str, Any]],
        project_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create Jira tasks from action items.
        
        Args:
            action_items: List of action items to create
            project_key: Jira project key (defaults to config)
            
        Returns:
            List of created issue details
        """
        project_key = project_key or config.integrations.jira_project_key
        logger.info(f"Creating Jira tasks in project: {project_key}")
        
        client = self._get_jira_client()
        if client is None:
            return [{"success": False, "error": "Jira not configured"}]
        
        results = []
        
        for item in action_items:
            try:
                priority_map = {
                    "high": "Highest",
                    "medium": "Medium",
                    "low": "Low",
                }
                
                issue_dict = {
                    "project": {"key": project_key},
                    "summary": item["description"][:255],
                    "description": f"Action item from meeting.\n\nOwner: {item.get('owner', 'Unassigned')}",
                    "issuetype": {"name": "Task"},
                    "priority": {"name": priority_map.get(item.get("priority", "medium"), "Medium")},
                }
                
                # Add due date if specified
                if item.get("due_date"):
                    issue_dict["duedate"] = item["due_date"]
                
                issue = client.create_issue(fields=issue_dict)
                
                results.append({
                    "success": True,
                    "key": issue.key,
                    "url": f"{config.integrations.jira_url}/browse/{issue.key}",
                    "description": item["description"],
                })
                
                logger.info(f"Created Jira issue: {issue.key}")
                
            except Exception as e:
                logger.error(f"Failed to create Jira issue: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "description": item["description"],
                })
        
        return results
    
    def publish_to_webhook(
        self,
        webhook_url: str,
        meeting_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Publish meeting data to a custom webhook.
        
        Args:
            webhook_url: The webhook URL
            meeting_data: Data to send
            
        Returns:
            Result with success status
        """
        import requests
        
        logger.info(f"Publishing to webhook: {webhook_url}")
        
        try:
            response = requests.post(
                webhook_url,
                json=meeting_data,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            
            return {
                "success": True,
                "status_code": response.status_code,
            }
            
        except Exception as e:
            logger.error(f"Webhook publish failed: {e}")
            return {"success": False, "error": str(e)}
