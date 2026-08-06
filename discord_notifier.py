import os
import logging
import requests
import datetime

def send_discord_notification(message=None, embeds=None, level="info"):
    """Sends a notification to the configured Discord webhook."""
    discord_webhook_url_local = os.getenv("DISCORD_WEBHOOK_URL")
    if not discord_webhook_url_local:
        if not getattr(send_discord_notification, 'logged_missing_url', False):
            logging.warning("DISCORD_WEBHOOK_URL not set. Discord notifications disabled.")
            setattr(send_discord_notification, 'logged_missing_url', True)
        return
    if getattr(send_discord_notification, 'logged_missing_url', False):
        setattr(send_discord_notification, 'logged_missing_url', False)

    payload = {}
    color_map = {"info": 0x3498db, "success": 0x2ecc71, "warning": 0xf1c40f, "error": 0xe74c3c}

    if message:
        payload['content'] = str(message)[:2000] 

    if embeds:
        if not isinstance(embeds, list): embeds = [embeds]
        payload['embeds'] = embeds[:10]
        if level:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for embed in payload['embeds']:
                if isinstance(embed, dict):
                    embed['color'] = color_map.get(str(level).lower(), 0x3498db)
                    if 'timestamp' not in embed: 
                        embed['timestamp'] = timestamp
    try:
        response = requests.post(discord_webhook_url_local, json=payload, timeout=10)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send Discord notification: {e}")
    except Exception as e: 
        logging.error(f"Unexpected error sending Discord notification: {e}")

