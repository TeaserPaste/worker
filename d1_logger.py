import logging
import json
import requests
import datetime
from config import D1_LOG_API_URL

session_id = None 

def open_log_session(app_name="os_sync_worker"):
    """Opens a new logging session with the D1 API."""
    global session_id
    session_id = None 
    if not D1_LOG_API_URL:
        logging.warning("D1_LOG_API_URL not set, cannot open log session.")
        return False
    try:
        response = requests.post(f"{D1_LOG_API_URL}/api/session/open", json={"app_name": app_name}, timeout=10)
        response.raise_for_status() 
        data = response.json()
        session_id = data.get("session_id")
        if session_id:
            logging.info(f"Opened D1 log session: {session_id}")
            return True
        else:
            logging.error(f"Failed to get session_id from D1 API response: {data}")
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to open D1 log session due to RequestException: {e}")
        session_id = None 
        return False
    except Exception as e:
        logging.error(f"Unexpected error opening D1 log session: {e}")
        session_id = None
        return False

def log_event(event_type, details=None, status="info", snippet_id=None):
    """Logs an event to the currently open D1 session."""
    if not D1_LOG_API_URL or not session_id:
        log_level_map = {"INFO": logging.INFO, "WARN": logging.WARNING, "ERROR": logging.ERROR, "DEBUG": logging.DEBUG}
        log_func = logging.log
        level = log_level_map.get(status.upper(), logging.INFO)
        log_message = f"D1_LOG_SKIP ({status.upper()}): {event_type}"
        if snippet_id: log_message += f" [Snippet: {snippet_id}]"
        log_details_str = ""
        if details:
            try: log_details_str = json.dumps(details)[:500] 
            except TypeError: log_details_str = "{Non-serializable details}"
            log_message += f" Details: {log_details_str}"
        log_func(level, log_message)
        return

    log_details = details if isinstance(details, dict) else {}
    if snippet_id: log_details["snippet_id"] = snippet_id

    serializable_details = {}
    for key, value in log_details.items():
        if isinstance(value, str) and len(value) > 250: 
            serializable_details[key] = value[:250] + "..."
        elif isinstance(value, Exception):
            try: serializable_details[key] = str(value)
            except Exception: serializable_details[key] = f"Unrepresentable Exception: {type(value).__name__}"
        elif isinstance(value, (datetime.datetime, datetime.date)):
            serializable_details[key] = value.isoformat()
        elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
             serializable_details[key] = value 
        else:
            try:
                repr_str = repr(value)
                serializable_details[key] = (repr_str[:250] + "...") if len(repr_str) > 250 else repr_str
            except Exception:
                serializable_details[key] = f"Non-representable type: {type(value).__name__}"


    log_entry = {
        "message": str(event_type),       
        "level": str(status).upper(),    
        "details": serializable_details 
    }

    try:
        response = requests.post(f"{D1_LOG_API_URL}/api/session/{session_id}/log", json=log_entry, timeout=5)
        if response.status_code == 400:
            logging.error(f"D1 Log Error 400 (Bad Request). Session: {session_id}. Payload: {json.dumps(log_entry)}. Response: {response.text[:500]}")
        elif response.status_code >= 400: 
            logging.error(f"D1 Log HTTP Error {response.status_code}. Session: {session_id}. Payload (truncated): {json.dumps(log_entry)[:500]}. Response: {response.text[:500]}")

        response.raise_for_status() 
        logging.debug(f"D1 log successful (status: {response.status_code})") 

    except requests.exceptions.RequestException as e:
        truncated_payload = json.dumps(log_entry)[:500] 
        logging.error(f"Failed to log event to D1 session {session_id} due to RequestException: {e}. Payload (truncated): {truncated_payload}")
    except Exception as e:
        truncated_payload = json.dumps(log_entry)[:500]
        logging.error(f"Unexpected error logging event to D1 session {session_id}: {e}. Payload (truncated): {truncated_payload}", exc_info=True)


def close_log_session():
    """Closes the current D1 logging session."""
    global session_id
    if not D1_LOG_API_URL or not session_id:
        return 
    current_session = session_id 
    session_id = None 
    try:
        response = requests.post(f"{D1_LOG_API_URL}/api/session/{current_session}/close", timeout=10)
        response.raise_for_status()
        logging.info(f"Closed D1 log session: {current_session}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to close D1 log session {current_session} due to RequestException: {e}")
    except Exception as e:
        logging.error(f"Unexpected error closing D1 log session {current_session}: {e}")

