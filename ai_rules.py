import os
import json
import logging
import re
import requests

# --- HTTP Connection Reuse Session ---
http_session = requests.Session()

# --- Load Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

AI_CONFIG = config["AI_CONFIG"]


def check_content_safety(content: str, openrouter_key: str) -> tuple[bool, str]:
    """Performs content safety check via Nvidia Nemotron.

    Returns (is_safe, assessment_string).
    """
    try:
        safety_check_text = content[:5000]
        payload = {
            "model": AI_CONFIG["NEMOTRON_MODEL"],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": safety_check_text}]}
            ],
            "temperature": 0.0,
            "max_tokens": 100,
        }
        response = http_session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=AI_CONFIG["NEMOTRON_TIMEOUT"],
        )
        if response.status_code == 200:
            resp_json = response.json()
            choices = resp_json.get("choices", [])
            if choices:
                resp_text = choices[0].get("message", {}).get("content", "")
                if "user safety: unsafe" in resp_text.lower():
                    return False, "Assessment: Rejected (Sensitive information or unsafe content flagged by AI)."
        else:
            logging.warning(f"Nemotron API returned status {response.status_code}: {response.text}")
    except Exception as e:
        logging.warning(f"Failed to check content safety with Nemotron: {e}")

    return True, ""


def get_ai_priority_rating(content: str, language: str, openrouter_key: str) -> tuple[float, str]:
    """Gets priority score (0.1 to 1.0) and assessment from Cohere model.

    Returns (score, assessment_string) or (None, None) if failed.
    """
    try:
        prompt_content = content[:5000]
        prompt_template = AI_CONFIG["COHERE_PROMPT_TEMPLATE"]
        prompt = prompt_template.format(lang=language, prompt_content=prompt_content)

        payload = {
            "model": AI_CONFIG["COHERE_MODEL"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 150,
        }
        response = http_session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=AI_CONFIG["COHERE_TIMEOUT"],
        )
        if response.status_code == 200:
            resp_json = response.json()
            choices = resp_json.get("choices", [])
            if choices:
                resp_text = choices[0].get("message", {}).get("content", "").strip()
                match = re.search(r"\{.*\}", resp_text, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                        score = float(data.get("priority_score", 0.1))
                        details = str(data.get("assessment_details", "No details provided."))
                        score = max(0.1, min(1.0, score))
                        return score, f"Priority={score:.3f} | Details: {details}"
                    except Exception as parse_err:
                        logging.warning(f"Failed to parse JSON from Cohere model: {parse_err}. Content: {resp_text}")
        else:
            logging.warning(f"Cohere API returned status {response.status_code}: {response.text}")
    except Exception as e:
        logging.warning(f"Failed to get priority rating from Cohere: {e}")

    return None, None


def get_non_programming_priority_rating(content: str, language: str, openrouter_key: str) -> tuple[float, str]:
    """Gets priority score (0.1 to 1.0) and assessment from Ling model for non-programming languages.

    Returns (score, assessment_string) or (None, None) if failed.
    """
    try:
        prompt_content = content[:5000]
        prompt_template = AI_CONFIG["LING_PROMPT_TEMPLATE"]
        prompt = prompt_template.format(lang=language, prompt_content=prompt_content)

        payload = {
            "model": AI_CONFIG["LING_MODEL"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 150,
        }
        response = http_session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=AI_CONFIG["LING_TIMEOUT"],
        )
        if response.status_code == 200:
            resp_json = response.json()
            choices = resp_json.get("choices", [])
            if choices:
                resp_text = choices[0].get("message", {}).get("content", "").strip()
                match = re.search(r"\{.*\}", resp_text, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                        score = float(data.get("priority_score", 0.1))
                        details = str(data.get("assessment_details", "No details provided."))
                        score = max(0.1, min(1.0, score))
                        return score, f"Priority={score:.3f} | Details: {details}"
                    except Exception as parse_err:
                        logging.warning(f"Failed to parse JSON from Ling model: {parse_err}. Content: {resp_text}")
        else:
            logging.warning(f"Ling API returned status {response.status_code}: {response.text}")
    except Exception as e:
        logging.warning(f"Failed to get priority rating from Ling: {e}")

    return None, None

