import os
import json
import re
from collections import Counter

# --- Load Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

THRESHOLDS = config["THRESHOLDS"]
LANGUAGE_PROFILES = config["LANGUAGE_PROFILES"]
SPAM_TLDS = set(config["SPAM_TLDS"])

# Compile SPAM_REGEX_BLACKLIST
SPAM_REGEX_BLACKLIST = [
    re.compile(pattern, re.IGNORECASE) for pattern in config["SPAM_REGEX_BLACKLIST"]
]

# Compile PII_REGEX_PATTERNS
PII_REGEX_PATTERNS = [
    re.compile(pattern) for pattern in config["PII_REGEX_PATTERNS"]
]


def is_spam_or_trivial(content: str, language: str) -> bool:
    """Advanced check for spam, trivial, or obfuscated content."""
    if not content or not content.strip():
        return True

    content_to_check = content[:2000].strip()

    # Rule 1: Very short content without high-value keywords
    if len(content_to_check) < THRESHOLDS["min_length_for_analysis"]:
        content_lower = content_to_check.lower()
        lang_profile = LANGUAGE_PROFILES.get(language.lower(), LANGUAGE_PROFILES["default"])
        keywords_to_check = {
            **LANGUAGE_PROFILES["default"]["high_value_keywords"],
            **lang_profile["high_value_keywords"],
        }
        if not any(kw in content_lower for kw in keywords_to_check):
            return True

    # Rule 2: Regex blacklist for common spam/trivial patterns
    for pattern in SPAM_REGEX_BLACKLIST:
        if pattern.search(content_to_check):
            return True

    # Rule 3: Highly repetitive characters (refined)
    char_counts = Counter(c for c in content_to_check if not c.isspace())
    total_non_space = sum(char_counts.values())
    if char_counts and total_non_space > 30:
        most_common_count = char_counts.most_common(1)[0][1]
        if most_common_count / total_non_space > 0.85:
            return True

    # Rule 4: Excessive URLs or spam TLDs
    urls = re.findall(r"https?://[^\s/$.?#].[^\s]*", content_to_check)
    if len(urls) > 4:
        return True
    if any(url.endswith(tld) for url in urls for tld in SPAM_TLDS):
        return True

    return False


def has_pii(content: str) -> bool:
    """Checks if the content contains PII or sensitive keys."""
    return any(pattern.search(content) for pattern in PII_REGEX_PATTERNS)


def has_spam_patterns(content: str) -> bool:
    """Checks if the content matches any known spam pattern from the blacklist."""
    return any(pattern.search(content) for pattern in SPAM_REGEX_BLACKLIST)

