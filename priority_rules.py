import os
import json
import logging
import datetime
import math
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.token import Token, String, Comment

# Import helper checks
from regex_rules import is_spam_or_trivial, has_pii, has_spam_patterns
from ai_rules import check_content_safety, get_ai_priority_rating

# --- Load Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

THRESHOLDS = config["THRESHOLDS"]
LANGUAGE_PROFILES = config["LANGUAGE_PROFILES"]
CONTEXTUAL_COMBOS = config["CONTEXTUAL_COMBOS"]
NON_PROGRAMMING_LANGS = set(config["NON_PROGRAMMING_LANGS"])


# --- HELPER FUNCTIONS ---

def get_base_priority(language: str) -> float:
    """Gets the base score for the language."""
    profile = LANGUAGE_PROFILES.get(language.lower(), LANGUAGE_PROFILES["default"])
    return profile["base_score"]


def calculate_age_decay(created_at: datetime.datetime, language: str) -> float:
    """Calculates age score based on decay rate."""
    profile = LANGUAGE_PROFILES.get(language.lower(), LANGUAGE_PROFILES["default"])
    decay_days = profile["decay_days"]

    # Ensure created_at is timezone-aware
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=datetime.timezone.utc)

    now = datetime.datetime.now(datetime.timezone.utc)
    age_days = (now - created_at).days

    if age_days < 0:
        return 1.0  # Future date, max score

    # Calculate Decay score: Max score is 1.0 (newest), decaying to 0.0
    # Formula: max(0.2, 1.0 - (age_days / decay_days))

    # Linear decay. Snippet older than decay_days will have score 0.2 (min)
    decay_score = 1.0 - (age_days / decay_days)

    # Set minimum threshold (e.g. 0.2)
    return max(0.2, decay_score)


def analyze_structural_complexity(content: str, language: str, tokens: list = None) -> float:
    """Analyzes code complexity using Pygments for tokenization.

    Scores based on token diversity, density of significant tokens, and language-specific heuristics.
    """
    if tokens is None:
        try:
            lexer = get_lexer_by_name(language, stripall=False)
        except pygments.util.ClassNotFound:
            try:
                lexer = get_lexer_by_name("plaintext", stripall=False)
            except pygments.util.ClassNotFound:
                logging.warning(f"Pygments lexer for '{language}' and even for 'plaintext' not found. Returning neutral score for syntax.")
                return 0.5
        tokens = list(pygments.lex(content, lexer))

    if not tokens:
        return 0.0

    total_tokens = len(tokens)

    # --- Metrics Calculation ---
    significant_token_count = 0
    unique_token_types = set()
    error_token_count = 0
    language_specific_bonus = 0.0

    # Define significant token types
    significant_tokens = {
        Token.Keyword, Token.Name.Function, Token.Name.Class, Token.Operator,
        Token.Literal.String.Interpol, Token.Name.Decorator
    }

    token_values = [t[1] for t in tokens]

    for ttype, tvalue in tokens:
        unique_token_types.add(ttype)
        if ttype in Token.Error:
            error_token_count += 1
        elif any(ttype in s for s in significant_tokens):
            significant_token_count += 1

    # --- Language-Specific Heuristics ---
    lang_lower = language.lower()

    if lang_lower == "python":
        if "async" in token_values and "def" in token_values:
            language_specific_bonus += 0.15
        if "@" in token_values:  # Decorators
            language_specific_bonus += 0.10

    elif lang_lower in ["javascript", "typescript"]:
        if "async" in token_values and "=>" in token_values:
            language_specific_bonus += 0.15
        if "import" in token_values or "export" in token_values:
            language_specific_bonus += 0.05
        # Check for modern JS keywords
        js_keywords = {"useEffect", "useState", "useContext", "Promise"}
        for kw in js_keywords:
            if kw in token_values:
                language_specific_bonus += 0.10

    elif lang_lower == "sql":
        sql_keywords = {"JOIN", "WITH", "GROUP BY", "PARTITION BY"}
        for kw in sql_keywords:
            if kw.upper() in [v.upper() for v in token_values]:
                language_specific_bonus += 0.15

    # --- Scoring ---
    # 1. Density Score: (significant tokens / total tokens)
    density_score = (significant_token_count / total_tokens) if total_tokens > 0 else 0

    # 2. Diversity Score: (unique token types / total tokens) - penalized for being too short
    # Use a log scale to reward initial diversity more
    diversity_score = math.log1p(len(unique_token_types)) / math.log1p(total_tokens) if total_tokens > 0 else 0

    # 3. Keyword Score
    keyword_score = 0
    content_lower = content.lower()

    # Get language-specific keywords and also check default keywords
    lang_profile = LANGUAGE_PROFILES.get(language.lower(), LANGUAGE_PROFILES["default"])
    keywords_to_check = {**LANGUAGE_PROFILES["default"]["high_value_keywords"], **lang_profile["high_value_keywords"]}

    for keyword, weight in keywords_to_check.items():
        if keyword in content_lower:
            keyword_score += weight

    # Apply Contextual Combos
    for combo, (terms, multiplier) in CONTEXTUAL_COMBOS.items():
        if all(term in content_lower for term in terms):
            keyword_score *= multiplier

    # Normalize keyword score (e.g., capped at 1.5)
    normalized_keyword_score = min(1.5, keyword_score / 10.0)

    # Final Score Combination
    syntax_score = (
        (density_score * 0.4) +
        (diversity_score * 0.3) +
        (normalized_keyword_score * 0.3) +
        language_specific_bonus
    )

    # 4. Error Penalty
    error_ratio = error_token_count / total_tokens if total_tokens > 0 else 0
    penalty_multiplier = max(0.2, 1.0 - (error_ratio * 2.5))
    final_score = syntax_score * penalty_multiplier

    return min(1.0, final_score)


def calculate_comment_utility(content: str, language: str, tokens: list = None) -> float:
    """Calculates utility score based on comment quality, readability, and structure.

    Uses Pygments for accurate comment and docstring detection.
    """
    if tokens is None:
        try:
            lexer = get_lexer_by_name(language, stripall=False)
        except pygments.util.ClassNotFound:
            try:
                lexer = get_lexer_by_name("plaintext", stripall=False)
            except pygments.util.ClassNotFound:
                logging.warning(f"Pygments lexer for '{language}' and 'plaintext' not found. Returning neutral score for utility.")
                return 0.5
        tokens = list(pygments.lex(content, lexer))

    lines = content.splitlines()

    if not tokens or not lines:
        return 0.0

    total_chars = len(content)
    comment_chars = 0
    docstring_chars = 0
    special_comments = 0
    long_comments = 0

    for ttype, tvalue in tokens:
        if ttype in Comment:
            comment_chars += len(tvalue)
            if len(tvalue) > THRESHOLDS["long_comment_threshold"]:
                long_comments += 1
            if any(kw in tvalue for kw in ["TODO:", "FIXME:", "NOTE:"]):
                special_comments += 1
        if ttype in String.Docstring:
            docstring_chars += len(tvalue)

    # --- Scoring Components ---

    # 1. Comment & Docstring Ratio Score
    comment_density = (comment_chars + docstring_chars) / total_chars if total_chars > 0 else 0
    # Ideal density is around 15-20%
    density_score = 1.0 - abs(comment_density - 0.18) * 3.0
    density_score = max(0.0, density_score)

    # Bonus for docstrings (highly valuable)
    docstring_bonus = min(0.3, (docstring_chars / total_chars) * 3.0)

    # 2. Comment Quality Score
    quality_score = 0.0
    if comment_chars > 0:
        quality_score += min(0.2, (special_comments * 0.1))
        quality_score += min(0.2, (long_comments * 0.05))

    # 3. Readability Penalties
    total_lines = len(lines)
    code_lines = [line for line in lines if line.strip()]
    num_code_lines = len(code_lines)

    long_line_penalty = 0
    total_line_length = 0
    if num_code_lines > 0:
        for line in code_lines:
            line_len = len(line)
            total_line_length += line_len
            if line_len > THRESHOLDS["max_line_length"]:
                long_line_penalty += 0.05  # 5% penalty per long line

    avg_line_length = total_line_length / num_code_lines if num_code_lines > 0 else 0
    short_line_penalty = 0
    if num_code_lines > THRESHOLDS["min_lines_for_complexity"] and avg_line_length < THRESHOLDS["min_avg_line_length"]:
        short_line_penalty = (THRESHOLDS["min_avg_line_length"] - avg_line_length) / THRESHOLDS["min_avg_line_length"] * 0.5

    readability_score = 1.0 - min(0.5, long_line_penalty) - short_line_penalty

    # --- Final Combination ---
    final_score = (
        (density_score * 0.5) +
        (docstring_bonus * 0.2) +
        (quality_score * 0.3)
    ) * readability_score

    return max(0.0, min(1.0, final_score))


# --- ORIGINAL RULE-BASED SCORING FALLBACK ---

def calculate_priority_rule_based(content: str, language: str, created_at: datetime.datetime) -> tuple[float, str]:
    """Original robust rule-based priority score calculation. Used as a safe fallback."""
    lang = language.lower() if language else "plaintext"

    if is_spam_or_trivial(content, lang):
        return 0.1, "Assessment: Rejected (Trivial or Spam)."

    lang_base_score = get_base_priority(lang)
    content_length = len(content)

    # 1. Length Score
    optimal_length = THRESHOLDS["optimal_length"]
    if content_length <= optimal_length:
        length_score = content_length / optimal_length
    else:
        # Apply decay function for snippets that are too long
        # Formula: score = 0.2 + 0.8 * e^(-(length_ratio - 1)^2 / 4)
        # Starts decreasing from 1.0 and asymptotes to 0.2
        length_ratio = content_length / optimal_length
        decay_factor = math.exp(-((length_ratio - 1) ** 2) / 4)
        length_score = 0.2 + 0.8 * decay_factor

    length_score = max(0.2, min(1.0, length_score))  # Ensure score is within [0.2, 1.0]

    # 2. Age Decay Score
    age_score = calculate_age_decay(created_at, lang)

    # Pygments tokenization share optimization
    try:
        lexer = get_lexer_by_name(lang, stripall=False)
    except pygments.util.ClassNotFound:
        try:
            lexer = get_lexer_by_name("plaintext", stripall=False)
        except pygments.util.ClassNotFound:
            lexer = None

    if lexer:
        shared_tokens = list(pygments.lex(content, lexer))
    else:
        shared_tokens = []

    # 3. Structural Complexity Score (Code-aware analysis)
    syntax_score = analyze_structural_complexity(content, lang, tokens=shared_tokens)

    # 4. Comment & Readability Utility Score
    utility_score = calculate_comment_utility(content, lang, tokens=shared_tokens)

    # --- PHASE 3: FINAL CALCULATION ---
    lang_profile = LANGUAGE_PROFILES.get(lang, LANGUAGE_PROFILES["default"])
    weights = lang_profile["scoring_weights"]

    raw_score = (
        weights["length"] * length_score +
        weights["syntax"] * syntax_score +
        weights["utility"] * utility_score +
        weights["age"] * age_score
    )

    final_priority = lang_base_score + (1.0 - lang_base_score) * raw_score

    # Ensure the score is within the valid range [0.1, 1.0]
    final_priority = max(0.1, min(1.0, final_priority))

    assessment_details = (
        f"Length:{length_score:.2f} (w:{weights['length']}), "
        f"Syntax:{syntax_score:.2f} (w:{weights['syntax']}), "
        f"Utility:{utility_score:.2f} (w:{weights['utility']}), "
        f"Age:{age_score:.2f} (w:{weights['age']})"
    )
    assessment_string = f"Priority={final_priority:.3f} | Details: {assessment_details}"

    return final_priority, assessment_string


# --- MAIN CALCULATION ENTRYPOINT ---

def calculate_priority(content: str, language: str, created_at: datetime.datetime, is_verified: bool = False) -> tuple[float, str]:
    """Calculates the final priority score (0.1 to 1.0) using hybrid priority logic (Regex -> AI).

    Returns: (priority_score, assessment_string)
    """
    # 1. Admin Verification Overrides
    if is_verified:
        return 1.0, "Assessment: Overridden by Admin Manual Verification."

    lang = language.lower() if language else "plaintext"

    # 2. Too Short check (< 15 characters)
    stripped_content = content.strip() if content else ""
    if len(stripped_content) < 15:
        return 0.1, "Assessment: Rejected (Too short - less than 15 characters)."

    # 3. Too Long check (1000+ lines)
    if len(content.splitlines()) >= 1000:
        return 0.1, "Assessment: Rejected (Too long - 1000+ lines)."

    # 4. Too Long check (50000+ characters)
    if len(content) >= 50000:
        return 0.1, "Assessment: Rejected (Too long - 50000+ characters)."

    # 5. Non-programming languages check
    if lang in NON_PROGRAMMING_LANGS:
        return 0.1, "Assessment: Rejected (Non-programming language)."

    # 6. Likely spam checks (Reusing spam regex checking helper)
    if has_spam_patterns(content):
        return 0.1, "Assessment: Rejected (Spam detected)."

    # 7. Sensitive public UGC / PII checks
    # Local pattern pre-check
    if has_pii(content):
        return 0.1, "Assessment: Rejected (Sensitive information detected via local checks)."

    # External Nemotron safety check
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        is_safe, safety_assessment = check_content_safety(content, openrouter_key)
        if not is_safe:
            return 0.1, safety_assessment

    # 8. Main LLM Priority Rating with cohere/north-mini-code:free
    if openrouter_key:
        ai_score, ai_assessment = get_ai_priority_rating(content, lang, openrouter_key)
        if ai_score is not None:
            return ai_score, ai_assessment

    # Fallback to Rule-based calculation if AI is disabled or fails
    logging.info("Falling back to robust rule-based priority calculation.")
    return calculate_priority_rule_based(content, lang, created_at)

