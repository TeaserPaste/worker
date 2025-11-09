import logging
import re
import datetime
from collections import Counter
import math
import pygments
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.token import Token, String, Comment

# --- Cấu hình Trọng số (Weights) và Ngưỡng (Thresholds) ---
# Ngưỡng và Giá trị tối ưu
THRESHOLDS = {
    'optimal_length': 4000,  # Chiều dài tối ưu cho snippet
    'min_length_for_analysis': 80, # Giảm nhẹ ngưỡng tối thiểu
    'min_lines_for_complexity': 5, # Số dòng tối thiểu để đánh giá độ phức tạp
    'max_line_length': 120,  # Phạt các dòng quá dài
    'min_avg_line_length': 15, # Phạt các dòng quá ngắn (tránh code bị gộp dòng)
    'long_comment_threshold': 20, # Số ký tự tối thiểu cho một comment chất lượng
    'gibberish_threshold': 0.4, # Ngưỡng phát hiện vô nghĩa (tỷ lệ nguyên âm/phụ âm)
    'long_unbroken_string': 100 # Ngưỡng phát hiện Base64/obfuscation
}

# --- Language Profiles ---
LANGUAGE_PROFILES = {
    'python': {
        'base_score': 0.60,
        'decay_days': 90,
        'scoring_weights': {'length': 0.15, 'syntax': 0.45, 'utility': 0.30, 'age': 0.10},
        'high_value_keywords': {
            'fastapi': 2.0, 'pydantic': 2.0, 'decorator': 2.0, 'contextmanager': 2.2,
            'pandas': 2.0, 'numpy': 1.8, 'scikitlearn': 2.2, 'pytorch': 2.5, 'tensorflow': 2.5,
            'async': 1.8, 'await': 1.5,
        }
    },
    'javascript': {
        'base_score': 0.50,
        'decay_days': 60,
        'scoring_weights': {'length': 0.20, 'syntax': 0.45, 'utility': 0.25, 'age': 0.10},
        'high_value_keywords': {
            'react': 1.5, 'vue': 1.5, 'svelte': 1.8, 'nextjs': 2.2, 'vite': 1.5,
            'useEffect': 2.5, 'useState': 2.0, 'redux': 1.8, 'tailwind': 1.7,
            'async': 1.8, 'await': 1.5, 'Promise': 1.5,
        }
    },
    'typescript': {
        'base_score': 0.60,
        'decay_days': 60,
        'scoring_weights': {'length': 0.20, 'syntax': 0.45, 'utility': 0.25, 'age': 0.10},
        'high_value_keywords': {
            'react': 1.5, 'vue': 1.5, 'svelte': 1.8, 'nextjs': 2.2, 'vite': 1.5,
            'useEffect': 2.5, 'useState': 2.0, 'redux': 1.8, 'tailwind': 1.7,
            'interface': 2.0, 'enum': 1.5, 'type': 1.5,
        }
    },
    'java': {
        'base_score': 0.55,
        'decay_days': 180,
        'scoring_weights': {'length': 0.25, 'syntax': 0.40, 'utility': 0.25, 'age': 0.10},
        'high_value_keywords': {'spring': 2.0, 'maven': 1.5}
    },
    'csharp': {
        'base_score': 0.55,
        'decay_days': 180,
        'scoring_weights': {'length': 0.25, 'syntax': 0.40, 'utility': 0.25, 'age': 0.10},
        'high_value_keywords': {'.net': 2.0, 'linq': 1.8}
    },
    'go': {
        'base_score': 0.45,
        'decay_days': 120,
        'scoring_weights': {'length': 0.25, 'syntax': 0.40, 'utility': 0.25, 'age': 0.10},
        'high_value_keywords': {'goroutine': 2.0, 'channel': 1.8}
    },
    'rust': {
        'base_score': 0.45,
        'decay_days': 120,
        'scoring_weights': {'length': 0.25, 'syntax': 0.40, 'utility': 0.25, 'age': 0.10},
        'high_value_keywords': {'cargo': 1.5, 'tokio': 2.0}
    },
    'sql': {
        'base_score': 0.35,
        'decay_days': 365,
        'scoring_weights': {'length': 0.30, 'syntax': 0.40, 'utility': 0.20, 'age': 0.10},
        'high_value_keywords': {'join': 1.5, 'with': 1.8, 'group by': 1.5, 'window function': 2.5}
    },
    'markdown': {
        'base_score': 0.15,
        'decay_days': 120,
        'scoring_weights': {'length': 0.40, 'syntax': 0.10, 'utility': 0.40, 'age': 0.10},
        'high_value_keywords': {}
    },
    'plaintext': {
        'base_score': 0.10,
        'decay_days': 120,
        'scoring_weights': {'length': 0.50, 'syntax': 0.10, 'utility': 0.30, 'age': 0.10},
        'high_value_keywords': {}
    },
    'bash': {
        'base_score': 0.30,
        'decay_days': 180,
        'scoring_weights': {'length': 0.30, 'syntax': 0.40, 'utility': 0.20, 'age': 0.10},
        'high_value_keywords': {'grep': 1.2, 'awk': 1.5, 'sed': 1.5}
    },
    'html': {
        'base_score': 0.30,
        'decay_days': 180,
        'scoring_weights': {'length': 0.40, 'syntax': 0.20, 'utility': 0.30, 'age': 0.10},
        'high_value_keywords': {}
    },
    'css': {
        'base_score': 0.25,
        'decay_days': 180,
        'scoring_weights': {'length': 0.40, 'syntax': 0.30, 'utility': 0.20, 'age': 0.10},
        'high_value_keywords': {}
    },
    'default': {
        'base_score': 0.30,
        'decay_days': 120,
        'scoring_weights': {'length': 0.25, 'syntax': 0.40, 'utility': 0.25, 'age': 0.10},
        'high_value_keywords': {
            'dockerfile': 3.0, 'kubernetes': 3.0, 'terraform': 2.8, 'aws lambda': 2.5,
            'ci/cd': 2.2, 'github actions': 2.2, 'argocd': 2.7, 'microservice': 2.5,
            'graphql': 2.3, 'grpc': 2.5
        }
    }
}

# Các "Contextual Combos": nhân điểm thưởng nếu các cặp từ khóa xuất hiện cùng nhau
CONTEXTUAL_COMBOS = {
    'react_hooks': (['react', 'useEffect'], 2.0),
    'python_async': (['python', 'async def'], 2.5),
    'api_design': (['graphql', 'resolver'], 2.2)
}

# Regex cho các mẫu spam hoặc code chất lượng rất thấp
SPAM_REGEX_BLACKLIST = [
    re.compile(r'buy now|crypto|forex|free trial|seo services|online casino', re.IGNORECASE),
    re.compile(r'(\b\w+\b\s*){1,2}Copyright \d{4}', re.IGNORECASE) # "MyComponent Copyright 2023"
]

# Các TLDs (Top-Level Domains) phổ biến trong spam
SPAM_TLDS = {'.xyz', '.top', '.tk', '.site', '.click', '.loan', '.online'}


# --- HELPER FUNCTIONS ---

def get_base_priority(language: str) -> float:
    """Gets the base score for the language."""
    profile = LANGUAGE_PROFILES.get(language.lower(), LANGUAGE_PROFILES['default'])
    return profile['base_score']

def _is_gibberish(text: str) -> bool:
    """Checks for gibberish text based on vowel-to-consonant ratio."""
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"

    v_count = 0
    c_count = 0

    for char in text.lower():
        if char in vowels:
            v_count += 1
        elif char in consonants:
            c_count += 1

    if v_count + c_count == 0:
        return False # Not enough alphabetic characters to judge

    ratio = v_count / (v_count + c_count)

    # Typical English text has a vowel ratio of ~35-45%
    # Ratios outside 15-65% are suspicious
    return not (0.15 < ratio < 0.65)

def is_spam_or_trivial(content: str, language: str) -> bool:
    """Advanced check for spam, trivial, or obfuscated content."""
    if not content or not content.strip():
        return True
    
    content_to_check = content[:2000].strip()
    
    # Rule 1: Very short content without high-value keywords
    if len(content_to_check) < THRESHOLDS['min_length_for_analysis']:
        content_lower = content_to_check.lower()
        lang_profile = LANGUAGE_PROFILES.get(language.lower(), LANGUAGE_PROFILES['default'])
        keywords_to_check = {**LANGUAGE_PROFILES['default']['high_value_keywords'], **lang_profile['high_value_keywords']}
        if not any(kw in content_lower for kw in keywords_to_check):
            return True

    # Rule 2: Regex blacklist for common spam/trivial patterns
    for pattern in SPAM_REGEX_BLACKLIST:
        if pattern.search(content_to_check):
            return True

    # Rule 3: Gibberish detection
    words = re.findall(r'\b\w{5,}\b', content_to_check) # Check longer words
    if len(words) > 5 and _is_gibberish("".join(words)):
        return True

    # Rule 4: Detect long, unbroken strings (potential Base64/obfuscation)
    if any(len(word) > THRESHOLDS['long_unbroken_string'] for word in content_to_check.split()):
        return True

    # Rule 5: Highly repetitive characters (refined)
    char_counts = Counter(c for c in content_to_check if not c.isspace())
    total_non_space = sum(char_counts.values())
    if char_counts and total_non_space > 30:
        most_common_count = char_counts.most_common(1)[0][1]
        if most_common_count / total_non_space > 0.6: # Lowered threshold
            return True

    # Rule 6: Excessive URLs or spam TLDs
    urls = re.findall(r'https?://[^\s/$.?#].[^\s]*', content_to_check)
    if len(urls) > 4:
        return True
    if any(url.endswith(tld) for url in urls for tld in SPAM_TLDS):
        return True

    return False

def calculate_age_decay(created_at: datetime.datetime, language: str) -> float:
    """Calculates age score based on decay rate."""
    profile = LANGUAGE_PROFILES.get(language.lower(), LANGUAGE_PROFILES['default'])
    decay_days = profile['decay_days']
    
    # Ensure created_at is timezone-aware
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=datetime.timezone.utc)
    
    now = datetime.datetime.now(datetime.timezone.utc)
    age_days = (now - created_at).days
    
    if age_days < 0:
        return 1.0 # Future date, max score
        
    # Tính điểm Decay: Điểm max là 1.0 (mới nhất), giảm dần về 0.0
    # Công thức: max(0.2, 1.0 - (age_days / decay_days))
    
    # Giảm điểm tuyến tính. Snippet cũ hơn decay_days sẽ có điểm 0.2 (min)
    decay_score = 1.0 - (age_days / decay_days)
    
    # Đặt ngưỡng tối thiểu (ví dụ: 0.2)
    return max(0.2, decay_score)


def analyze_structural_complexity(content: str, language: str) -> float:
    """
    Analyzes code complexity using Pygments for tokenization.
    Scores based on token diversity, density of significant tokens, and language-specific heuristics.
    """
    try:
        lexer = get_lexer_by_name(language, stripall=False)
    except pygments.util.ClassNotFound:
        # Lỗi nghiêm trọng: nếu 'plaintext' cũng không tìm thấy, fallback an toàn
        try:
            lexer = get_lexer_by_name('plaintext', stripall=False)
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
    
    if lang_lower == 'python':
        if 'async' in token_values and 'def' in token_values:
            language_specific_bonus += 0.15
        if '@' in token_values: # Decorators
            language_specific_bonus += 0.10
    
    elif lang_lower in ['javascript', 'typescript']:
        if 'async' in token_values and '=>' in token_values:
            language_specific_bonus += 0.15
        if 'import' in token_values or 'export' in token_values:
            language_specific_bonus += 0.05
        # Check for modern JS keywords
        js_keywords = {'useEffect', 'useState', 'useContext', 'Promise'}
        for kw in js_keywords:
            if kw in token_values:
                language_specific_bonus += 0.10

    elif lang_lower == 'sql':
        sql_keywords = {'JOIN', 'WITH', 'GROUP BY', 'PARTITION BY'}
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
    lang_profile = LANGUAGE_PROFILES.get(language.lower(), LANGUAGE_PROFILES['default'])
    keywords_to_check = {**LANGUAGE_PROFILES['default']['high_value_keywords'], **lang_profile['high_value_keywords']}

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

def calculate_comment_utility(content: str, language: str) -> float:
    """
    Calculates utility score based on comment quality, readability, and structure.
    Uses Pygments for accurate comment and docstring detection.
    """
    try:
        lexer = get_lexer_by_name(language, stripall=False)
    except pygments.util.ClassNotFound:
        try:
            lexer = get_lexer_by_name('plaintext', stripall=False)
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
            if len(tvalue) > THRESHOLDS['long_comment_threshold']:
                long_comments += 1
            if any(kw in tvalue for kw in ['TODO:', 'FIXME:', 'NOTE:']):
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
            if line_len > THRESHOLDS['max_line_length']:
                long_line_penalty += 0.05 # 5% penalty per long line

    avg_line_length = total_line_length / num_code_lines if num_code_lines > 0 else 0
    short_line_penalty = 0
    if num_code_lines > THRESHOLDS['min_lines_for_complexity'] and avg_line_length < THRESHOLDS['min_avg_line_length']:
        short_line_penalty = (THRESHOLDS['min_avg_line_length'] - avg_line_length) / THRESHOLDS['min_avg_line_length'] * 0.5
    
    readability_score = 1.0 - min(0.5, long_line_penalty) - short_line_penalty

    # --- Final Combination ---
    final_score = (
        (density_score * 0.5) +
        (docstring_bonus * 0.2) +
        (quality_score * 0.3)
    ) * readability_score

    return max(0.0, min(1.0, final_score))


# --- MAIN CALCULATION FUNCTION ---

# SỬA LỖI: Thay đổi kiểu trả về từ (float, str) sang tuple[float, str]
def calculate_priority(content: str, language: str, created_at: datetime.datetime, is_verified: bool = False) -> tuple[float, str]:
    """
    Calculates the final priority score (0.1 to 1.0) using the advanced rule-based engine.
    
    Returns: (priority_score, assessment_string)
    """

    if is_verified:
        return 1.0, "Assessment: Overridden by Admin Manual Verification."

    lang = language.lower() if language else 'plaintext'
    
    # --- PHASE 1: PRE-CHECK (Spam/Trivial Rule) ---
    if is_spam_or_trivial(content, lang):
        return 0.1, "Assessment: Rejected (Trivial or Spam)."
    
    lang_base_score = get_base_priority(lang)
    content_length = len(content)
    
    # --- PHASE 2: SCORE COMPONENTS ---
    
    # 1. Length Score
    optimal_length = THRESHOLDS['optimal_length']
    if content_length <= optimal_length:
        length_score = content_length / optimal_length
    else:
        # Áp dụng hàm suy giảm (decay) cho các snippet quá dài
        # Công thức: score = 0.2 + 0.8 * e^(-(length_ratio - 1)^2 / 4)
        # Bắt đầu giảm từ 1.0 và tiệm cận 0.2
        length_ratio = content_length / optimal_length
        decay_factor = math.exp(-((length_ratio - 1) ** 2) / 4)
        length_score = 0.2 + 0.8 * decay_factor

    length_score = max(0.2, min(1.0, length_score)) # Đảm bảo score trong khoảng [0.2, 1.0]

    # 2. Age Decay Score
    age_score = calculate_age_decay(created_at, lang)
    
    # 3. Structural Complexity Score (Code-aware analysis)
    syntax_score = analyze_structural_complexity(content, lang)
    
    # 4. Comment & Readability Utility Score
    utility_score = calculate_comment_utility(content, lang)
    
    # --- PHASE 3: FINAL CALCULATION ---
    
    # Calculate raw score using the new weights
    lang_profile = LANGUAGE_PROFILES.get(lang, LANGUAGE_PROFILES['default'])
    weights = lang_profile['scoring_weights']

    raw_score = (
        weights['length'] * length_score +
        weights['syntax'] * syntax_score +
        weights['utility'] * utility_score +
        weights['age'] * age_score
    )
    
    # The final priority is a blend of the language's base score and the calculated raw score.
    # This ensures that good code in a low-priority language is still ranked lower than
    # exceptional code in a high-priority language.
    final_priority = lang_base_score + (1.0 - lang_base_score) * raw_score
    
    # Ensure the score is within the valid range [0.1, 1.0]
    final_priority = max(0.1, min(1.0, final_priority))
    
    # --- ASSESSMENT STRING ---
    assessment_details = (
        f"Length:{length_score:.2f} (w:{weights['length']}), "
        f"Syntax:{syntax_score:.2f} (w:{weights['syntax']}), "
        f"Utility:{utility_score:.2f} (w:{weights['utility']}), "
        f"Age:{age_score:.2f} (w:{weights['age']})"
    )
    assessment_string = f"Priority={final_priority:.3f} | Details: {assessment_details}"
    
    return final_priority, assessment_string