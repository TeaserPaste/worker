import re
import datetime
from collections import Counter
import math
import pygments
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.token import Token, String, Comment

# --- Cấu hình Trọng số (Weights) và Ngưỡng (Thresholds) ---
WEIGHTS = {
    'length': 0.25,          # Trọng số cho Độ dài (giảm nhẹ)
    'syntax': 0.40,          # Trọng số cho Phân tích Cú pháp (tăng mạnh)
    'utility': 0.25,         # Trọng số cho Độ hữu dụng (Comment/Readability)
    'age': 0.10,             # Trọng số cho Độ mới (Age decay)
}

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

# Điểm cơ sở theo ngôn ngữ
LANGUAGE_BASE_SCORES = {
    'plaintext': 0.10,
    'markdown': 0.15,
    'javascript': 0.50,
    'python': 0.60,
    'java': 0.55,
    'csharp': 0.55,
    'typescript': 0.60,
    'go': 0.45,
    'rust': 0.45,
    'sql': 0.35,
    'bash': 0.30,
    'html': 0.30,
    'css': 0.25,
    'default': 0.30
}

# Tốc độ giảm điểm theo tuổi (Decay Rate - tính bằng ngày)
# JS/Python giảm nhanh hơn (API thay đổi)
# C++/SQL/Java giảm chậm hơn (ít thay đổi)
DECAY_RATES_DAYS = {
    'javascript': 60,
    'typescript': 60,
    'python': 90,
    'go': 120,
    'rust': 120,
    'java': 180,
    'sql': 365,
    'default': 120  # 4 tháng
}

# --- Whitelist / Blacklist / Keywords ---

# Từ khóa có trọng số: cao hơn cho các chủ đề phức tạp hoặc hiện đại
HIGH_VALUE_KEYWORDS = {
    # DevOps & Cloud
    'dockerfile': 3.0, 'kubernetes': 3.0, 'terraform': 2.8, 'aws lambda': 2.5,
    'ci/cd': 2.2, 'github actions': 2.2, 'argocd': 2.7,
    # Backend & API
    'async': 1.8, 'await': 1.5, 'goroutine': 2.0, 'microservice': 2.5,
    'fastapi': 2.0, 'graphql': 2.3, 'grpc': 2.5,
    # Frontend & UI
    'react': 1.5, 'vue': 1.5, 'svelte': 1.8,
    'useEffect': 2.5, 'useState': 2.0, 'nextjs': 2.2, 'redux': 1.8,
    'tailwind': 1.7, 'vite': 1.5,
    # Data & ML
    'pandas': 2.0, 'numpy': 1.8, 'scikitlearn': 2.2, 'pytorch': 2.5, 'tensorflow': 2.5,
    # Python Specific
    'decorator': 2.0, 'contextmanager': 2.2, 'pydantic': 2.0,
    # SQL
    'join': 1.5, 'with': 1.8, 'group by': 1.5, 'window function': 2.5
}

# Các "Contextual Combos": nhân điểm thưởng nếu các cặp từ khóa xuất hiện cùng nhau
CONTEXTUAL_COMBOS = {
    'react_hooks': (['react', 'useEffect'], 2.0),
    'python_async': (['python', 'async def'], 2.5),
    'api_design': (['graphql', 'resolver'], 2.2)
}

# Regex cho các mẫu spam hoặc code chất lượng rất thấp
SPAM_REGEX_BLACKLIST = [
    re.compile(r'\b(hello|world|print|console\.log)\b', re.IGNORECASE),
    re.compile(r'buy now|crypto|forex|free trial|seo services', re.IGNORECASE),
    re.compile(r'(\b\w+\b\s*){1,2}Copyright \d{4}', re.IGNORECASE) # "MyComponent Copyright 2023"
]

# Các TLDs (Top-Level Domains) phổ biến trong spam
SPAM_TLDS = {'.xyz', '.top', '.tk', '.site', '.click', '.loan', '.online'}


# --- HELPER FUNCTIONS ---

def get_base_priority(language: str) -> float:
    """Gets the base score for the language."""
    lang = language.lower()
    return LANGUAGE_BASE_SCORES.get(lang, LANGUAGE_BASE_SCORES['default'])

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

def is_spam_or_trivial(content: str) -> bool:
    """Advanced check for spam, trivial, or obfuscated content."""
    if not content or not content.strip():
        return True
    
    content_to_check = content[:2000].strip()
    
    # Rule 1: Very short content without high-value keywords
    if len(content_to_check) < THRESHOLDS['min_length_for_analysis']:
        content_lower = content_to_check.lower()
        if not any(kw in content_lower for kw in HIGH_VALUE_KEYWORDS):
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
    decay_days = DECAY_RATES_DAYS.get(language.lower(), DECAY_RATES_DAYS['default'])
    
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
        lexer = get_lexer_by_name('plaintext', stripall=False)

    tokens = list(pygments.lex(content, lexer))
    if not tokens:
        return 0.0

    total_tokens = len(tokens)

    # --- Metrics Calculation ---
    significant_token_count = 0
    unique_token_types = set()
    language_specific_bonus = 0.0

    # Define significant token types
    significant_tokens = {
        Token.Keyword, Token.Name.Function, Token.Name.Class, Token.Operator,
        Token.Literal.String.Interpol, Token.Name.Decorator
    }

    token_values = [t[1] for t in tokens]

    for ttype, tvalue in tokens:
        unique_token_types.add(ttype)
        if any(ttype in s for s in significant_tokens):
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
    for keyword, weight in HIGH_VALUE_KEYWORDS.items():
        if keyword in content_lower:
            keyword_score += weight

    # Apply Contextual Combos
    for combo, (terms, multiplier) in CONTEXTUAL_COMBOS.items():
        if all(term in content_lower for term in terms):
            keyword_score *= multiplier

    # Normalize keyword score (e.g., capped at 1.5)
    normalized_keyword_score = min(1.5, keyword_score / 10.0)

    # Final Score Combination
    final_score = (
        (density_score * 0.4) +
        (diversity_score * 0.3) +
        (normalized_keyword_score * 0.3) +
        language_specific_bonus
    )

    return min(1.0, final_score)

def calculate_comment_utility(content: str, language: str) -> float:
    """
    Calculates utility score based on comment quality, readability, and structure.
    Uses Pygments for accurate comment and docstring detection.
    """
    try:
        lexer = get_lexer_by_name(language, stripall=False)
    except pygments.util.ClassNotFound:
        lexer = get_lexer_by_name('plaintext', stripall=False)

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

def calculate_priority(content: str, language: str, created_at: datetime.datetime) -> (float, str):
    """
    Calculates the final priority score (0.1 to 1.0) using the advanced rule-based engine.
    
    Returns: (priority_score, assessment_string)
    """
    lang = language.lower() if language else 'plaintext'
    
    # --- PHASE 1: PRE-CHECK (Spam/Trivial Rule) ---
    if is_spam_or_trivial(content):
        return 0.1, "Assessment: Rejected (Trivial or Spam)."
    
    lang_base_score = get_base_priority(lang)
    content_length = len(content)
    
    # --- PHASE 2: SCORE COMPONENTS ---
    
    # 1. Length Score
    length_score = min(1.0, content_length / THRESHOLDS['optimal_length'])
    
    # 2. Age Decay Score
    age_score = calculate_age_decay(created_at, lang)
    
    # 3. Structural Complexity Score (Code-aware analysis)
    syntax_score = analyze_structural_complexity(content, lang)
    
    # 4. Comment & Readability Utility Score
    utility_score = calculate_comment_utility(content, lang)
    
    # --- PHASE 3: FINAL CALCULATION ---
    
    # Calculate raw score using the new weights
    raw_score = (
        WEIGHTS['length'] * length_score +
        WEIGHTS['syntax'] * syntax_score +
        WEIGHTS['utility'] * utility_score +
        WEIGHTS['age'] * age_score
    )
    
    # The final priority is a blend of the language's base score and the calculated raw score.
    # This ensures that good code in a low-priority language is still ranked lower than
    # exceptional code in a high-priority language.
    final_priority = lang_base_score + (1.0 - lang_base_score) * raw_score
    
    # Ensure the score is within the valid range [0.1, 1.0]
    final_priority = max(0.1, min(1.0, final_priority))
    
    # --- ASSESSMENT STRING ---
    assessment_details = (
        f"Length:{length_score:.2f} (w:{WEIGHTS['length']}), "
        f"Syntax:{syntax_score:.2f} (w:{WEIGHTS['syntax']}), "
        f"Utility:{utility_score:.2f} (w:{WEIGHTS['utility']}), "
        f"Age:{age_score:.2f} (w:{WEIGHTS['age']})"
    )
    assessment_string = f"Priority={final_priority:.3f} | Details: {assessment_details}"
    
    return final_priority, assessment_string