import re
import datetime
from collections import Counter

# --- Cấu hình Trọng số (Weights) và Ngưỡng (Thresholds) ---
WEIGHTS = {
    'length': 0.35,          # Trọng số cho Độ dài
    'syntax': 0.30,          # Trọng số cho Mật độ cú pháp/đa dạng
    'utility': 0.25,         # Trọng số cho Độ hữu dụng (Comment/Line Length)
    'age': 0.10,             # Trọng số cho Độ mới (Age decay)
    'base_score': 0.20       # Điểm cơ sở tối thiểu cho code không phải spam
}

# Ngưỡng và Giá trị tối ưu
THRESHOLDS = {
    'optimal_length': 4000,  # Chiều dài tối ưu cho snippet (tính bằng ký tự)
    'min_length_for_analysis': 100, # Độ dài tối thiểu để bắt đầu phân tích sâu
    'min_lines': 5,          # Số dòng tối thiểu cho code phức tạp
    'max_url_density': 0.10, # Mật độ URL tối đa (URL chars / total chars)
    'n_gram_size': 3,        # Kích thước N-Gram cho phân tích đa dạng
    'optimal_comment_ratio_min': 0.10, # Tỷ lệ comment tối ưu (min)
    'optimal_comment_ratio_max': 0.35, # Tỷ lệ comment tối ưu (max)
    'line_length_penalty_threshold': 40 # Độ dài dòng trung bình tối thiểu (tránh code dòng đơn)
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

# --- Whitelist / Blacklist ---
# Các từ khóa cao cấp (giúp các snippet ngắn nhưng chất lượng cao đạt điểm)
HIGH_VALUE_KEYWORDS = {
    'dockerfile', 'kubernetes', 'aws lambda', 'ci/cd', 'async', 'microservice',
    'decorator', 'useEffect', 'useState', 'nextjs', 'react hook', 'tailwind'
}

# Các TLDs (Top-Level Domains) phổ biến trong spam
SPAM_TLDS = {'.xyz', '.top', '.tk', '.site', '.click', '.loan', '.online'}


# --- HELPER FUNCTIONS ---

def get_base_priority(language: str) -> float:
    """Gets the base score for the language."""
    lang = language.lower()
    return LANGUAGE_BASE_SCORES.get(lang, LANGUAGE_BASE_SCORES['default'])

def is_spam_or_trivial(content: str) -> bool:
    """Basic check for highly repetitive or trivial content (pre-analysis)."""
    if not content or not content.strip(): return True
    
    content_to_check = content[:2000].strip()
    
    # Rule: Very short content
    if len(content_to_check) < 50:
        if len(content_to_check.splitlines()) <= 3:
            return True
        # Allow if it contains a high-value keyword (e.g., a short React hook snippet)
        if not any(kw in content_to_check for kw in HIGH_VALUE_KEYWORDS):
             if len(content_to_check) < 100:
                return True

    # Rule: Highly repetitive characters
    char_counts = Counter(c for c in content_to_check if not c.isspace())
    total_non_space = sum(char_counts.values())
    if char_counts and total_non_space > 30:
        most_common_char, most_common_count = char_counts.most_common(1)[0]
        threshold = 0.7 if most_common_char.isalnum() else 0.9
        if most_common_count / total_non_space > threshold:
            return True

    # Rule: Excessive URLs and TLD spam check
    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    urls = re.findall(url_pattern, content_to_check)
    if len(urls) > 5:
        return True # Too many URLs

    for url in urls:
        if any(url.endswith(tld) for tld in SPAM_TLDS):
            return True # Contains spam TLD

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


def calculate_n_gram_diversity(content: str, n: int) -> float:
    """Calculates diversity based on unique N-Grams of words."""
    words = re.findall(r'\b\w+\b', content.lower())
    if len(words) < n:
        return 0.0 # Không đủ từ
    
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    
    if not ngrams:
        return 0.0

    diversity = len(set(ngrams)) / len(ngrams)
    return min(1.0, diversity) # Đảm bảo không vượt quá 1.0

def calculate_comment_utility(content: str, language: str) -> float:
    """Calculates score based on Comment/Code ratio and line length."""
    lines = content.splitlines()
    total_lines = len(lines)
    code_lines = 0
    comment_lines = 0
    total_line_length = 0
    
    comment_markers = {
        'python': ['#'], 'javascript': ['//', '/*'], 'typescript': ['//', '/*'],
        'java': ['//', '/*'], 'csharp': ['//', '/*'], 'html': ['<!--'],
        'css': ['/*', '//'], 'sql': ['--', '/*']
    }
    markers = comment_markers.get(language.lower(), ['#', '//'])

    if total_lines == 0:
        return 0.1

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
            
        is_comment = False
        for marker in markers:
            if stripped_line.startswith(marker):
                is_comment = True
                break
        
        if is_comment:
            comment_lines += 1
        else:
            code_lines += 1
            total_line_length += len(stripped_line)
            
    # Tính Tỷ lệ Comment
    total_active_lines = code_lines + comment_lines
    if total_active_lines == 0:
        return 0.1 # Vẫn là trivial nếu không có dòng code/comment nào

    comment_ratio = comment_lines / total_active_lines

    # 1. Điểm Tỷ lệ Comment (Peak around optimal range)
    ratio_score = 0.0
    min_ratio = THRESHOLDS['optimal_comment_ratio_min']
    max_ratio = THRESHOLDS['optimal_comment_ratio_max']
    
    if min_ratio <= comment_ratio <= max_ratio:
        ratio_score = 1.0 # Điểm tuyệt đối nếu nằm trong vùng tối ưu
    elif comment_ratio < min_ratio:
        # Phạt nếu quá ít comment
        ratio_score = comment_ratio / min_ratio * 0.8
    else:
        # Phạt nếu quá nhiều comment (thường là snippet dài)
        ratio_score = max(0.5, 1.0 - (comment_ratio - max_ratio) * 1.5)

    # 2. Phạt Độ dài dòng (tránh code dòng đơn)
    line_avg = (total_line_length / code_lines) if code_lines > 0 else 0
    length_penalty = 1.0
    
    if code_lines > THRESHOLDS['min_lines'] and line_avg < THRESHOLDS['line_length_penalty_threshold']:
        # Phạt nếu code dài nhưng độ dài dòng quá ngắn
        length_penalty = max(0.5, line_avg / THRESHOLDS['line_length_penalty_threshold'])

    return ratio_score * 0.7 + length_penalty * 0.3


# --- MAIN CALCULATION FUNCTION ---

def calculate_priority(content: str, language: str, created_at: datetime.datetime) -> (float, str):
    """
    Calculates the Rule-Based Priority Score (0.1 to 1.0) and an assessment string.
    
    Returns: (priority_score, assessment_string)
    """
    lang = language or 'plaintext'
    lang_base_score = get_base_priority(lang)
    
    # --- PHASE 1: PRE-CHECK (Spam/Trivial Rule) ---
    if is_spam_or_trivial(content):
        return 0.1, "Rule-based: Highly trivial or suspected spam content."
    
    content_length = len(content)
    lines = content.splitlines()
    total_code_lines = len([line.strip() for line in lines if line.strip()])
    
    # Nếu nội dung quá ngắn nhưng không bị đánh dấu là spam/trivial, vẫn cho điểm thấp
    if content_length < THRESHOLDS['min_length_for_analysis']:
        # Ngoại lệ nếu có keyword giá trị cao
        if not any(kw in content for kw in HIGH_VALUE_KEYWORDS):
            return max(lang_base_score * 0.4, 0.2), "Rule-based: Very short content, minimal priority."
        
    # --- PHASE 2: SCORE COMPONENTS ---
    
    # 1. Score Độ dài (Length Score)
    # Tăng dần đến ngưỡng tối ưu, sau đó giữ nguyên. max(1.0)
    length_score = min(1.0, content_length / THRESHOLDS['optimal_length'])
    
    # 2. Score Độ mới (Age Decay Score)
    age_score = calculate_age_decay(created_at, lang)
    
    # 3. Score Mật độ/Đa dạng (Syntax/Diversity Score) - Phase 1
    # Dùng N-Gram Diversity làm proxy cho tính độc đáo
    diversity_score = calculate_n_gram_diversity(content, THRESHOLDS['n_gram_size'])
    # Tăng cường nếu có keyword cao cấp (bonus lên tới 0.3 điểm)
    keyword_bonus = 0.0
    if total_code_lines > 0:
        found_keywords = {word for word in re.findall(r'\b\w+\b', content) if word in HIGH_VALUE_KEYWORDS}
        keyword_bonus = min(0.3, len(found_keywords) * 0.05) # Max 6 keywords
        
    syntax_score = min(1.0, diversity_score * 0.7 + keyword_bonus * 0.3)
    
    # 4. Score Độ hữu dụng (Utility Score) - Phase 2
    utility_score = calculate_comment_utility(content, lang)
    
    # --- PHASE 3: FINAL CALCULATION ---
    
    # Tính điểm thô (Raw Score)
    raw_score = (
        WEIGHTS['length'] * length_score +
        WEIGHTS['syntax'] * syntax_score +
        WEIGHTS['utility'] * utility_score +
        WEIGHTS['age'] * age_score
    )
    
    # Áp dụng Điểm Cơ sở của ngôn ngữ
    # Final Priority = Điểm Cơ sở * Điểm Thô (để điểm tối đa không vượt quá 1.0)
    final_priority = lang_base_score + (1.0 - lang_base_score) * raw_score
    
    # Đảm bảo điểm nằm trong phạm vi 0.1 đến 1.0
    final_priority = max(0.1, min(1.0, final_priority))
    
    # --- ASSESSMENT STRING ---
    
    # Tổng hợp chi tiết để dễ debug và theo dõi
    assessment_details = (
        f"Length:{length_score:.2f} (x{WEIGHTS['length']:.2f}), "
        f"Diversity:{syntax_score:.2f} (x{WEIGHTS['syntax']:.2f}), "
        f"Utility:{utility_score:.2f} (x{WEIGHTS['utility']:.2f}), "
        f"Age:{age_score:.2f} (x{WEIGHTS['age']:.2f})"
    )
    assessment_string = f"Rule-based: Priority={final_priority:.2f}. Details: {assessment_details}"
    
    return final_priority, assessment_string