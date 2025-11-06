import re
from collections import Counter
import logging
import datetime
import math 

# --- Cấu hình Logger ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Định nghĩa các tham số quy tắc ---
# Trọng số cho các yếu tố khác nhau (Điều chỉnh Phase 2)
WEIGHT_LENGTH = 0.20 # Giảm trọng số Length
WEIGHT_SYNTAX = 0.25
WEIGHT_DIVERSITY = 0.20 
WEIGHT_KEYWORD_RATIO = 0.10
WEIGHT_UTILITY = 0.15 # Trọng số mới cho Comment Density/Line Length Penalty
WEIGHT_AGE = 0.10 # Giữ nguyên trọng số Age

# Độ dài tối ưu để đạt điểm 1.0 (ví dụ: 3000 ký tự)
OPTIMAL_LENGTH = 3000 
# Điểm cơ sở cho các loại ngôn ngữ (Language base scores)
LANGUAGE_BASE_SCORES = {
    'python': 0.85, 'javascript': 0.85, 'typescript': 0.90,
    'go': 0.75, 'rust': 0.75, 'html': 0.65,
    'css': 0.50, 'sql': 0.60, 'json': 0.30,
    'xml': 0.30, 'plaintext': 0.15, 'markdown': 0.20,
    'default': 0.50 
}
# Các ký tự cú pháp quan trọng để tính "Mật độ cú pháp"
SYNTAX_CHARS = ['{', '}', '(', ')', '[', ']', ';', ':', '=', '>', '<', '|', '&', '*', '+', '-']

# Danh sách TLD thường bị spam
SPAM_TLDS = ['.xyz', '.top', '.tk', '.cf', '.gq', '.ml', '.ga', '.biz']
# Whitelist keywords
QUALITY_KEYWORDS = {'async', 'await', 'useEffect', 'useState', 'component', 'class', 
                    'def', 'import', 'router', 'database', 'migration', 'benchmark', 
                    'production', 'CI/CD', 'teardown'}

# Các từ khóa/Hàm gọi để tính Tỷ lệ Keyword
CODE_KEYWORDS = {'def', 'class', 'function', 'import', 'const', 'let', 'var', 'struct', 'if', 'while', 'for'}
API_CALL_TERMS = {'requests', 'fetch', 'axios', 'subprocess', 'os.', 'fs.', 'db.', 'query', 'console.log', 'print'}

# Tham số Decay theo ngôn ngữ (Phase 2: MỚI)
# Giá trị là số ngày sau đó điểm AGE sẽ giảm về 0.0
DECAY_RATE_DAYS = {
    'python': 60, # Code Python/JS thường lỗi thời nhanh hơn
    'javascript': 60,
    'typescript': 60,
    'html': 90,
    'css': 90,
    'sql': 180,
    'go': 120,
    'rust': 120,
    'default': 90 # 3 tháng
}
MIN_AGE_SCORE = 0.1 # Điểm AGE tối thiểu để snippet không hoàn toàn bị loại bỏ vì tuổi


# --- HÀM HELPER MỚI (Phase 2) ---

def _calculate_utility_score(content: str, lang: str) -> dict:
    """
    Tính điểm Hữu dụng (Comment Density & Average Line Length Penalty).
    Trả về dict {'score': float, 'avg_line_len': float, 'comment_ratio': float}
    """
    lines = content.splitlines()
    code_lines = []
    comment_lines = 0
    total_code_chars = 0
    
    # Simple comment patterns (tùy thuộc vào ngôn ngữ)
    if lang in ['python']:
        comment_pattern = r'^\s*#|^\s*"""'
    elif lang in ['javascript', 'typescript', 'go', 'rust', 'c++']:
        comment_pattern = r'^\s*//|^\s*/\*'
    elif lang in ['html', 'xml']:
        comment_pattern = r'^\s*<!--'
    elif lang in ['css']:
        comment_pattern = r'^\s*/\*'
    else:
        comment_pattern = r'^\s*#' # Mặc định là #

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        
        # Kiểm tra comment (rất đơn giản, chỉ kiểm tra đầu dòng)
        if re.match(comment_pattern, stripped_line, re.IGNORECASE) or stripped_line.startswith('"""') or stripped_line.startswith("'''"):
            comment_lines += 1
        else:
            code_lines.append(stripped_line)
            total_code_chars += len(stripped_line)

    total_lines = len(lines)
    active_code_lines = len(code_lines)
    
    # 1. Comment Density Score (Phase 2: Comment/Code Ratio)
    # Mục tiêu: ưu tiên tỷ lệ comment hợp lý (ví dụ: 10% - 30%)
    if active_code_lines > 0:
        comment_ratio = comment_lines / active_code_lines
        # Ngưỡng tối ưu (target): 0.2 (20% comment)
        OPTIMAL_RATIO = 0.2
        # Giả sử tỷ lệ 0.0 hoặc > 0.6 bị phạt nặng
        # Tính khoảng cách từ OPTIMAL_RATIO
        ratio_diff = abs(comment_ratio - OPTIMAL_RATIO)
        # Chuẩn hoá điểm: max(1 - (khoảng cách / OPTIMAL_RATIO) ^ 2, 0)
        # Sử dụng hàm mũ để phạt nặng hơn nếu lệch xa
        comment_score = max(0.0, 1 - math.pow(ratio_diff / OPTIMAL_RATIO, 2))
    else:
        comment_ratio = 0.0
        comment_score = 0.0
        
    # 2. Average Line Length Penalty (Phase 2: Phạt Code dòng đơn)
    if active_code_lines > 0:
        avg_line_len = total_code_chars / active_code_lines
        # Phạt nếu độ dài dòng trung bình quá ngắn (dưới 50 ký tự)
        LINE_LEN_THRESHOLD = 50
        if avg_line_len < LINE_LEN_THRESHOLD:
            # Penalty: Giảm điểm tuyến tính khi avg_line_len tiến về 0
            len_penalty = 1.0 - (avg_line_len / LINE_LEN_THRESHOLD)
            line_len_score = max(0.0, 1.0 - len_penalty) # Điểm thấp
        else:
            line_len_score = 1.0 # Điểm cao
    else:
        avg_line_len = 0.0
        line_len_score = 0.0
        
    # Kết hợp: Lấy giá trị trung bình nhân (geometric mean) để đảm bảo cả hai đều phải tốt
    # Đảm bảo điểm kết hợp luôn là float > 0.0 để tránh lỗi math.pow
    combined_score = math.pow(comment_score * line_len_score, 0.5)

    return {
        'score': combined_score,
        'avg_line_len': avg_line_len,
        'comment_ratio': comment_ratio
    }

def _calculate_n_gram_diversity(content: str, n=2) -> float:
    """Tính tỷ lệ N-Gram (cặp từ) duy nhất trên tổng số N-Gram."""
    words = re.findall(r'\b\b', content.lower())
    if len(words) < n:
        return 0.0
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i:i+n]))
    
    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))
    
    diversity_ratio = unique_ngrams / total_ngrams
    return math.pow(diversity_ratio, 0.5) 

def _calculate_keyword_ratio(content: str) -> float:
    """Tính tỷ lệ giữa từ khóa cú pháp và từ khóa API/hàm gọi."""
    all_words = re.findall(r'\b\w+\b', content.lower())
    if not all_words: return 0.0

    code_keyword_count = sum(1 for word in all_words if word in CODE_KEYWORDS)
    api_call_count = sum(1 for word in all_words if word in API_CALL_TERMS)
    
    total_relevant_keywords = code_keyword_count + api_call_count
    if total_relevant_keywords == 0: return 0.0
    
    ratio = api_call_count / total_relevant_keywords
    
    return min(1.0, ratio / 0.3)


def is_spam_or_trivial(content: str, lang: str = 'plaintext', max_lines_check=30, max_chars_check=1500) -> dict:
    """
    Kiểm tra xem nội dung có phải là spam/trivial dựa trên các quy tắc tĩnh hay không.
    (Giữ nguyên từ Phase 1)
    """
    if not content or not content.strip(): 
        return {'is_spam': True, 'reason': 'EMPTY_CONTENT'}

    lines = content.splitlines()
    content_to_check = content[:max_chars_check].strip()
    lines_to_check = [line.strip() for line in lines[:max_lines_check] if line.strip()]

    # Rule 1: Very short snippets
    if len(lines_to_check) <= 2 and len(content_to_check) < 50: 
        if any(keyword in content_to_check for keyword in QUALITY_KEYWORDS):
             logger.debug(f"Snippet passed short check due to quality keyword.")
             pass
        else:
             return {'is_spam': True, 'reason': 'VERY_SHORT_LINES'}
             
    if len(lines_to_check) <= 3 and len(content_to_check) < 100:
        lower_content = content_to_check.lower()
        if "hello world" in lower_content: 
            return {'is_spam': True, 'reason': 'HELLO_WORLD'}
        simple_output_pattern = r'^\s*(print|console\.log|echo|write|alert|puts|fmt\.Println|System\.out\.print)\b.*$'
        if lines_to_check and all(re.match(simple_output_pattern, line.lower()) for line in lines_to_check):
            return {'is_spam': True, 'reason': 'SIMPLE_OUTPUT_ONLY'}

    # Rule 2: Highly repetitive characters
    char_counts = Counter(c for c in content_to_check if not c.isspace())
    total_non_space = sum(char_counts.values())
    if char_counts and total_non_space > 30:
        most_common_char, most_common_count = char_counts.most_common(1)[0]
        allowed_high_rep_chars = ['*', '-', '_', '=', '#', '/', '\\', ' ', '\t']
        threshold = 0.9 if most_common_char in allowed_high_rep_chars else 0.7
        if most_common_count / total_non_space > threshold:
            return {'is_spam': True, 'reason': 'HIGHLY_REPETITIVE_CHARS'}

    # Rule 3: Highly repetitive lines
    if len(lines_to_check) > 5:
        line_counts = Counter(lines_to_check)
        most_common_line, most_common_line_count = line_counts.most_common(1)[0]
        if most_common_line and most_common_line_count / len(lines_to_check) > 0.6: 
            return {'is_spam': True, 'reason': 'HIGHLY_REPETITIVE_LINES'}

    # Rule 4: Excessive URLs and Blacklisted TLDs 
    url_pattern = r'https?://[^\s/$.?#].[^\s]*' 
    urls = re.findall(url_pattern, content_to_check)
    
    for url in urls:
        for tld in SPAM_TLDS:
            if url.endswith(tld):
                return {'is_spam': True, 'reason': f'BLACKLISTED_TLD:{tld}'}

    if len(urls) > 3 or (len(content_to_check) > 50 and len(lines_to_check)>0 and len(urls) / len(lines_to_check) > 0.3): 
        return {'is_spam': True, 'reason': 'EXCESSIVE_URLS'}

    return {'is_spam': False, 'reason': 'NOT_SPAM'}


def calculate_priority(content: str, language: str, created_at: datetime.datetime, is_verified: bool = False) -> dict:
    """
    Tính toán điểm Priority (0.1 - 1.0) dựa trên quy tắc.
    """
    # 1. Xử lý trường hợp đặc biệt (Verified)
    if is_verified:
        return {
            'priority': 1.0, 
            'assessment': 'Snippet Verified, assigned maximum priority.'
        }

    # 2. Kiểm tra Spam/Trivial (Quyết định điểm sàn)
    spam_check = is_spam_or_trivial(content, language)
    if spam_check['is_spam']:
        return {
            'priority': 0.1, 
            'assessment': f"Rule-based: Trivial/Spam content detected ({spam_check['reason']})."
        }

    # --- Chuẩn hoá dữ liệu ---
    normalized_lang = language.lower()
    content_length = len(content)
    now = datetime.datetime.now(datetime.timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=datetime.timezone.utc)

    # --- 3. Tính Điểm Độ Dài (LENGTH Score) ---
    length_score = min(1.0, content_length / OPTIMAL_LENGTH)

    # --- 4. Tính Điểm Mật độ Cú pháp (SYNTAX Score) ---
    total_chars = content_length
    syntax_char_count = sum(content.count(char) for char in SYNTAX_CHARS)
    
    if total_chars > 100:
        syntax_density = syntax_char_count / total_chars
        target_max_density = 0.05 
        syntax_score = min(1.0, syntax_density / target_max_density)
    else:
        syntax_score = 0.1 

    # --- 5. Tính Điểm Đa dạng N-Gram (DIVERSITY Score) ---
    diversity_score = _calculate_n_gram_diversity(content)

    # --- 6. Tính Điểm Tỷ lệ Keyword (KEYWORD RATIO Score) ---
    keyword_ratio_score = _calculate_keyword_ratio(content)
    
    # --- 7. Tính Điểm Hữu dụng UTILITY Score (Phase 2: MỚI) ---
    utility_metrics = _calculate_utility_score(content, normalized_lang)
    utility_score = utility_metrics['score']

    # --- 8. Tính Điểm Thời gian (AGE Score) ---
    # Phase 2: Dùng DECAY_RATE_DAYS
    time_delta = now - created_at
    decay_days = DECAY_RATE_DAYS.get(normalized_lang, DECAY_RATE_DAYS['default'])
    decay_seconds = decay_days * 86400 # 86400 giây/ngày

    age_score_raw = 1.0 - (time_delta.total_seconds() / decay_seconds)
    age_score = max(MIN_AGE_SCORE, min(1.0, age_score_raw)) # Kẹp giá trị

    # --- 9. Tính Điểm Cơ sở Ngôn ngữ (LANGUAGE Base Score) ---
    language_base = LANGUAGE_BASE_SCORES.get(normalized_lang, LANGUAGE_BASE_SCORES['default'])
    
    # --- 10. Kết hợp các điểm (COMBINED Score) ---
    # Phải đảm bảo tổng trọng số bằng 1 (0.20+0.25+0.20+0.10+0.15+0.10 = 1.0)
    combined_score = (
        (WEIGHT_LENGTH * length_score) + 
        (WEIGHT_SYNTAX * syntax_score) + 
        (WEIGHT_DIVERSITY * diversity_score) + 
        (WEIGHT_KEYWORD_RATIO * keyword_ratio_score) + 
        (WEIGHT_UTILITY * utility_score) + # Thêm Utility Score
        (WEIGHT_AGE * age_score)
    )
    
    final_priority_raw = combined_score * language_base
    
    # Đảm bảo điểm nằm trong khoảng 0.2 đến 1.0
    final_priority = max(0.2, min(1.0, round(final_priority_raw, 2)))

    # --- 11. Tạo Assessment (Cập nhật phiên bản và thêm chi tiết mới) ---
    assessment = (
        f"Rule-based (v1.2): Priority={final_priority:.2f}. "
        f"Details: Len={length_score:.2f}, Syn={syntax_score:.2f}, "
        f"Div={diversity_score:.2f}, Key={keyword_ratio_score:.2f}, "
        f"Util={utility_score:.2f} (LineLenAvg={utility_metrics['avg_line_len']:.1f}, CommentRatio={utility_metrics['comment_ratio']:.2f}), "
        f"Age={age_score:.2f} ({decay_days}d), Base={language_base:.2f} ({language})."
    )

    return {
        'priority': final_priority,
        'assessment': assessment
    }

__all__ = ['calculate_priority']