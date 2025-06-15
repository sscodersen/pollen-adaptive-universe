
from typing import List

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that',
    'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
    'may', 'might'
}

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    words = text.lower().split()
    keywords = []
    for word in words:
        word_clean = ''.join(c for c in word if c.isalnum())
        if len(word_clean) > 2 and word_clean not in STOP_WORDS:
            keywords.append(word_clean)
    return keywords[:max_keywords]
