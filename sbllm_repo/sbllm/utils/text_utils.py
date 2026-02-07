
import re

def clean_text(text: str) -> str:
    """
    Remove CJK (Chinese, Japanese, Korean) characters from text.
    
    Args:
        text: Input string
        
    Returns:
        Cleaned string with CJK characters removed
    """
    if not text:
        return ""
    text = re.sub('[\u4e00-\u9fa5]+', '', text)
    text = re.sub('[\u3040-\u309F]+', '', text)
    text = re.sub('[\u30A0-\u30FF]+', '', text)
    text = re.sub('[\uAC00-\uD7A3]+', '', text)
    return text
