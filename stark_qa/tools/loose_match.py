import re

def convert_wildcard_to_regex(pattern):
    # Escape special regex characters and replace '*' with '.*' for suffix matching
    return re.escape(pattern).replace(r'\*', '.*')

def loose_match(input_string, patterns):
    patterns = [convert_wildcard_to_regex(p) for p in patterns]
    
    # Check if input_string matches any of the patterns
    for pattern in patterns:
        if re.fullmatch(pattern, input_string):
            return True
    return False