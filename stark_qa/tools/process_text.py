import string
import re
import codecs
from collections import Counter
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from langchain.text_splitter import RecursiveCharacterTextSplitter

def compact_text(text):
    """
    Compact the text by removing unnecessary spaces and punctuation issues.

    Args:
        text (str): Input text to be compacted.

    Returns:
        str: Compacted text.
    """
    text = text.replace("\n", ". ").replace("\r", "")
    text = text.replace("- ", "")
    text = text.replace(": .", ":").replace(":.", ":")
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace(".. ", ". ")

    return text

def remove_punctuation(text):
    """
    Remove all punctuation from the given text.

    Args:
        text (str): Input text from which punctuation will be removed.

    Returns:
        str: Text without punctuation.
    """
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


def clean_data(item):
    """
    Clean the text data.

    Args:
        item (Union[str, list, dict]): An object that contains text data which is cleaned iteratively.

    Returns:
        The cleaned data in the same format as item.
    """
    if isinstance(item, str):
        item = ' '.join(BeautifulSoup(item, "lxml").text.split())
    elif isinstance(item, list):
        item = [clean_data(i) for i in item]
    elif isinstance(item, dict):
        item = {remove_punctuation(clean_data(k).lower()).replace(' ', '_'): clean_data(i) for k, i in item.items()}
    return item


def chunk_text(text, chunk_size):
    """
    Split text into chunks of specified size.

    Args:
        text (str): Input text to be chunked.
        chunk_size (int): Size of each chunk.

    Returns:
        list: List of text chunks.
    """
    custom_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 5,
        length_function=len
    )
    texts = custom_text_splitter.create_documents([text])
    chunks = [text.page_content for text in texts]
    return chunks


def clean_dict(dictionary, remove_values=['', 'nan']):
    """
    Clean the dictionary by removing specific values.

    Args:
        dictionary (dict): A dictionary to be cleaned.
        remove_values (list): List of values to remove from the dictionary.

    Returns:
        dict: Cleaned dictionary.
    """
    new_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            new_dict[k] = clean_dict(v, remove_values)
        elif str(v) in remove_values:
            pass
        else:
            new_dict[k] = v
    return new_dict


def normalize_answer(s):
    """
    Normalize text by removing punctuation, articles and extra whitespace, and lowercasing the text.

    Args:
        s (str): Input text to be normalized.

    Returns:
        str: Normalized text.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def recall_score(prediction, ground_truth):
    """
    Calculate the recall score between prediction and ground truth.

    Args:
        prediction (str): Predicted text.
        ground_truth (str): Ground truth text.

    Returns:
        float: Recall score.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return recall


def f1_score(prediction, ground_truth):
    """
    Calculate the F1 score between prediction and ground truth.

    Args:
        prediction (str): Predicted text.
        ground_truth (str): Ground truth text.

    Returns:
        float: F1 score.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """
    Calculate the exact match score between prediction and ground truth.

    Args:
        prediction (str): Predicted text.
        ground_truth (str): Ground truth text.

    Returns:
        float: Exact match score.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# Pluralization and Synonym extraction

ABERRANT_PLURAL_MAP = {
    'appendix': 'appendices',
    'barracks': 'barracks',
    'cactus': 'cacti',
    'child': 'children',
    'criterion': 'criteria',
    'deer': 'deer',
    'echo': 'echoes',
    'elf': 'elves',
    'embargo': 'embargoes',
    'focus': 'foci',
    'fungus': 'fungi',
    'goose': 'geese',
    'hero': 'heroes',
    'hoof': 'hooves',
    'index': 'indices',
    'knife': 'knives',
    'leaf': 'leaves',
    'life': 'lives',
    'man': 'men',
    'mouse': 'mice',
    'nucleus': 'nuclei',
    'person': 'people',
    'phenomenon': 'phenomena',
    'potato': 'potatoes',
    'self': 'selves',
    'syllabus': 'syllabi',
    'tomato': 'tomatoes',
    'torpedo': 'torpedoes',
    'veto': 'vetoes',
    'woman': 'women',
}

VOWELS = set('aeiou')


def synonym_extractor(phrase):
    """
    Extract synonyms for a given phrase using WordNet.

    Args:
        phrase (str): Input phrase to find synonyms for.

    Returns:
        list: List of synonyms.
    """
    synonyms = []
    for syn in wordnet.synsets(phrase):
        if '.n.' in syn.name():
            for l in syn.lemmas():
                synonyms.append(l.name())
    return list(set(synonyms))


def pluralize(singular):
    """
    Return the plural form of a given lowercase singular word (English only).

    Args:
        singular (str): Singular word.

    Returns:
        str: Plural form of the word.
    """
    if not singular:
        return ''
    plural = ABERRANT_PLURAL_MAP.get(singular)
    if plural:
        return plural
    root = singular
    try:
        if singular[-1] == 'y' and singular[-2] not in VOWELS:
            root = singular[:-1]
            suffix = 'ies'
        elif singular[-1] == 's':
            if singular[-2] in VOWELS:
                if singular[-3:] == 'ius':
                    root = singular[:-2]
                    suffix = 'i'
                else:
                    root = singular[:-1]
                    suffix = 'ses'
            else:
                suffix = 'es'
        elif singular[-2:] in ('ch', 'sh'):
            suffix = 'es'
        else:
            suffix = 's'
    except IndexError:
        suffix = 's'
    plural = root + suffix
    return plural


def decode_escapes(s):
    """
    Decode escape sequences in a string.

    Args:
        s (str): Input string with escape sequences.

    Returns:
        str: Decoded string.
    """
    ESCAPE_SEQUENCE_RE = re.compile(r'''
        ( \\U........      # 8-digit hex escapes
        | \\u....          # 4-digit hex escapes
        | \\x..            # 2-digit hex escapes
        | \\[0-7]{1,3}     # Octal escapes
        | \\N\{[^}]+\}     # Unicode characters by name
        | \\[\\'"abfnrtv]  # Single-character escapes
        )''', re.UNICODE | re.VERBOSE)

    def decode_match(match):
        return codecs.decode(match.group(0), 'unicode-escape')

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)

