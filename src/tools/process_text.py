from bs4 import BeautifulSoup
import string
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from collections import Counter


import codecs
import re
import codecs


def compact_text(text):
    text = text.replace("\n", ". ").replace("- ", "")
    text = text.replace(": .", ":").replace(":.", ":")
    text = text.replace("  ", " ")
    text = text.replace(".. ", ". ")
    return text


def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


def clean_data(item):
    '''
    clean the text data
    Args:
        item (Union[str, list, dict]): An object that contains text data which is cleaned iteratively
    Return: 
        the cleaned data in the same format as item
    '''
    if isinstance(item, str):
        item = ' '.join(BeautifulSoup(item, "lxml").text.split())
    elif isinstance(item, list):
        item = [clean_data(i) for i in item]
    elif isinstance(item, dict):
        item = {remove_punctuation(clean_data(k).lower()).replace(' ', '_'): clean_data(i) for k, i in item.items()}
    return item


def chunk_text(text, chunk_size):

    custom_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap  = chunk_size // 5,
        # Use length of the text as the size measure
        length_function = len
    )

    # Create the chunks
    texts = custom_text_splitter.create_documents([text])
    chunks = [text.page_content for text in texts]
    return chunks


def clean_dict(dictionary: dict, remove_values=['', 'nan']) -> dict:
    '''
    Clean the dictionary by removing specific values
    Args:
        dictionary (dict): a dictionary
    '''
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
    """Lower text and remove punctuation, articles and extra whitespace."""
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
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return recall


def f1_score(prediction, ground_truth):
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
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# from https://code.activestate.com/recipes/577781-pluralize-word-convert-singular-word-to-its-plural/
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

import nltk
from nltk.corpus import wordnet

def synonym_extractor(phrase):
    synonyms = []

    for syn in wordnet.synsets(phrase):
        if '.n.' in syn.name():
            for l in syn.lemmas():
                synonyms.append(l.name())
    return list(set(synonyms))


def pluralize(singular):
    """Return plural form of given lowercase singular word (English only). Based on
    ActiveState recipe http://code.activestate.com/recipes/413172/
    
    >>> pluralize('')
    ''
    >>> pluralize('goose')
    'geese'
    >>> pluralize('dolly')
    'dollies'
    >>> pluralize('genius')
    'genii'
    >>> pluralize('jones')
    'joneses'
    >>> pluralize('pass')
    'passes'
    >>> pluralize('zero')
    'zeros'
    >>> pluralize('casino')
    'casinos'
    >>> pluralize('hero')
    'heroes'
    >>> pluralize('church')
    'churches'
    >>> pluralize('x')
    'xs'
    >>> pluralize('car')
    'cars'

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


if __name__ == '__main__':
    print(chunk_text("Based on the given product information, you need to (1) identify the product's generic category, (2) list all of the negative perspectives and their sources, and (2) extract up to five hard and five soft requirements relevant to customers' interests along with their sources. (1) For example, the product's generic category can be ", 100))
    print(normalize_answer("Sparkling White Smiles Professional Sport Mouth Guards"))
    print(normalize_answer("I also got a 2-pack <Sparkling White "))
    print(f1_score(normalize_answer("Professional Sport Mouth Guards Sparkling White Smiles haha"), normalize_answer("Sparkling White Smiles Professional Sport Mouth Guards")))
    print(recall_score(normalize_answer("Professional Sport Mouth Guards Sparkling White Smiles haha"), normalize_answer("Sparkling White Smiles Professional Sport Mouth Guards")))