""" from https://github.com/keithito/tacotron """
from src.data.components.text import cleaners
from src.data.components.text.symbols import symbols
from src.data.components.text.symbols_korean import KOR_SYMBOLS

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

Kchar_to_id={c: i for i, c in enumerate(KOR_SYMBOLS)}
id_to_Kchar={i: c for i, c in enumerate(KOR_SYMBOLS)}

def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    clean_text = _clean_text(text, cleaner_names)
    for symbol in clean_text:
        if symbol in _symbol_to_id.keys():
            symbol_id = _symbol_to_id[symbol]
            sequence += [symbol_id]
        else:
            continue
    return sequence


def cleaned_text_to_sequence(cleaned_text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    if 'korean_cleaners' in cleaner_names:
        for symbol in cleaned_text:
            if symbol in Kchar_to_id.keys():
                symbol_id = Kchar_to_id[symbol]
                sequence += [symbol_id]
            else:
                continue
        return sequence
    else:
        for symbol in cleaned_text:
            if symbol in _symbol_to_id.keys():
                symbol_id = _symbol_to_id[symbol]
                sequence += [symbol_id]
            else:
                continue
    return sequence


def sequence_to_text(sequence, cleaner_names):
    """Converts a sequence of IDs back to a string"""
    result = ""
    if 'korean_cleaners' in cleaner_names:
        for symbol_id in sequence:
            s = id_to_Kchar[symbol_id]
            result += s
    else:
        for symbol_id in sequence:
            s = _id_to_symbol[symbol_id]
            result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text