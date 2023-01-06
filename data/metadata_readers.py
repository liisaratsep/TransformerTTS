import sys
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def get_preprocessor_by_name(name: str):
    """
    Returns the respective data function.
    Taken from https://github.com/mozilla/TTS/blob/master/TTS/tts/datasets/preprocess.py
    """
    this_module = sys.modules[__name__]
    return getattr(this_module, name.lower())


def ljspeech(_metadata_path: str, multispeaker: bool, column_sep='|') -> dict:
    text_dict = {}
    with open(_metadata_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            l_split = line.split(column_sep)
            filename, text = l_split[0], l_split[1]
            if filename.endswith('.wav'):
                filename = filename.split('.')[0]
            text = text.replace('\n', '')

            speaker = int(l_split[2]) if multispeaker else 0
            text_dict.update({filename: (text, filename.replace('/', '_'), speaker)})

    return text_dict


def post_processed_reader(_metadata_path: str, multispeaker: bool,
                          column_sep='|', upsample_indicators='?!', upsample_factor=10) -> Tuple[Dict, List]:
    """
    Used to read metadata files created within the repo.
    """
    text_dict = {}
    upsample = []
    with open(_metadata_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            l_split = line.split(column_sep)
            filename, text = l_split[0], l_split[1]
            text = text.replace('\n', '')
            if any(el in text for el in list(upsample_indicators)):
                upsample.extend([filename] * upsample_factor)

            speaker = int(l_split[2]) if multispeaker else 0
            text_dict.update({filename: (text, speaker)})
    return text_dict, upsample


if __name__ == '__main__':
    metadata_path = '/Volumes/data/datasets/LJSpeech-1.1/metadata.csv'
    d = get_preprocessor_by_name('ljspeech')(metadata_path)
    key_list = list(d.keys())
    logger.info('metadata head')
    for key in key_list[:5]:
        logger.info(f'{key}: {d[key]}')
    logger.info('metadata tail')
    for key in key_list[-5:]:
        logger.info(f'{key}: {d[key]}')
