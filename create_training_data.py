import logging

from utils.training_config_manager import TrainingConfigManager, TTSMode
from utils.argparser import tts_argparser

MODE = TTSMode("data")
parser = tts_argparser(MODE)
args = parser.parse_args()
config = TrainingConfigManager(mode=MODE, **vars(args))

logger = logging.getLogger(__name__)
logger.setLevel(args.log_level)

for arg in vars(args):
    logger.info('{}: {}'.format(arg, getattr(args, arg)))

if __name__ == '__main__':

    from pathlib import Path
    import pickle

    import numpy as np
    from p_tqdm import p_uimap, p_umap

    from utils.logging_utils import SummaryManager
    from data.text import TextToTokens
    from data.datasets import DataReader

    from data.audio import Audio
    from data.text import symbols

    if config.seed is not None:
        np.random.seed(config.seed)

    config.create_remove_dirs()
    metadatareader = DataReader.from_config(config, kind='original', scan_wavs=True)
    summary_manager = SummaryManager(model=None, log_dir=config.log_dir / 'data_preprocessing', config=config.config,
                                     default_writer='data_preprocessing')
    file_ids_from_wavs = list(metadatareader.wav_paths.keys())
    logger.info(f"Reading wavs from {metadatareader.wav_directory}")
    logger.info(f"Reading metadata from {metadatareader.metadata_path}")
    logger.info(f'\nFound {len(metadatareader.filenames)} metadata lines.')
    logger.info(f'\nFound {len(file_ids_from_wavs)} wav files.')
    cross_file_ids = file_ids_from_wavs

    if not args.skip_mels:

        def process_wav(wav_path: Path):
            file_name = str(wav_path)[len(str(config.wav_directory)):].replace('/', '_').split('.')[0].strip('_')
            y, sr = audio.load_wav(str(wav_path))
            pitch = audio.extract_pitch(y)
            mel = audio.mel_spectrogram(y)
            assert mel.shape[1] == audio.config['mel_channels'], len(mel.shape) == 2
            assert mel.shape[0] == pitch.shape[0], f'{mel.shape[0]} == {pitch.shape[0]} (wav {y.shape})'
            mel_path = (config.mel_dir / file_name).with_suffix('.npy')
            pitch_path = (config.pitch_dir / file_name).with_suffix('.npy')
            np.save(mel_path, mel)
            np.save(pitch_path, pitch)
            return {'fname': file_name, 'mel.len': mel.shape[0], 'pitch.path': pitch_path, 'pitch': pitch}


        logger.info(f"\nMels will be stored stored under")
        logger.info(f"{config.mel_dir}")
        audio = Audio.from_config(config=config.config)
        wav_files = [metadatareader.wav_paths[k] for k in cross_file_ids]
        len_dict = {}
        remove_files = []
        mel_lens = []
        pitches = {}
        wav_iter = p_uimap(process_wav, wav_files)
        for out_dict in wav_iter:
            len_dict.update({out_dict['fname']: out_dict['mel.len']})
            pitches.update({out_dict['pitch.path']: out_dict['pitch']})
            if out_dict['mel.len'] > config.config['max_mel_len'] or out_dict['mel.len'] < config.config['min_mel_len']:
                remove_files.append(out_dict['fname'])
            else:
                mel_lens.append(out_dict['mel.len'])


        def normalize_pitch_vectors(pitch_vecs):
            nonzeros = np.concatenate([v[np.where(v != 0.0)[0]]
                                       for v in pitch_vecs.values()])
            _mean, _std = np.mean(nonzeros), np.std(nonzeros)
            return _mean, _std


        def process_pitches(_item: tuple):
            _fname, pitch = _item
            zero_idxs = np.where(pitch == 0.0)[0]
            pitch -= mean
            pitch /= std
            pitch[zero_idxs] = 0.0
            np.save(_fname, pitch)


        mean, std = normalize_pitch_vectors(pitches)
        pickle.dump({'pitch_mean': mean, 'pitch_std': std}, open(config.data_dir / 'pitch_stats.pkl', 'wb'))
        pitch_iter = p_umap(process_pitches, pitches.items())

        pickle.dump(len_dict, open(config.data_dir / 'mel_len.pkl', 'wb'))
        pickle.dump(remove_files, open(config.data_dir / 'under-over_sized_mels.pkl', 'wb'))
        summary_manager.add_histogram('Mel Lengths', values=np.array(mel_lens))
        total_mel_len = np.sum(mel_lens)
        total_wav_len = total_mel_len * audio.config['hop_length']
        summary_manager.display_scalar('Total duration (hours)',
                                       scalar_value=total_wav_len / audio.config['sampling_rate'] / 60. ** 2)

    if not args.skip_phonemes:
        remove_files = pickle.load(open(config.data_dir / 'under-over_sized_mels.pkl', 'rb'))
        phonemized_metadata_path = config.phonemized_metadata_path
        train_metadata_path = config.train_metadata_path
        test_metadata_path = config.valid_metadata_path

        alphabet = config.config['alphabet']
        if not alphabet:
            alphabet = symbols.alphabet

        logger.info(f'\nReading metadata from {metadatareader.metadata_path}')
        logger.info(f'\nFound {len(metadatareader.filenames)} lines.')
        filter_metadata = []
        fnames = metadatareader.text_dict.keys()
        for fname in fnames:
            item = metadatareader.text_dict[fname][0]
            non_p = [c for c in item if c in alphabet]
            if len(non_p) < 1:
                filter_metadata.append(fname)
        if len(filter_metadata) > 0:
            logger.info(f'Removing {len(filter_metadata)} suspiciously short line(s):')
            for fname in filter_metadata:
                logger.info(f'{fname}: {metadatareader.text_dict[fname]}')
        logger.info(f'\nRemoving {len(remove_files)} line(s) due to mel filtering.')
        remove_files += filter_metadata
        metadata_file_ids = [fname for fname in fnames if fname not in remove_files]
        metadata_len = len(metadata_file_ids)
        sample_items = np.random.choice(metadata_file_ids, 5)
        test_len = config.config['n_test']
        train_len = metadata_len - test_len
        logger.info(f'\nMetadata contains {metadata_len} lines.')
        logger.info(f'\nFiles will be stored under {config.data_dir}')
        logger.info(f' - all: {phonemized_metadata_path}')
        logger.info(f' - {train_len} training lines: {train_metadata_path}')
        logger.info(f' - {test_len} validation lines: {test_metadata_path}')

        logger.info('\nMetadata samples:')
        for i in sample_items:
            logger.info(f'{i}:{metadatareader.text_dict[i]}')
            summary_manager.add_text(f'{i}/text', text=metadatareader.text_dict[i][:2])

        # run cleaner on raw text
        text_proc = TextToTokens.default(config.config['phoneme_language'],
                                         add_start_end=False,
                                         with_stress=config.config['with_stress'],
                                         model_breathing=config.config['model_breathing'],
                                         njobs=1,
                                         alphabet=config.config['alphabet'],
                                         collapse_whitespace=config.config['collapse_whitespace'])


        def process_phonemes(_file_id):
            text = metadatareader.text_dict[_file_id][0]
            _speaker = metadatareader.text_dict[_file_id][-1]
            try:
                phon = text_proc.phonemizer(text)
            except Exception as e:
                logger.info(f'{e}\nFile id {_file_id}')
                raise BrokenPipeError
            return _file_id, phon, _speaker


        logger.info('\nPHONEMIZING')
        phonemized_data = {}
        phon_iter = p_uimap(process_phonemes, metadata_file_ids)
        for (file_id, phonemes, speaker) in phon_iter:
            phonemized_data.update({file_id.replace('/', '_'): (phonemes, speaker)})

        logger.info('\nPhonemized metadata samples:')
        for i in sample_items:
            i = i.replace('/', '_')
            logger.info(f'{i}:{phonemized_data[i]}')
            summary_manager.add_text(f'{i}/phonemes', text=phonemized_data[i][0])

        new_metadata = [f'{k}|{v[0]}|{v[1]}\n' for k, v in phonemized_data.items()]
        shuffled_metadata = np.random.permutation(new_metadata)
        train_metadata = shuffled_metadata[0:train_len]
        test_metadata = shuffled_metadata[-test_len:]

        with open(phonemized_metadata_path, 'w+', encoding='utf-8') as file:
            file.writelines(new_metadata)
        with open(train_metadata_path, 'w+', encoding='utf-8') as file:
            file.writelines(train_metadata)
        with open(test_metadata_path, 'w+', encoding='utf-8') as file:
            file.writelines(test_metadata)
        # some checks
        assert metadata_len == len(set(list(phonemized_data.keys()))), \
            f'Length of metadata ({metadata_len}) does not match the length of the phoneme array ' \
            f'({len(set(list(phonemized_data.keys())))}). Check for empty text lines in metadata.'
        assert len(train_metadata) + len(test_metadata) == metadata_len, \
            f'Train and/or validation lengths incorrect. ({len(train_metadata)}+{len(test_metadata)}!={metadata_len})'

    logger.info('\nDone')
