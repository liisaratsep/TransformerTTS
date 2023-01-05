import logging

from utils.training_config_manager import TrainingConfigManager, TTSMode
from utils.argparser import tts_argparser

MODE = TTSMode("predict")
parser = tts_argparser(MODE)
args = parser.parse_args()
config = TrainingConfigManager(mode=MODE, **vars(args))

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    from pathlib import Path

    import numpy as np

    from data.audio import Audio
    from model.models import ForwardTransformer

    fname = None
    text = None

    if args.file is not None:
        with open(args.file, 'r') as file:
            text = file.readlines()
        fname = Path(args.file).stem
    elif args.text is not None:
        text = [args.text]
        fname = 'custom_text'
    else:
        logger.error(f'Specify either an input text (-t "some text") or a text input file (-f /path/to/file.txt)')
        exit()

    # load the appropriate model
    outdir = Path(args.outdir) if args.outdir is not None else Path('.')

    if args.path is not None:
        logger.info(f'Loading model from {args.path}')
        model = ForwardTransformer.load_model(args.path)
    else:
        logger.info(f'Trying to load the latest checkpoint from model from {args.save_directory}')
        model = config.load_model()

    file_name = f"{fname}_{model.config['step']}"
    outdir = outdir / 'outputs' / f'{fname}'
    outdir.mkdir(exist_ok=True, parents=True)
    output_path = (outdir / file_name).with_suffix('.wav')

    audio = Audio.from_config(model.config)
    logger.info(f'Output wav under {output_path.parent}')
    wavs = []
    for i, text_line in enumerate(text):
        phons = model.text_pipeline.phonemizer(text_line)
        tokens = model.text_pipeline.tokenizer(phons, args.speaker_id)
        if args.verbose:
            logger.info(f'Predicting {text_line}')
            logger.info(f'Phonemes: "{phons}"')
            logger.info(f'Tokens: "{tokens}"')
        out = model.predict(tokens, speaker_id=args.speaker_id, encode=False, phoneme_max_duration=None)
        mel = out['mel'].numpy().T
        wav = audio.reconstruct_waveform(mel)
        wavs.append(wav)
        if args.store_mel:
            np.save(str((outdir / (file_name + f'_{i}')).with_suffix('.mel')), out['mel'].numpy())
        if args.single:
            audio.save_wav(wav, (outdir / (file_name + f'_{i}')).with_suffix('.wav'))
    audio.save_wav(np.concatenate(wavs), output_path)
