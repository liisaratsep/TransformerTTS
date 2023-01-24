from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .training_config_manager import TTSMode


def tts_argparser(mode: TTSMode):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', dest='config_path', type=str, default=None,
                        help="Path to the configuration file, "
                             "defaults to the automatically saved config file in the model's directory.")
    parser.add_argument('--seed', type=int, help="The seed value for model initialization and data shuffling.")
    parser.add_argument('--log-level', type=str, help="Script logger level", default="INFO")
    parser.add_argument('--save-directory', type=str, default="",
                        help="Directory for training metadata, models and logs. Will be created if it doesn't exist.")

    if mode == TTSMode.DATA:
        data_group = parser.add_argument_group("Data preprocessing settings")
        data_group.add_argument('--metadata-path', type=str, default="",
                                help="A path to a metadata file that specifies wav file paths, transcriptions and "
                                     "optionally the speaker ID")
        data_group.add_argument('--wav-directory', type=str, default="",
                                help="A directory with .wav files. File paths in metadata are relative to this path.")
        data_group.add_argument('--skip-phonemes', action='store_true', help="Skip phoneme generation step.")
        data_group.add_argument('--skip-mels', action='store_true', help="Skip mel-spectrogram generation step.")

    elif mode == TTSMode.EXTRACT:
        extraction_group = parser.add_argument_group("Duration extraction settings")
        extraction_group.add_argument('--best', dest='best', action='store_true',
                                      help='Use best head instead of weighted average of heads.')
        extraction_group.add_argument('--autoregressive_weights', type=str,
                                      help='Explicit path to autoregressive model weights.')
        extraction_group.add_argument('--skip-durations', dest='skip_durations', action='store_true',
                                      help="Skip character duration calculation, use this if you are using a different "
                                           "alignment model.")
        extraction_group.add_argument('--skip-char-pitch', dest='skip_char_pitch', action='store_true',
                                      help="Skip character pitch calculation.")

    if mode not in [TTSMode.PREDICT, TTSMode.WEIGHTS]:
        preprocessed_group = parser.add_argument_group("Preprocessed training data location options")
        preprocessed_group.add_argument('--mel-directory', type=str, default="mels")
        preprocessed_group.add_argument('--pitch-directory', type=str, default="pitch")
        preprocessed_group.add_argument('--duration-directory', type=str, default="duration")
        preprocessed_group.add_argument('--character-pitch-directory', type=str, default="char_pitch")

    if mode in [TTSMode.ALIGNER, TTSMode.TTS]:
        training_group = parser.add_argument_group("Options for training")
        training_group.add_argument('--test-files', nargs='+',
                                    default=[
                                        f"test_files/{'aligner_' if mode == TTSMode.ALIGNER else ''}test_sentences.txt"
                                    ],
                                    help="A list of text files with one sentence per line to generate test samples. "
                                         "Predictions are stored in tensorboard logs.")

        training_group.add_argument('--reset-dir', dest='clear_dir', action='store_true',
                                    help="Delete weights and logs (reset training).")
        training_group.add_argument('--reset-logs', dest='clear_logs', action='store_true',
                                    help="Deletes logs.")
        training_group.add_argument('--reset-weights', dest='clear_weights', action='store_true',
                                    help="Deletes weights.")

    elif mode == TTSMode.PREDICT:
        predict_group = parser.add_argument_group("Options for inference")
        predict_group.add_argument('--path', type=str,
                                   help="Optional path to a model, latest will be loaded if not specified.")
        predict_group.add_argument('--file', type=str, help="Text input file to be synthesized.")
        predict_group.add_argument('--text', type=str, help="Text to be synthesized if an input file is not specified.")
        predict_group.add_argument('--outdir', type=str, help="Directory for the output file.")
        predict_group.add_argument('--store-mel', action='store_true',
                                   help="Also saves a Numpy array of the mel-spectrogram.")
        predict_group.add_argument('--verbose', action='store_true', help="Verbose mode.")
        predict_group.add_argument('--single', action='store_true', help="Saves each line of text in a separate file.")
        predict_group.add_argument('--speaker-id', default=0, type=int, help="Speaker ID for multispeaker models.")

    elif mode == TTSMode.WEIGHTS:
        weights_group = parser.add_argument_group("Options for extracting weights from a checkpoint")
        weights_group.add_argument('--checkpoint-path', type=str, default=None,
                                   help="Checkpoint path, defaults to latest in the save directory.")
        weights_group.add_argument('--target-dir', type=str)

    return parser
