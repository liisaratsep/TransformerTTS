import subprocess
import shutil
from pathlib import Path
from argparse import Namespace, ArgumentParser
from enum import Enum

import numpy as np
import ruamel.yaml


class TTSMode(str, Enum):
    DATA = "data"
    ALIGNER = "aligner"
    EXTRACT = "extract"
    TTS = "tts"
    PREDICT = "predict"
    WEIGHTS = "weights"


class TrainingConfigManager:
    def __init__(self, args: Namespace, mode: TTSMode):
        if mode in ['aligner', 'extract']:
            self.model_kind = 'aligner'
        else:
            self.model_kind = 'tts'

        self.config_path = Path(args.config_path)
        self.yaml = ruamel.yaml.YAML()
        self.config = self._load_config()

        vargs = vars(args)
        for k in self.config:
            if k in vargs and vargs[k] is not None:
                self.config[k] = vargs[k]

        self.git_hash = self._get_git_hash()
        self.metadata_reader = self.config['metadata_reader']

        # create paths
        self.wav_directory = Path(self.config['wav_directory'])
        self.metadata_path = Path(self.config['metadata_path'])

        self.data_dir = Path(f"{self.config['train_data_directory']}")
        self.train_metadata_path = self.data_dir / f"train_metadata.txt"
        self.valid_metadata_path = self.data_dir / f"valid_metadata.txt"
        self.phonemized_metadata_path = self.data_dir / f"phonemized_metadata.txt"

        self.mel_dir = self.data_dir / args.mel_directory
        self.pitch_dir = self.data_dir / args.pitch_directory
        self.duration_dir = self.data_dir / args.duration_directory
        self.pitch_per_char = self.data_dir / args.character_pitch_directory

        self.base_dir = Path(self.config['save_directory']) / self.model_kind
        self.log_dir = self.base_dir / 'logs'
        self.weights_dir = self.base_dir / 'weights'

        # training parameters
        self.learning_rate = np.array(self.config['learning_rate_schedule'])[0, 1].astype(np.float32)
        if self.model_kind == 'aligner':
            self.max_r = np.array(self.config['reduction_factor_schedule'])[0, 1].astype(np.int32)
            self.stop_scaling = self.config.get('stop_loss_scaling', 1.)

        self.seed = args.seed

    def _load_config(self):
        all_config = {}
        with open(str(self.config_path), 'rb') as session_yaml:
            session_config = self.yaml.load(session_yaml)
        for key in ['paths', 'dataset', 'training_data_settings', 'audio_settings',
                    'text_settings', f'{self.model_kind}_settings']:
            all_config.update(session_config[key])
        return all_config

    @staticmethod
    def _get_git_hash():
        try:
            return subprocess.check_output(['git', 'describe', '--always']).strip().decode()
        except Exception as e:
            print(f'WARNING: could not retrieve git hash. {e}')

    def _check_hash(self):
        try:
            git_hash = subprocess.check_output(['git', 'describe', '--always']).strip().decode()
            if self.config['git_hash'] != git_hash:
                print(
                    f"WARNING: git hash mismatch. Current: {git_hash}. Training config hash: {self.config['git_hash']}")
        except Exception as e:
            print(f'WARNING: could not check git hash. {e}')

    @staticmethod
    def _print_dict_values(values, key_name, level=0, tab_size=2):
        tab = level * tab_size * ' '
        print(tab + '-', key_name, ':', values)

    def _print_dictionary(self, dictionary, recursion_level=0):
        for key in dictionary.keys():
            if isinstance(key, dict):
                recursion_level += 1
                self._print_dictionary(dictionary[key], recursion_level)
            else:
                self._print_dict_values(dictionary[key], key_name=key, level=recursion_level)

    def print_config(self):
        print('\nCONFIGURATION', self.model_kind)
        self._print_dictionary(self.config)

    def update_config(self):
        self.config['git_hash'] = self.git_hash
        self.config['automatic'] = True

    def get_model(self, ignore_hash=False):
        from model.models import Aligner, ForwardTransformer
        if not ignore_hash:
            self._check_hash()
        if self.model_kind == 'aligner':
            return Aligner.from_config(self.config, max_r=self.max_r)
        else:
            return ForwardTransformer.from_config(self.config)

    def compile_model(self, model, beta_1=0.9, beta_2=0.98):
        import tensorflow as tf
        optimizer = tf.keras.optimizers.Adam(self.learning_rate,
                                             beta_1=beta_1,
                                             beta_2=beta_2,
                                             epsilon=1e-9)
        if self.model_kind == 'aligner':
            model._compile(stop_scaling=self.stop_scaling, optimizer=optimizer)
        else:
            model._compile(optimizer=optimizer)

    def dump_config(self):
        self.update_config()
        with open(self.base_dir / f"config.yaml", 'w') as model_yaml:
            self.yaml.dump(self.config, model_yaml)

    def create_remove_dirs(self, clear_dir=False, clear_logs=False, clear_weights=False):
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir.mkdir(exist_ok=True)
        self.pitch_dir.mkdir(exist_ok=True)
        self.pitch_per_char.mkdir(exist_ok=True)
        self.mel_dir.mkdir(exist_ok=True)
        self.duration_dir.mkdir(exist_ok=True)
        if clear_dir:
            delete = input(f'Delete {self.log_dir} AND {self.weights_dir}? (y/[n])')
            if delete == 'y':
                shutil.rmtree(self.log_dir, ignore_errors=True)
                shutil.rmtree(self.weights_dir, ignore_errors=True)
        if clear_logs:
            delete = input(f'Delete {self.log_dir}? (y/[n])')
            if delete == 'y':
                shutil.rmtree(self.log_dir, ignore_errors=True)
        if clear_weights:
            delete = input(f'Delete {self.weights_dir}? (y/[n])')
            if delete == 'y':
                shutil.rmtree(self.weights_dir, ignore_errors=True)
        self.log_dir.mkdir(exist_ok=True)
        self.weights_dir.mkdir(exist_ok=True)

    def load_model(self, checkpoint_path: str = None, verbose=True):
        import tensorflow as tf
        from utils.scheduling import reduction_schedule
        model = self.get_model()
        self.compile_model(model)
        ckpt = tf.train.Checkpoint(net=model)
        manager = tf.train.CheckpointManager(ckpt, self.weights_dir,
                                             max_to_keep=None)
        if checkpoint_path:
            ckpt.restore(checkpoint_path)
            if verbose:
                print(f'restored weights from {checkpoint_path} at step {model.step}')
        else:
            if manager.latest_checkpoint is None:
                print(f'WARNING: could not find weights file. Trying to load from \n {self.weights_dir}.')
                print('Edit config to point at the right log directory.')
            ckpt.restore(manager.latest_checkpoint)
            if verbose:
                print(f'restored weights from {manager.latest_checkpoint} at step {model.step}')
        if self.model_kind == 'aligner':
            reduction_factor = reduction_schedule(model.step, self.config['reduction_factor_schedule'])
            model.set_constants(reduction_factor=reduction_factor)
        return model


def tts_argparser(mode: TTSMode):
    parser = ArgumentParser()

    parser.add_argument('--config', dest='config_path', type=str, default='config/training_config.yaml',
                        help="Path to the configuration file")
    parser.add_argument('--seed', type=int, required=False, default=None)

    config_group = parser.add_argument_group("Options to override values in the session config file")

    config_group.add_argument('--wav-directory', type=str)
    config_group.add_argument('--metadata-path', type=str)
    config_group.add_argument('--save-directory', type=str)
    config_group.add_argument('--train-data-directory', type=str)

    config_group.add_argument('--metadata-reader', type=str)

    data_group = parser.add_argument_group(
        "Options to override data subdirectory paths relative to the training data directory root"
    )

    data_group.add_argument('--mel-directory', type=str, default="mels")
    data_group.add_argument('--pitch-directory', type=str, default="pitch")
    data_group.add_argument('--duration-directory', type=str, default="duration")
    data_group.add_argument('--character-pitch-directory', type=str, default="char_pitch")

    if mode in [TTSMode.ALIGNER, TTSMode.TTS]:
        parser.add_argument('--reset_dir', dest='clear_dir', action='store_true',
                            help="deletes everything under this config's folder.")
        parser.add_argument('--reset_logs', dest='clear_logs', action='store_true',
                            help="deletes logs under this config's folder.")
        parser.add_argument('--reset_weights', dest='clear_weights', action='store_true',
                            help="deletes weights under this config's folder.")

    elif mode == TTSMode.DATA:
        parser.add_argument('--skip-phonemes', action='store_true')
        parser.add_argument('--skip-mels', action='store_true')

    elif mode == TTSMode.EXTRACT:
        parser.add_argument('--best', dest='best', action='store_true',
                            help='Use best head instead of weighted average of heads.')
        parser.add_argument('--autoregressive_weights', type=str, default=None,
                            help='Explicit path to autoregressive model weights.')
        parser.add_argument('--skip_char_pitch', dest='skip_char_pitch', action='store_true')
        parser.add_argument('--skip_durations', dest='skip_durations', action='store_true')

    elif mode == TTSMode.PREDICT:
        parser.add_argument('--path', '-p', dest='path', default=None, type=str)
        parser.add_argument('--step', dest='step', default='90000', type=str)
        parser.add_argument('--text', '-t', dest='text', default=None, type=str)
        parser.add_argument('--file', '-f', dest='file', default=None, type=str)
        parser.add_argument('--outdir', '-o', dest='outdir', default=None, type=str)
        parser.add_argument('--store_mel', '-m', dest='store_mel', action='store_true')
        parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
        parser.add_argument('--single', '-s', dest='single', action='store_true')
        parser.add_argument('--speaker-id', dest='speaker_id', default=1, type=int)

    return parser
