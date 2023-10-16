import logging
from enum import Enum
from pathlib import Path
import subprocess
from typing import Optional

import tensorflow as tf
import numpy as np
from ruamel.yaml import YAML

from model.transformer_utils import create_encoder_padding_mask, create_mel_padding_mask
from utils.losses import weighted_sum_losses, masked_mean_absolute_error
from data.text import TextToTokens
from model.layers import StatPredictor, Expand, SelfAttentionBlocks

logger = logging.getLogger(__name__)


class Multispeaker(str, Enum):
    GST = "GST"
    embedding = "embedding"


class ForwardTransformer(tf.keras.models.Model):
    def __init__(self,
                 encoder_model_dimension: int,
                 decoder_model_dimension: int,
                 dropout_rate: float,
                 decoder_num_heads: list,
                 encoder_num_heads: list,
                 encoder_max_position_encoding: int,
                 decoder_max_position_encoding: int,
                 encoder_dense_blocks: int,
                 decoder_dense_blocks: int,
                 duration_conv_filters: list,
                 pitch_conv_filters: list,
                 duration_kernel_size: int,
                 pitch_kernel_size: int,
                 predictors_dropout: float,
                 mel_channels: int,
                 phoneme_language: str,
                 with_stress: bool,
                 model_breathing: bool,
                 transposed_attn_convs: bool,
                 encoder_attention_conv_filters: list = None,
                 decoder_attention_conv_filters: list = None,
                 encoder_attention_conv_kernel: int = None,
                 decoder_attention_conv_kernel: int = None,
                 encoder_feed_forward_dimension: int = None,
                 decoder_feed_forward_dimension: int = None,
                 alphabet: str = None,
                 collapse_whitespace: bool = True,
                 use_layernorm: bool = False,
                 multispeaker: Optional[str] = None,
                 n_speakers: int = 1,
                 n_languages: int = 1,
                 n_styles: int = 1,
                 debug=False,
                 **kwargs):
        super(ForwardTransformer, self).__init__()
        self.config = self._make_config(locals(), kwargs)
        self.multispeaker = Multispeaker(multispeaker) if multispeaker is not None else None
        self.n_speakers = n_speakers
        self.n_languages = n_languages
        self.n_styles = n_styles
        self.text_pipeline = TextToTokens.default(phoneme_language,
                                                  add_start_end=False,
                                                  with_stress=with_stress,
                                                  model_breathing=model_breathing,
                                                  alphabet=alphabet,
                                                  collapse_whitespace=collapse_whitespace,
                                                  gst=(self.multispeaker == Multispeaker.GST),
                                                  zfill=len(str(int(n_speakers - 1))))
        self.symbols = self.text_pipeline.tokenizer.alphabet
        self.mel_channels = mel_channels
        self.encoder_prenet = tf.keras.layers.Embedding(self.text_pipeline.tokenizer.vocab_size,
                                                        encoder_model_dimension,
                                                        name='Embedding')
        self.encoder = SelfAttentionBlocks(model_dim=encoder_model_dimension,
                                           dropout_rate=dropout_rate,
                                           num_heads=encoder_num_heads,
                                           feed_forward_dimension=encoder_feed_forward_dimension,
                                           maximum_position_encoding=encoder_max_position_encoding,
                                           dense_blocks=encoder_dense_blocks,
                                           conv_filters=encoder_attention_conv_filters,
                                           kernel_size=encoder_attention_conv_kernel,
                                           conv_activation='relu',
                                           transposed_convs=transposed_attn_convs,
                                           use_layernorm=use_layernorm,
                                           name='Encoder')
        if self.multispeaker == Multispeaker.embedding:
            self.speaker_embedder = tf.keras.layers.Embedding(self.n_speakers,
                                                              encoder_model_dimension,
                                                              name='speaker_embedder')
        else:
            self.speaker_embedder = None
        if self.n_languages > 1:
            assert self.multispeaker == Multispeaker.embedding, "Multilingual models must use multispeaker embedding"
            self.language_embedder = tf.keras.layers.Embedding(self.n_languages,
                                                               encoder_model_dimension,
                                                               name='language_embedder')
        if self.n_styles > 1:
            assert self.multispeaker == Multispeaker.embedding, "Multi-style models must use multispeaker embedding"
            self.style_embedder = tf.keras.layers.Embedding(self.n_styles,
                                                            encoder_model_dimension,
                                                            name='style_embedder')
        elif self.speaker_embedder is not None:
            self.style_embedder = self.speaker_embedder
        else:
            self.style_embedder = None
        self.dur_pred = StatPredictor(conv_filters=duration_conv_filters,
                                      kernel_size=duration_kernel_size,
                                      conv_padding='same',
                                      conv_activation='relu',
                                      dense_activation='relu',
                                      dropout_rate=predictors_dropout,
                                      name='dur_pred')
        self.expand = Expand(name='expand', model_dim=encoder_model_dimension)
        self.pitch_pred = StatPredictor(conv_filters=pitch_conv_filters,
                                        kernel_size=pitch_kernel_size,
                                        conv_padding='same',
                                        conv_activation='relu',
                                        dense_activation='linear',
                                        dropout_rate=predictors_dropout,
                                        name='pitch_pred')
        self.pitch_embed = tf.keras.layers.Dense(encoder_model_dimension, activation='relu')
        self.decoder = SelfAttentionBlocks(model_dim=decoder_model_dimension,
                                           dropout_rate=dropout_rate,
                                           num_heads=decoder_num_heads,
                                           feed_forward_dimension=decoder_feed_forward_dimension,
                                           maximum_position_encoding=decoder_max_position_encoding,
                                           dense_blocks=decoder_dense_blocks,
                                           conv_filters=decoder_attention_conv_filters,
                                           kernel_size=decoder_attention_conv_kernel,
                                           conv_activation='relu',
                                           transposed_convs=transposed_attn_convs,
                                           use_layernorm=use_layernorm,
                                           name='Decoder')
        self.out = tf.keras.layers.Dense(mel_channels)
        self.training_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32)
        ]
        self.forward_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ]
        self.forward_masked_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        ]
        self.loss_weights = [1., 1., 3.]
        self.debug = debug
        self._apply_all_signatures()

    def _apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)

    def _apply_all_signatures(self):
        self.forward = self._apply_signature(self._forward, self.forward_input_signature)
        self.train_step = self._apply_signature(self._train_step, self.training_input_signature)
        self.val_step = self._apply_signature(self._val_step, self.training_input_signature)

    @staticmethod
    def _make_config(locals_: dict, kwargs: dict) -> dict:
        config = {}
        keys = [k for k in locals_.keys() if (k not in kwargs) and (k not in ['self', '__class__', 'kwargs'])]
        for k in keys:
            if isinstance(locals_[k], dict):
                config.update(locals_[k])
            else:
                config.update({k: locals_[k]})
        config.update(kwargs)
        return config

    def _train_step(self, input_sequence, speaker_id, language_id, style_id, mel_coef, target_sequence,
                    target_durations, target_pitch):
        target_durations = tf.expand_dims(target_durations, -1)
        target_pitch = tf.expand_dims(target_pitch, -1)
        mel_len = int(tf.shape(target_sequence)[1])
        mel_coef = tf.expand_dims(tf.expand_dims(mel_coef, -1), -1)
        with tf.GradientTape() as tape:
            model_out = self.__call__(input_sequence, target_durations, target_pitch=target_pitch,
                                      training=True, speaker_id=speaker_id, language_id=language_id, style_id=style_id)
            loss, loss_vals = weighted_sum_losses((target_sequence * mel_coef,
                                                   target_durations,
                                                   target_pitch),
                                                  (model_out['mel'][:, :mel_len, :] * mel_coef,
                                                   model_out['duration'],
                                                   model_out['pitch']),
                                                  self.loss,
                                                  self.loss_weights)  # [1., 1., 3.] for mel, duration, pitch
        model_out.update({'loss': loss})
        model_out.update({'losses': {'mel': loss_vals[0], 'duration': loss_vals[1], 'pitch': loss_vals[2]}})
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out

    def compile_model(self, optimizer):
        self.compile(loss=[masked_mean_absolute_error,
                           masked_mean_absolute_error,
                           masked_mean_absolute_error],
                     loss_weights=self.loss_weights,
                     optimizer=optimizer)

    def _val_step(self, input_sequence, speaker_id, language_id, style_id, mel_coef, target_sequence, target_durations, target_pitch):
        target_durations = tf.expand_dims(target_durations, -1)
        target_pitch = tf.expand_dims(target_pitch, -1)
        mel_len = int(tf.shape(target_sequence)[1])
        mel_coef = tf.expand_dims(tf.expand_dims(mel_coef, -1), -1)
        model_out = self.__call__(input_sequence, target_durations, target_pitch=target_pitch,
                                  training=False, speaker_id=speaker_id, language_id=language_id, style_id=style_id)
        loss, loss_vals = weighted_sum_losses((target_sequence * mel_coef,
                                               target_durations,
                                               target_pitch),
                                              (model_out['mel'][:, :mel_len, :] * mel_coef,
                                               model_out['duration'],
                                               model_out['pitch']),
                                              self.loss,
                                              self.loss_weights)
        model_out.update({'loss': loss})
        model_out.update({'losses': {'mel': loss_vals[0], 'duration': loss_vals[1], 'pitch': loss_vals[2]}})
        return model_out

    def _forward(self, input_sequence, speaker_id, language_id, style_id, durations_scalar):
        return self.__call__(input_sequence, target_durations=None, target_pitch=None, training=False,
                             durations_scalar=durations_scalar, max_durations_mask=None,
                             min_durations_mask=None, speaker_id=speaker_id, language_id=language_id, style_id=style_id)

    @property
    def step(self):
        return int(self.optimizer.iterations)

    def call(self, x, target_durations=None, target_pitch=None, training=False, durations_scalar=1.,
             max_durations_mask=None, min_durations_mask=None, speaker_id=None, language_id=None, style_id=None,
             pitch_id=None, dur_id=None):
        encoder_padding_mask = create_encoder_padding_mask(x)
        x = self.encoder_prenet(x)

        if self.language_embedder is not None:
            language_emb = self.language_embedder(language_id)
            if len(x.shape) == 3:
                language_emb = tf.expand_dims(language_emb, axis=1)
            x = x + language_emb

        x, encoder_attention = self.encoder(x, training=training, padding_mask=encoder_padding_mask)
        padding_mask = 1. - tf.squeeze(encoder_padding_mask, axis=(1, 2))[:, :, None]

        if self.speaker_embedder is not None:
            speaker_emb = self.speaker_embedder(speaker_id)

            style_id = speaker_id if style_id is None or self.n_styles == 1 else style_id
            dur_emb = self.style_embedder(style_id) if dur_id is None else self.style_embedder(dur_id)
            pitch_emb = self.style_embedder(style_id) if pitch_id is None else self.style_embedder(pitch_id)

            if len(x.shape) == 3:
                speaker_emb = tf.expand_dims(speaker_emb, axis=1)
                pitch_emb = tf.expand_dims(pitch_emb, axis=1)
                dur_emb = tf.expand_dims(dur_emb, axis=1)

            durations = self.dur_pred(x + dur_emb, training=training, mask=padding_mask)
            pitch = self.pitch_pred(x + pitch_emb, training=training, mask=padding_mask)

        else:
            speaker_emb = None
            durations = self.dur_pred(x, training=training, mask=padding_mask)
            pitch = self.pitch_pred(x, training=training, mask=padding_mask)

        if target_pitch is not None:
            pitch_embed = self.pitch_embed(target_pitch)
        else:
            pitch_embed = self.pitch_embed(pitch)
        x = x + pitch_embed
        if target_durations is not None:
            use_durations = target_durations
        else:
            use_durations = durations * durations_scalar
        if max_durations_mask is not None:
            use_durations = tf.math.minimum(use_durations, tf.expand_dims(max_durations_mask, -1))
        if min_durations_mask is not None:
            use_durations = tf.math.maximum(use_durations, tf.expand_dims(min_durations_mask, -1))
        mels = self.expand(x, use_durations)
        if self.speaker_embedder is not None:
            mels = tf.concat((speaker_emb, mels), 1)
        expanded_mask = create_mel_padding_mask(mels)
        mels, decoder_attention = self.decoder(mels, training=training, padding_mask=expanded_mask, reduction_factor=1)
        if self.speaker_embedder is not None:
            mels = mels[:, 1:, :]
        mels = self.out(mels)
        model_out = {'mel': mels,
                     'duration': durations,
                     'pitch': pitch,
                     'expanded_mask': expanded_mask,
                     'encoder_attention': encoder_attention,
                     'decoder_attention': decoder_attention}
        return model_out

    def set_constants(self, learning_rate: float = None, **_):
        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)

    def encode_text(self, text, speaker_id):
        return self.text_pipeline(text, speaker_id=speaker_id)

    def predict(self, inp, encode=True, speed_regulator=1., phoneme_max_duration=None, phoneme_min_duration=None,
                max_durations_mask=None, min_durations_mask=None, phoneme_durations=None, phoneme_pitch=None,
                speaker_id=0, language_id=0, style_id=None, dur_id=None, pitch_id=None):
        if encode:
            inp = self.encode_text(inp, speaker_id=speaker_id)
        if len(tf.shape(inp)) < 2:
            inp = tf.expand_dims(inp, 0)
        inp = tf.cast(inp, tf.int32)
        duration_scalar = tf.cast(1. / speed_regulator, tf.float32)
        max_durations_mask = self._make_max_duration_mask(inp, phoneme_max_duration)
        min_durations_mask = self._make_min_duration_mask(inp, phoneme_min_duration)

        speaker_id = tf.cast([speaker_id], tf.int32) if type(speaker_id) == int else speaker_id
        language_id = tf.cast([language_id], tf.int32) if type(language_id) == int else language_id
        style_id = tf.cast([style_id], tf.int32) if type(style_id) == int else style_id
        dur_id = tf.cast([dur_id], tf.int32) if type(dur_id) == int else dur_id
        pitch_id = tf.cast([pitch_id], tf.int32) if type(pitch_id) == int else pitch_id

        out = self.call(inp,
                        target_durations=phoneme_durations,
                        target_pitch=phoneme_pitch,
                        training=False,
                        durations_scalar=duration_scalar,
                        max_durations_mask=max_durations_mask,
                        min_durations_mask=min_durations_mask,
                        speaker_id=speaker_id,
                        language_id=language_id,
                        style_id=style_id,
                        dur_id=dur_id,
                        pitch_id=pitch_id)
        out['mel'] = tf.squeeze(out['mel'])
        return out

    def _make_max_duration_mask(self, encoded_text, phoneme_max_duration):
        np_text = np.array(encoded_text)
        new_mask = np.ones(tf.shape(encoded_text)) * float('inf')
        if phoneme_max_duration is not None:
            for item in phoneme_max_duration.items():
                phon_idx = self.text_pipeline.tokenizer(item[0])[0]
                new_mask[np_text == phon_idx] = item[1]
        return tf.cast(tf.convert_to_tensor(new_mask), tf.float32)

    def _make_min_duration_mask(self, encoded_text, phoneme_min_duration):
        np_text = np.array(encoded_text)
        new_mask = np.zeros(tf.shape(encoded_text))
        if phoneme_min_duration is not None:
            for item in phoneme_min_duration.items():
                phon_idx = self.text_pipeline.tokenizer(item[0])[0]
                new_mask[np_text == phon_idx] = item[1]
        return tf.cast(tf.convert_to_tensor(new_mask), tf.float32)

    def build_model_weights(self) -> None:
        _ = self(tf.zeros((1, 1)), target_durations=None, target_pitch=None, training=False, speaker_id=tf.zeros(1, ), language_id=tf.zeros(1, ))

    def save_model(self, path: str):
        yaml = YAML()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self, 'text_pipeline'):
            save_list = ''.join(self.symbols)
            self.config.update({'alphabet': save_list})
        if hasattr(self, 'step'):
            self.config.update({'step': self.step})
        try:
            git_hash = subprocess.check_output(['git', 'describe', '--always']).strip().decode()
            self.config.update({'git_hash': git_hash})
        except Exception as e:
            logger.warning(f'could not retrieve git hash. {e}')
        with open(path / 'config.yaml', 'w') as f:
            yaml.dump(dict(self.config), f)  # conversion necessary (is tf wrapper otherwise)
        # only needed when model was loaded from a checkpoint
        self.build_model_weights()
        self.save_weights(path / 'model_weights.hdf5')

    @classmethod
    def load_model(cls, path):
        yaml = YAML()
        path = Path(path)
        with open(path / 'config.yaml', 'r') as f:
            config = yaml.load(f)
        model = cls.from_config(config)
        try:
            git_hash = subprocess.check_output(['git', 'describe', '--always']).strip().decode()
            if 'git_hash' in config:
                if config['git_hash'] != git_hash:
                    logger.warning(f"git_hash mismatch: {config['git_hash']}(config) vs {git_hash}(local).")
            else:
                logger.warning(f'could not check git hash from config.')
        except Exception as e:
            logger.warning(f'could not retrieve git hash. {e}')
        model.build_model_weights()
        model.load_weights(path / 'model_weights.hdf5')
        return model

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        return cls(**config)

    def get_config(self):
        pass
