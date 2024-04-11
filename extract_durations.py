import logging

from utils.training_config_manager import TrainingConfigManager, TTSMode
from utils.argparser import tts_argparser

MODE = TTSMode("extract")
parser = tts_argparser(MODE)
args = parser.parse_args()
config = TrainingConfigManager(mode=MODE, **vars(args))

logger = logging.getLogger(__name__)
logger.setLevel(args.log_level)

if __name__ == '__main__':
    import pickle

    import tensorflow as tf
    import numpy as np
    from tqdm import tqdm
    from p_tqdm import p_umap

    from utils.logging_utils import SummaryManager
    from data.datasets import AlignerPreprocessor
    from utils.alignments import get_durations_from_alignment
    from utils.scripts_utils import dynamic_memory_allocation
    from data.datasets import AlignerDataset
    from data.datasets import DataReader

    dynamic_memory_allocation()

    if config.seed is not None:
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

    weighted = not args.best
    tag_description = ''.join([
        f'{"_weighted" * weighted}{"_best" * (not weighted)}',
    ])
    writer_tag = f'DurationExtraction{tag_description}'
    logger.info(writer_tag)

    config.print_config()

    if not args.skip_durations:
        model = config.load_model(args.autoregressive_weights)
        if model.r != 1:
            logger.error(f"model's reduction factor is greater than 1, check config. (r={model.r}")

        data_prep = AlignerPreprocessor.from_config(config=config,
                                                    tokenizer=model.text_pipeline.tokenizer)
        data_handler = AlignerDataset.from_config(config,
                                                  preprocessor=data_prep,
                                                  kind='phonemized')
        target_dir = config.duration_dir
        config.dump_config()
        dataset = data_handler.get_dataset(bucket_batch_sizes=config.config['bucket_batch_sizes'],
                                           bucket_boundaries=config.config['bucket_boundaries'],
                                           shuffle=False,
                                           drop_remainder=False)

        last_layer_key = 'Decoder_LastBlock_CrossAttention'
        logger.info(f'Extracting attention from layer {last_layer_key}')

        summary_manager = SummaryManager(model=model, log_dir=config.log_dir / 'Duration Extraction',
                                         config=config.config,
                                         default_writer='Duration Extraction')
        all_durations = np.array([])
        new_alignments = []
        iterator = tqdm(enumerate(dataset.all_batches()))
        step = 0
        for c, (mel_batch, text_batch, stop_batch, file_name_batch) in iterator:
            iterator.set_description(f'Processing dataset')
            outputs = model.val_step(inp=text_batch,
                                     tar=mel_batch,
                                     stop_prob=stop_batch)
            attention_values = outputs['decoder_attention'][last_layer_key].numpy()
            text = text_batch.numpy()

            mel = mel_batch.numpy()

            durations, final_align, jumpiness, peakiness, diag_measure = get_durations_from_alignment(
                batch_alignments=attention_values,
                mels=mel,
                phonemes=text,
                weighted=weighted,
                zfill=model.text_pipeline.tokenizer.zfill)
            batch_avg_jumpiness = tf.reduce_mean(jumpiness, axis=0)
            batch_avg_peakiness = tf.reduce_mean(peakiness, axis=0)
            batch_avg_diag_measure = tf.reduce_mean(diag_measure, axis=0)
            for i in range(tf.shape(jumpiness)[1]):
                summary_manager.display_scalar(tag=f'DurationAttentionJumpiness/head{i}',
                                               scalar_value=tf.reduce_mean(batch_avg_jumpiness[i]), step=c)
                summary_manager.display_scalar(tag=f'DurationAttentionPeakiness/head{i}',
                                               scalar_value=tf.reduce_mean(batch_avg_peakiness[i]), step=c)
                summary_manager.display_scalar(tag=f'DurationAttentionDiagonality/head{i}',
                                               scalar_value=tf.reduce_mean(batch_avg_diag_measure[i]), step=c)

            for i, name in enumerate(file_name_batch):
                all_durations = np.append(all_durations, durations[i])  # for plotting only
                summary_manager.add_image(tag='ExtractedAlignments',
                                          image=tf.expand_dims(tf.expand_dims(final_align[i], 0), -1),
                                          step=step)

                step += 1
                np.save(str(target_dir / f"{name.numpy().decode('utf-8')}.npy"), durations[i])

        all_durations[all_durations >= 20] = 20  # for plotting only
        buckets = len(set(all_durations))  # for plotting only
        summary_manager.add_histogram(values=all_durations, tag='ExtractedDurations', buckets=buckets)

    if not args.skip_char_pitch:
        def _pitch_per_char(pitch, _durations, mel_len, speaker_id):
            durs_cum = np.cumsum(np.pad(_durations, (1, 0)))
            pitch_char = np.zeros((_durations.shape[0],), dtype=np.float)
            for idx, a, b in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
                values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
                values = values[np.where((values * pitch_stats[speaker_id]['pitch_std'] + pitch_stats[speaker_id]['pitch_mean']) < 400)[0]]
                pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0
            return pitch_char


        def process_per_char_pitch(sample_name: str, speaker_id: int):
            # noinspection PyBroadException
            try:
                pitch = np.load((config.pitch_dir / sample_name).with_suffix('.npy').as_posix())
                _durations = np.load((config.duration_dir / sample_name).with_suffix('.npy').as_posix())
                _mel = np.load((config.mel_dir / sample_name).with_suffix('.npy').as_posix())
                char_wise_pitch = _pitch_per_char(pitch, _durations, _mel.shape[0], speaker_id)
                np.save((config.pitch_per_char / sample_name).with_suffix('.npy').as_posix(), char_wise_pitch)
            except Exception:
                logger.exception(f"Failed to process {sample_name}")


        metadatareader = DataReader.from_config(config, kind='phonemized', scan_wavs=False)
        filenames = metadatareader.filenames
        speakers = [metadatareader.text_dict[filename][1] for filename in filenames]
        pitch_stats = {}
        for speaker in set(speakers):
            pitch_stats[speaker] = pickle.load(open(config.data_dir / f'pitch_stats_{speaker}.pkl', 'rb'))
        logger.info(f'Computing phoneme-wise pitch')
        logger.info(f'{len(metadatareader.filenames)} items found in {metadatareader.metadata_path}.')
        wav_iter = p_umap(process_per_char_pitch, metadatareader.filenames, speakers)

    logger.info('Done.')
