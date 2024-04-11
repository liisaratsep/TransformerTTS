import logging

from utils.training_config_manager import TrainingConfigManager, TTSMode
from utils.argparser import tts_argparser

MODE = TTSMode("aligner")
parser = tts_argparser(MODE)
args = parser.parse_args()
config = TrainingConfigManager(mode=MODE, **vars(args))

logger = logging.getLogger(__name__)
logger.setLevel(args.log_level)

if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    from tqdm import trange

    from data.datasets import AlignerDataset, AlignerPreprocessor
    from utils.decorators import ignore_exception, time_it
    from utils.scheduling import piecewise_linear_schedule, reduction_schedule
    from utils.logging_utils import SummaryManager
    from utils.scripts_utils import dynamic_memory_allocation
    from utils.metrics import attention_score
    from utils.spectrogram_ops import mel_lengths, phoneme_lengths
    from utils.alignments import get_durations_from_alignment

    dynamic_memory_allocation()

    if config.seed is not None:
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)


    def cut_with_durations(durations, _mel, _phonemes, snippet_len=10):
        phon_dur = np.pad(durations, (1, 0))
        starts = np.cumsum(phon_dur)[:-1]
        ends = np.cumsum(phon_dur)[1:]
        cut_mels = []
        cut_texts = []
        for end_idx in range(snippet_len, len(phon_dur), snippet_len):
            start_idx = end_idx - snippet_len
            cut_mels.append(_mel[starts[start_idx]: ends[end_idx - 1], :])
            cut_texts.append(_phonemes[start_idx: end_idx])
        return cut_mels, cut_texts


    @ignore_exception
    @time_it
    def validate(_model,
                 val_dataset,
                 _summary_manager,
                 weighted_durations):
        _val_loss = {'loss': 0.}
        norm = 0.
        current_r = _model.r
        _model.set_constants(reduction_factor=1)

        val_mel = val_text = fname = model_out = None

        for val_mel, val_text, val_stop, fname in val_dataset.all_batches():
            model_out = _model.val_step(inp=val_text,
                                        tar=val_mel,
                                        stop_prob=val_stop)
            norm += 1
            _val_loss['loss'] += model_out['loss']
        _val_loss['loss'] /= norm
        _summary_manager.display_loss(model_out, tag='Validation', plot_all=True)
        _summary_manager.display_last_attention(model_out, tag='ValidationAttentionHeads', fname=fname)
        attention_values = model_out['decoder_attention']['Decoder_LastBlock_CrossAttention'].numpy()
        _text = val_text.numpy()
        _mel = val_mel.numpy()
        _model.set_constants(reduction_factor=current_r)
        modes = list({False, weighted_durations})
        for mode in modes:
            durations, final_align, jumpiness, peakiness, _diag_measure = get_durations_from_alignment(
                batch_alignments=attention_values,
                mels=_mel,
                phonemes=_text,
                weighted=mode)
            for _i in range(len(durations)):
                phon_dur = durations[_i]
                i_mel = _mel[_i][1:]  # remove start token (is padded so end token can't be removed/not an issue)
                i_text = _text[_i][1:]  # remove start token (is padded so end token can't be removed/not an issue)
                i_phon = _model.text_pipeline.tokenizer.decode(i_text).replace('/', '')
                cut_mels, cut_texts = cut_with_durations(durations=phon_dur, _mel=i_mel, _phonemes=i_phon)
                for cut_idx, cut_text in enumerate(cut_texts):
                    weighted_label = 'weighted_' * mode
                    _summary_manager.display_audio(
                        tag=f'CutAudio {weighted_label}{fname[_i].numpy().decode("utf-8")}/{cut_idx}/{cut_text}',
                        mel=cut_mels[cut_idx], description=i_phon)
        return _val_loss['loss']


    config.create_remove_dirs(clear_dir=args.clear_dir,
                              clear_logs=args.clear_logs,
                              clear_weights=args.clear_weights)
    config.dump_config()
    config.print_config()

    # get model, prepare data for model, create datasets
    model = config.get_model()
    config.compile_model(model)
    data_prep = AlignerPreprocessor.from_config(config,
                                                tokenizer=model.text_pipeline.tokenizer)
    train_data_handler = AlignerDataset.from_config(config,
                                                    preprocessor=data_prep,
                                                    kind='train')
    valid_data_handler = AlignerDataset.from_config(config,
                                                    preprocessor=data_prep,
                                                    kind='valid')

    train_dataset = train_data_handler.get_dataset(bucket_batch_sizes=config.config['bucket_batch_sizes'],
                                                   bucket_boundaries=config.config['bucket_boundaries'],
                                                   shuffle=True)
    valid_dataset = valid_data_handler.get_dataset(bucket_batch_sizes=config.config['val_bucket_batch_size'],
                                                   bucket_boundaries=config.config['bucket_boundaries'],
                                                   shuffle=False, drop_remainder=True)

    # create logger and checkpointer and restore the latest model

    summary_manager = SummaryManager(model=model, log_dir=config.log_dir, config=config.config)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                     optimizer=model.optimizer,
                                     net=model)
    manager = tf.train.CheckpointManager(checkpoint, str(config.weights_dir),
                                         max_to_keep=config.config['keep_n_weights'],
                                         keep_checkpoint_every_n_hours=config.config[
                                             'keep_checkpoint_every_n_hours'])
    manager_training = tf.train.CheckpointManager(checkpoint, str(config.weights_dir / 'latest'),
                                                  max_to_keep=1, checkpoint_name='latest')

    checkpoint.restore(manager_training.latest_checkpoint)
    if manager_training.latest_checkpoint:
        logger.info(f'resuming training from step {model.step} ({manager_training.latest_checkpoint})')
    else:
        logger.info(f'starting training from scratch')

    if config.config['debug'] is True:
        logger.warning('DEBUG is set to True. Training in eager mode.')
    # main event
    logger.info('TRAINING')
    losses = []

    texts = []
    for text_file in args.test_files:
        with open(text_file, 'r') as file:
            texts.append(file.readlines())

    test_mel, test_phonemes, _, test_fname = valid_dataset.next_batch()
    val_test_sample, val_test_fname, val_test_mel = test_phonemes[0], test_fname[0], test_mel[0]
    val_test_sample = tf.boolean_mask(val_test_sample, val_test_sample != 0)

    _ = train_dataset.next_batch()
    t = trange(model.step, config.config['max_steps'], leave=True)
    for _ in t:
        t.set_description(f'step {model.step}')
        mel, phonemes, stop, sample_name = train_dataset.next_batch()
        learning_rate = piecewise_linear_schedule(model.step, config.config['learning_rate_schedule'])
        reduction_factor = reduction_schedule(model.step, config.config['reduction_factor_schedule'])
        t.display(f'reduction factor {reduction_factor}', pos=10)
        force_encoder_diagonal = model.step < config.config['force_encoder_diagonal_steps']
        force_decoder_diagonal = model.step < config.config['force_decoder_diagonal_steps']
        model.set_constants(learning_rate=learning_rate,
                            reduction_factor=reduction_factor,
                            force_encoder_diagonal=force_encoder_diagonal,
                            force_decoder_diagonal=force_decoder_diagonal)

        output = model.train_step(inp=phonemes,
                                  tar=mel,
                                  stop_prob=stop)
        losses.append(float(output['loss']))

        t.display(f'step loss: {losses[-1]}', pos=1)
        for pos, n_steps in enumerate(config.config['n_steps_avg_losses']):
            if len(losses) > n_steps:
                t.display(f'{n_steps}-steps average loss: {sum(losses[-n_steps:]) / n_steps}', pos=pos + 2)

        summary_manager.display_loss(output, tag='Train')
        summary_manager.display_scalar(tag='Meta/learning_rate', scalar_value=model.optimizer.lr)
        summary_manager.display_scalar(tag='Meta/reduction_factor', scalar_value=model.r)
        summary_manager.display_scalar(scalar_value=t.avg_time, tag='Meta/iter_time')
        summary_manager.display_scalar(scalar_value=tf.shape(sample_name)[0], tag='Meta/batch_size')
        if model.step % config.config['train_images_plotting_frequency'] == 0:
            summary_manager.display_attention_heads(output, tag='TrainAttentionHeads')
            summary_manager.display_mel(mel=output['mel'][0], tag=f'Train/predicted_mel')
            for layer, k in enumerate(output['decoder_attention'].keys()):
                mel_lens = mel_lengths(mel_batch=mel, padding_value=0) // model.r  # [N]
                phon_len = phoneme_lengths(phonemes)
                loc_score, peak_score, diag_measure = attention_score(att=output['decoder_attention'][k],
                                                                      mel_len=mel_lens,
                                                                      phon_len=phon_len,
                                                                      r=model.r)
                loc_score = tf.reduce_mean(loc_score, axis=0)
                peak_score = tf.reduce_mean(peak_score, axis=0)
                diag_measure = tf.reduce_mean(diag_measure, axis=0)
                for i in range(tf.shape(loc_score)[0]):
                    summary_manager.display_scalar(tag=f'TrainDecoderAttentionJumpiness/layer{layer}_head{i}',
                                                   scalar_value=tf.reduce_mean(loc_score[i]))
                    summary_manager.display_scalar(tag=f'TrainDecoderAttentionPeakiness/layer{layer}_head{i}',
                                                   scalar_value=tf.reduce_mean(peak_score[i]))
                    summary_manager.display_scalar(tag=f'TrainDecoderAttentionDiagonality/layer{layer}_head{i}',
                                                   scalar_value=tf.reduce_mean(diag_measure[i]))

        if model.step % 1000 == 0:
            save_path = manager_training.save()
        if model.step % config.config['weights_save_frequency'] == 0:
            save_path = manager.save()
            t.display(f'checkpoint at step {model.step}: {save_path}',
                      pos=len(config.config['n_steps_avg_losses']) + 2)

        if model.step % config.config['validation_frequency'] == 0 and (
                model.step >= config.config['prediction_start_step']):
            val_loss, time_taken = validate(_model=model,
                                            val_dataset=valid_dataset,
                                            _summary_manager=summary_manager,
                                            weighted_durations=config.config['extract_attention_weighted'])
            t.display(f'validation loss at step {model.step}: {val_loss} (took {time_taken}s)',
                      pos=len(config.config['n_steps_avg_losses']) + 3)

        if model.step % config.config['prediction_frequency'] == 0 and (
                model.step >= config.config['prediction_start_step']):
            for j, text in enumerate(texts):
                for i, text_line in enumerate(text):
                    text_line = text_line.split('|')
                    if len(text_line) > 1:
                        speaker_id = text_line[1]
                    else:
                        speaker_id = 0
                    out = model.predict(text_line[0], encode=True, speaker_id=speaker_id)
                    wav = summary_manager.audio.reconstruct_waveform(out['mel'].numpy().T)
                    wav = tf.expand_dims(wav, 0)
                    wav = tf.expand_dims(wav, -1)
                    summary_manager.add_audio(f'Predictions/{text_line}', wav.numpy(),
                                              sr=summary_manager.config['sampling_rate'],
                                              step=summary_manager.global_step)

            out = model.predict(val_test_sample, encode=False)  # , max_length=tf.shape(val_test_mel)[-2])
            wav = summary_manager.audio.reconstruct_waveform(out['mel'].numpy().T)
            wav = tf.expand_dims(wav, 0)
            wav = tf.expand_dims(wav, -1)
            summary_manager.add_audio(f'Predictions/val_sample {val_test_fname.numpy().decode("utf-8")}', wav.numpy(),
                                      sr=summary_manager.config['sampling_rate'],
                                      step=summary_manager.global_step)
    logger.info('Done.')
