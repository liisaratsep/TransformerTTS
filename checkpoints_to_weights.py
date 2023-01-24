import logging
from utils.training_config_manager import TrainingConfigManager, TTSMode
from utils.argparser import tts_argparser

MODE = TTSMode.WEIGHTS
parser = tts_argparser(MODE)
args = parser.parse_args()
config = TrainingConfigManager(mode=MODE, **vars(args))

logger = logging.getLogger(__name__)
logger.setLevel(args.log_level)

if __name__ == '__main__':
    model = config.load_model(checkpoint_path=args.checkpoint_path)  # None defaults to latest

    if args.target_dir is None:
        args.target_dir = config.base_dir / 'weights' / f'weights_step_{model.step}'

    model.save_model(args.target_dir)

    logger.info('Done.')
    logger.info(f'Model weights saved under {args.target_dir}')
