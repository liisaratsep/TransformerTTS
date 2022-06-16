from utils.training_config_manager import TrainingConfigManager, TTSMode
from utils.argparser import tts_argparser

MODE = TTSMode.WEIGHTS

if __name__ == '__main__':
    parser = tts_argparser(MODE)
    args = parser.parse_args()

    config = TrainingConfigManager(args.config_path)
    model = config.load_model(checkpoint_path=args.checkpoint_path)  # None defaults to latest
    model.save_model(config.base_dir / 'weights')

    print('Done.')
    print(f'Model weights saved under {config.base_dir / "weights"}')
