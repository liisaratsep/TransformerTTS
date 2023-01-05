import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


def dynamic_memory_allocation():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # noinspection PyBroadException
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
        except Exception:
            logging.exception("Unknown error.")
