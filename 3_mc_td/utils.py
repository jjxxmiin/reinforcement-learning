import logging
from logging import handlers


def get_logger(file_name='log.log'):
    # create logger
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    # formatter handler
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    logger.addHandler(stream_hander)

    # file handler
    log_max_size = 10 * 1024 * 1024
    log_file_count = 20

    file_handler = handlers.RotatingFileHandler(filename=file_name, maxBytes=log_max_size, backupCount=log_file_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger