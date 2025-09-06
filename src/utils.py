import logging
from pprint import pformat


def log_event(logger: logging.Logger, level: str, message: str, **kwargs):
    if kwargs:
        message = f"{message} - {pformat(kwargs)}"
    if level == logging.INFO:
        logger.info(message)
    elif level == logging.WARNING:
        logger.warning(message)
    elif level == logging.ERROR:
        logger.error(message)
    elif level == logging.DEBUG:
        logger.debug(message)
    elif level == logging.CRITICAL:
        logger.critical(message)
    else:
        raise ValueError(f"Unknown log level: {level}")
