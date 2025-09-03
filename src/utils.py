import logging


def log_event(logger: logging.Logger, level: str, message: str, **kwargs):
    if level == logging.INFO:
        logger.info(message, kwargs)
    elif level == logging.WARNING:
        logger.warning(message, kwargs)
    elif level == logging.ERROR:
        logger.error(message, kwargs)
    elif level == logging.DEBUG:
        logger.debug(message, kwargs)
    elif level == logging.CRITICAL:
        logger.critical(message, kwargs)
    else:
        raise ValueError(f"Unknown log level: {level}")
