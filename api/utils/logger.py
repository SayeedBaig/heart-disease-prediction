import logging


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a reusable logger.
    """

    logger = logging.getLogger(name)

    if not logger.handlers:

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

    return logger