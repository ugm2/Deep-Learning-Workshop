"""Tests utils functions."""


def _log_test_title(title, logger):
    line = "=" * len(title)
    logger.info("\n\n%s\n%s\n%s" % (line, title, line))
