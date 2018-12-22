import sys
from enum import Enum


class LogLevel(Enum):
    INFO = 0
    DEBUG = 1
    ERROR = 2
    NONE = 3


class Logger:

    def __init__(self, print_debug=False):
        self.print_debug = print_debug

    def log(self, log_level, message):
        if not self.print_debug and log_level == LogLevel.DEBUG:
            return

        if log_level == LogLevel.NONE:
            to_print = message
        else:
            to_print = "[" + log_level.name + "] " + message
        if log_level == LogLevel.ERROR:
            print(to_print, file=sys.stderr)
        else:
            print(to_print)


logger_instance = Logger(print_debug=True)
