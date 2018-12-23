import sys
from enum import Enum
import datetime


class LogLevel(Enum):
    INFO = 0
    DEBUG = 1
    ERROR = 2
    NONE = 3
    STATUS = 4

    def get_color(self):
        if self == LogLevel.INFO:
            return BColors.OKBLUE
        if self == LogLevel.DEBUG:
            return BColors.WARNING
        if self == LogLevel.NONE:
            return BColors.OKGREEN
        if self == LogLevel.ERROR:
            return BColors.FAIL
        if self == LogLevel.STATUS:
            return BColors.HEADER


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Logger:

    def __init__(self, print_debug=False, print_time=False):
        self.print_debug = print_debug
        self.print_time = print_time

    def log(self, log_level, message):
        if not self.print_debug and log_level == LogLevel.DEBUG:
            return

        color = log_level.get_color()

        tab_form = "\t\t"
        if log_level == LogLevel.STATUS:
            tab_form = "\t"

        if log_level == LogLevel.NONE:
            to_print = message
        else:
            if self.print_time:
                to_print = color + "[" + log_level.name + ":" + tab_form + str(datetime.datetime.now()) + "]\t" + message + BColors.ENDC
            else:
                to_print = color + "[" + log_level.name + "]" + tab_form + message + BColors.ENDC

        if log_level == LogLevel.ERROR:
            print(to_print, file=sys.stderr)
        else:
            print(to_print)


logger_instance = Logger(print_debug=True, print_time=True)
