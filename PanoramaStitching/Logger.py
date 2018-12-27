import sys
from enum import Enum
import datetime
import traceback


class LogLevel(Enum):
    """
    Enum for different types of log messages.
    """
    INFO = 0
    DEBUG = 1
    ERROR = 2
    NONE = 3
    STATUS = 4

    def get_color(self):
        """
        Get console color code for log level
        :return: color code escape string
        """
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
    """
    Stdout/stderr logger
    """
    def __init__(self, print_debug=False, print_time=False):
        """

        :param print_debug: print messages marked LogLevel.DEBUG
        :param print_time: print time with the message
        """
        self.print_debug = print_debug
        self.print_time = print_time

    def log(self, log_level, message):
        if not self.print_debug and log_level == LogLevel.DEBUG:
            return

        tab_form = "\t\t"
        if log_level == LogLevel.STATUS:
            tab_form = "\t"

        if log_level == LogLevel.NONE:
            to_print = message
        else:
            if self.print_time:
                to_print = log_level.get_color() + "[" + log_level.name + ":" + tab_form + str(datetime.datetime.now()) + "]\t" + message + BColors.ENDC
            else:
                to_print = log_level.get_color() + "[" + log_level.name + "]" + tab_form + message + BColors.ENDC

        if log_level == LogLevel.ERROR:
            print(to_print, file=sys.stderr, flush=True)
        else:
            print(to_print, flush=True)

    def log_exc(self, exception: Exception):
        tab_form = "\t\t"
        message = "Error occured, exception type: " + exception.__class__.__name__ + ", exception contents: " + str(
                                exception.args)
        if self.print_time:
            to_print = LogLevel.ERROR.get_color() + "[" + LogLevel.ERROR.name + ":" + tab_form + str(
                datetime.datetime.now()) + "]\t" + message + BColors.ENDC
        else:
            to_print = LogLevel.ERROR.get_color() + "[" + LogLevel.ERROR.name + "]" + tab_form + message + BColors.ENDC

        print(to_print, file=sys.stderr, flush=True)

        traceback.print_exc()

    def set_debug(self, value):
        self.print_debug = value


logger_instance = Logger(print_debug=False, print_time=False)
