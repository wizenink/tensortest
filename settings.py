import configparser
import os

class EnvInterpolation(configparser.BasicInterpolation):
    """Interpolation which expands environment variables in values."""

    def before_get(self, parser, section, option, value, defaults):
        return os.path.expandvars(value)

def init():
    global config
    config = configparser.ConfigParser(interpolation=EnvInterpolation())
    config.read('colorize.cfg')