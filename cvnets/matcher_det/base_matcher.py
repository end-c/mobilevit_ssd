
import argparse

class BaseMatcher(object):
    """
    Base class for matching anchor boxes and labels for the task of object detection
    """
    def __init__(self, opts, *args, **kwargs):
        super(BaseMatcher, self).__init__()
        self.opts = opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        "Add class-specific arguments"
        return parser

    def __call__(self, *args, **kwargs):
        raise NotImplementedError