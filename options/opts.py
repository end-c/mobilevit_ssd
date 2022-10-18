
import argparse
from typing import Optional

from utils import logger
from options.utils import load_config_file
from data.sampler import arguments_sampler
from data.datasets import arguments_dataset
from data.collate_fns import arguments_collate_fn
from data.transforms import arguments_augmentation

from cvnets.anchor_generator import arguments_anchor_gen
from cvnets.matcher_det import arguments_box_matcher
from cvnets import

def get_training_arguments(parse_args: Optional[bool] = True):
    parser = argparse.ArgumentParser(description="Training arguments", add_help=True)

    # sampler related arguments
    parser = arguments_sampler(parser=parser)

    # dataset related arguments
    parser = arguments_dataset(parser=parser)

    # anchor generator arguments
    parser = arguments_anchor_gen(parser=parser)

    # arguments related to box matcher
    parser = arguments_box_matcher(parser=parser)

    # collate fn  related arguments
    parser = arguments_collate_fn(parser=parser)

    # transform related arguments
    parser = arguments_augmentation(parser=parser)

    # model related arguments
    parser = arguments_nn_layers(parser=parser)
    parser = arguments_model(parser=parser)
    parser = arguments_ema(parser=parser)

