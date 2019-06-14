import logging
import argparse

logger = logging.getLogger(__name__)


def build_arg_parser():
    parser = argparse.ArgumentParser('Comment Removal Prediction task')

    parser.add_argument('task', type=str, default='train',
                        help="one of {train, eval}")
    parser.add_argument('--rseed', type=int, default=123,
                        help="Random seed ")

    input_group = parser.add_argument_group('Input Data Options')
    input_group.add_argument('--train-file',
                             default="./data/reddit_train.csv",
                             help='CSV training input file')
    input_group.add_argument('--test-file',
                             default="./data/reddit_test.csv",
                             help='CSV test input file')
    input_group.add_argument('--workdir', default="./workdir",
                             help='temporary work directory')

    # verbosity options
    verbosity_group = parser.add_argument_group('Verbosity Options')
    verbosity_group.add_argument('-v', "--verbose", action='store_true')
    verbosity_group.add_argument('-d', "--debug", action='store_true',
                                 help="set verbosity to debug")

    return parser


