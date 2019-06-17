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
    input_group.add_argument('--predictions-file',
                             default="./results/test_predictions.csv",
                             help='CSV test output file')
    input_group.add_argument('--workdir', default="./workdir",
                             help='temporary work directory')

    # Add encoder options
    encoder_group = parser.add_argument_group('Encoder Options')
    encoder_group.add_argument('--encoder-type', default="LSI")
    encoder_group.add_argument('--encoder', type=str,
                               default=("./external/models/LASER"
                                        "/bilstm.93langs.2018-12-26.pt"),
                               help='which encoder to be used')
    encoder_group.add_argument('--bpe-codes', type=str,
                               default=("./external/models/LASER/"
                                        "93langs.fcodes"),
                               help='Apply BPE using specified codes')
    encoder_group.add_argument('--vocab-file', type=str,
                               default=("./external/models/LASER/"
                                        "93langs.fvocab"),
                               help='Apply BPE using specified vocab')
    encoder_group.add_argument('--target-encoding', default='onehot',
                               help="How to encode the target {onehot|laser}")
    encoder_group.add_argument('--buffer-size', type=int, default=100,
                               help='Buffer size (sentences)')
    encoder_group.add_argument('--max-tokens', type=int, default=12000,
                               help='Max num tokens to process in a batch')
    encoder_group.add_argument('--max-sentences', type=int, default=None,
                               help='Max num sentences to process in a batch')
    encoder_group.add_argument('--cpu', action='store_true',
                               help='Use CPU instead of GPU')
    encoder_group.add_argument('--parallel', action='store_true',
                               help='parallel text processing')

    # Add classifier options
    classifier_group = parser.add_argument_group('Classifier options')
    classifier_group.add_argument('--clf-type', default='mlp',
                                  help=('Classifer type to use: '
                                        '{mlp, randomforest, svc}'))
    classifier_group.add_argument('--batch-size', type=int, default=32,
                                  help='Batch size to train the classifier')

    # verbosity options
    verbosity_group = parser.add_argument_group('Verbosity Options')
    verbosity_group.add_argument('-v', "--verbose", action='store_true')
    verbosity_group.add_argument('-d', "--debug", action='store_true',
                                 help="set verbosity to debug")

    return parser


