from comment_removal import build_arg_parser
from comment_removal.utils.loaders import RedditDataLoader
from comment_removal.utils import configure_colored_logging
from comment_removal.utils.mutils import (encode_or_load_data,
                                          make_classifier_and_predict,
                                          load_model, eval_model)

from comment_removal import logger


def get_args():
    """Append command-line options specific to the encode-classify approach."""
    parser = build_arg_parser()

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

    return parser


if __name__ == '__main__':

    args = get_args().parse_args()

    configure_colored_logging(loglevel='debug',
                              logger=logger)

    target_names = ['kept', 'removed']

    if args.task in {'train', 'eval'}:
        # Load dataset
        data_loader = RedditDataLoader(args.train_file, args.test_file)
        # Create preprocessor
        train_set, test_set = encode_or_load_data(args, data_loader)

    # Train a classifier and evaluate on the test set
    if args.task == 'train':
        clf = make_classifier_and_predict(args,
                                          train_set, test_set,
                                          target_names,
                                          args.rseed,
                                          args.clf_type)

    # Load a trained clasifier and evaluate on the test set
    elif args.task == 'eval':
        clf = load_model(args)
        x_test, y_test = test_set
        eval_model(args, clf, x_test, y_test, target_names)

    else:
        logger.error("Task: {} is not a valid task. "
                     "Valid tasks are: {train,eval}".format(args.task))
