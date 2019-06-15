from comment_removal import build_arg_parser
from comment_removal.utils.loaders import RedditDataLoader
from comment_removal.utils import configure_colored_logging
from comment_removal.utils.mutils import (encode_or_load_data,
                                          make_classifier_and_predict,
                                          load_model, eval_model)

from comment_removal import logger


if __name__ == '__main__':

    args = build_arg_parser().parse_args()

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
