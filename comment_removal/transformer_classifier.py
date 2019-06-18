import random
import numpy as np
import torch

from comment_removal import build_arg_parser, logger
from comment_removal.utils.loaders import RedditDataLoader
from comment_removal.utils import configure_colored_logging
from external.models.transformer import (ClassifierModel,
                                         DEFAULT_CONFIG)


def freeze_seeds(args):
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)
    torch.cuda.manual_seed_all(args.rseed)


def build_encoder():
    return {'_classify_': 1}    # TODO: Dummy encoder for now


def build_model(args, encoder):
    clf_token = encoder['_classify_']
    model = ClassifierModel(DEFAULT_CONFIG, clf_token)
    model.load_pretrained(args)
    return model


def get_args():
    """Append command-line options specific to the encode-classify approach."""
    parser = build_arg_parser()

    # Add transformer options
    transformer_group = parser.add_argument_group('Transformer Options')
    transformer_group.add_argument('--n_embd', type=int, default=768,
                                   help="Embedding dimension")
    transformer_group.add_argument('--weights',
                                   default="external/models/transformer/",
                                   help="pretrained weights directory")

    return parser


if __name__ == '__main__':

    args = get_args().parse_args()

    # prepare random seed and logging
    freeze_seeds(args)
    configure_colored_logging(loglevel='debug', logger=logger)

    # load pretrained openAI model
    model = build_model(args, build_encoder())
    logger.debug(model)

    x = torch.LongTensor(np.arange(250).reshape(5, -1))
    logger.debug(model(x).shape)

    # target_names = ['kept', 'removed']

    # if args.task in {'train', 'eval'}:
    #     # Load dataset
    #     data_loader = RedditDataLoader(args.train_file, args.test_file)

