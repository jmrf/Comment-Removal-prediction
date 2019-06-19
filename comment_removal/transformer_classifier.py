import random
import numpy as np
import torch
from tqdm import tqdm

from comment_removal import build_arg_parser, logger
from comment_removal.encoders import TextEncoder
from comment_removal.utils.loaders import RedditDataLoader
from comment_removal.utils import configure_colored_logging, chunk
from external.models.transformer import (ClassifierModel,
                                         DEFAULT_CONFIG)


def freeze_seeds(args):
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)
    torch.cuda.manual_seed_all(args.rseed)


def build_encoder(args):
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    # Add specia tokens
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)

    return text_encoder, n_vocab


def build_model(args, clf_token, device,
                for_training=True, n_train=None):
    model = ClassifierModel(args, DEFAULT_CONFIG, clf_token)
    model.load_pretrained(args)

    if for_training:
        n_updates_total = (n_train // args.n_batch) * args.n_iter
        model.make_loss_func(args, n_updates_total)

    model.to(device)
    return model


def encode_dataset(args, data_loader, text_encoder, split):
    # Encode inputs (comments)
    x_encoded = text_encoder.encode(data_loader.get('BODY', split)[:50])

    # Padding
    N = len(x_encoded)
    max_len = args.n_ctx - 2  # substract due to control tokens
    start = text_encoder.encoder['_start_']
    clf_token = text_encoder.encoder['_classify_']

    inputs = np.zeros((N, args.n_ctx, 2))
    for i, x in enumerate(x_encoded):
        x = [start] + x[:max_len] + [clf_token]
        inputs[i, :len(x), 0] = x

    # positional encoding info
    n_special = 3  # special chars
    n_vocab = len(text_encoder.encoder) - n_special

    logger.debug("min: {} - max: {}".format(n_vocab,
                                            n_vocab + args.n_ctx))
    inputs[:, :, 1] = np.arange(n_vocab,
                                n_vocab + args.n_ctx)

    # Encode labels
    labels = np.array(data_loader.get('REMOVED', split)[:50])

    logger.debug("Inputs shape: {} | "
                 "labels shape: {}".format(inputs.shape, labels.shape))
    return inputs, labels


def run_epoch(args, model, dataset, device):
    global n_updates
    loss = 0

    train_x, train_y = dataset
    assert len(train_x) == len(train_y)

    dataset_it = tqdm(zip(chunk(train_x, args.n_batch),
                          chunk(train_y, args.n_batch)))
    for i, batch in enumerate(dataset_it):
        dataset_it.set_description("Batch {}".format(i))
        # send batch to appropiate device
        x_batch, y_batch = batch
        x = torch.tensor(x_batch, dtype=torch.long).to(device)
        y = torch.tensor(y_batch, dtype=torch.long).to(device)
        # set model in training mode
        model.train()
        loss += model.train_batch(x, y)
        # update counter
        n_updates += 1

    return loss / i


def run_batched_prediction(args, model, test_x, device):
    model.eval()
    model.to(device)
    preds = []
    for i, x_batch in enumerate(chunk(test_x, args.n_batch)):
        logger.debug("Pred on batch {}".format(i))
        with torch.no_grad():
            preds.extend(torch.nn.functional.softmax(
                model(torch.tensor(x_batch, dtype=torch.long).to(device))
            ).data.cpu().numpy())

    return np.stack(preds)


def get_args():
    """Append command-line options specific to the encode-classify approach."""
    parser = build_arg_parser()

    # Add encoder options
    encoder_group = parser.add_argument_group('Transformer Options')
    # paths
    encoder_group.add_argument(
        '--encoder_path', type=str,
        default='external/models/transformer/encoder_bpe_40000.json')
    encoder_group.add_argument(
        '--bpe_path', type=str,
        default="external/models/transformer/vocab_40000.bpe")

    # Add transformer options
    transformer_group = parser.add_argument_group('Transformer Options')
    # size configurations
    transformer_group.add_argument('--n_iter', type=int, default=30)
    transformer_group.add_argument('--n_batch', type=int, default=8)
    transformer_group.add_argument('--n_ctx', type=int, default=512)
    transformer_group.add_argument('--n_embd', type=int, default=768,
                                   help="Embedding dimension")
    # optimization parameters
    transformer_group.add_argument('--lm_coef', type=float, default=0.5)
    transformer_group.add_argument('--b1', type=float, default=0.9)
    transformer_group.add_argument('--b2', type=float, default=0.999)
    transformer_group.add_argument('--e', type=float, default=1e-8)
    # L2 norm
    transformer_group.add_argument('--l2', type=float, default=0.01)
    transformer_group.add_argument('--vector_l2', action='store_true')
    # learning rate
    transformer_group.add_argument('--lr', type=float, default=6.25e-5)
    transformer_group.add_argument('--lr_warmup', type=float, default=0.002)
    transformer_group.add_argument('--lr_schedule', default='warmup_linear')
    transformer_group.add_argument('--max_grad_norm', type=int, default=1)
    # paths
    transformer_group.add_argument('--weights',
                                   default="external/models/transformer/",
                                   help="pretrained weights directory")

    return parser


if __name__ == '__main__':

    args = get_args().parse_args()

    # prepare random seed and logging
    freeze_seeds(args)
    configure_colored_logging(loglevel='debug', logger=logger)

    # Load dataset
    data_loader = RedditDataLoader(args.train_file, args.test_file)
    n_train = len(data_loader.train_df)
    n_valid = len(data_loader.test_df)
    logger.info("Total training samples: {}\n"
                "Total testing samples: {}\n".format(n_train, n_valid))

    # Load BPE encoder
    text_encoder, vocab_size = build_encoder(args)
    logger.info("Loaded vocab with size: {}".format(vocab_size))

    # Encode train and test sets
    train_x, train_y = encode_dataset(args, data_loader, text_encoder, 'train')
    # test_x, test_y = encode_dataset(args, data_loader, text_encoder, 'test')

    # load pretrained openAI transformer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    model = build_model(args, text_encoder.encoder['_classify_'], device,
                        for_training=True, n_train=n_train)

    logger.debug(model)

    if args.task in {'train', 'eval'}:

        if args.task == 'train':
            n_updates = 0
            n_epochs = 0
            train_it = tqdm(range(args.n_iter))
            for i in train_it:
                loss = run_epoch(model, (train_x, train_y), device)
                train_it.set_description("Running epoch {} - "
                                         "loss: {:.3f}".format(i, loss))

                n_epochs += 1
