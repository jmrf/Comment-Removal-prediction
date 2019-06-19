import torch
import random
import numpy as np
from tqdm import tqdm

from comment_removal import build_arg_parser, logger
from comment_removal.encoders import TextEncoder
from comment_removal.utils.loaders import RedditDataLoader
from comment_removal.utils import (configure_colored_logging,
                                   chunk, parallel_shuffle)
from external.models.transformer import (ClassifierModel,
                                         DEFAULT_CONFIG,
                                         dotdict)


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


def encode_dataset(args, data_loader, text_encoder, split, lim=10000):
    x, y = parallel_shuffle(data_loader.get('BODY', split),
                            data_loader.get('REMOVED', split))

    # Encode inputs (comments)
    x_encoded = text_encoder.encode(x[:lim])

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
    labels = np.array(y[:lim])
    logger.debug("Inputs shape: {} | "
                 "labels shape: {}".format(inputs.shape, labels.shape))
    return inputs, labels


def run_epoch(args, model, dataset, device):
    global n_updates
    loss = 0

    train_x, train_y = dataset
    assert len(train_x) == len(train_y)

    for i, batch in tqdm(enumerate(zip(chunk(train_x, args.n_batch),
                                       chunk(train_y, args.n_batch)))):
        # send batch to appropiate device
        x_batch, y_batch = batch
        # set model in training mode
        model.train()
        loss += model.train_batch(
            torch.tensor(x_batch, dtype=torch.long).to(device),
            torch.tensor(y_batch, dtype=torch.long).to(device))
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


def eval_model(args, y_test, y_scores, target_names):
    from sklearn.metrics import classification_report
    from comment_removal.utils.metrics import compute_roc_curve
    from comment_removal.utils.plotting import plot_confidence_historgram

    logger.debug("Prediction probs shape: {}".format(y_scores.shape))

    # Calculate score and clasificatin report
    # TODO: calculate weights for scoring function
    y_pred = np.argmax(y_scores, axis=1)
    print(classification_report(y_test, y_pred, target_names=target_names))

    # ROC metrics:
    logger.debug("Prediction scores: {}".format(y_scores.shape))
    plot_confidence_historgram(y_test, y_scores)

    # To compute the ROC curve we keep only p(removed)
    compute_roc_curve(y_test, y_scores[:, 1])


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

        # predict on the test data
        preds = run_batched_prediction(args, model, test_x, device)

        # evaluation
        eval_model(args, test_y, preds, ['kept', 'removed'])
