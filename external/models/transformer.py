import copy
import json
import math
import re
import collections
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from comment_removal.utils import configure_colored_logging

logger = logging.getLogger(__name__)
configure_colored_logging(logger, loglevel='debug')


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    ))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


def warmup_cosine(x, warmup=0.002):
    s = 1 if x <= warmup else 0
    return s*(x/warmup) + (1-s)*(0.5 * (1 + torch.cos(math.pi * x)))


def warmup_constant(x, warmup=0.002):
    s = 1 if x <= warmup else 0
    return s*(x/warmup) + (1-s)*1


def warmup_linear(x, warmup=0.002):
    s = 1 if x <= warmup else 0
    return (s*(x/warmup) + (1-s))*(1-x)


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


class OpenAIAdam(Optimizer):
    """Implements Open AI version of Adam algorithm with weight decay fix."""

    def __init__(self, params, lr, schedule, warmup, t_total,
                 b1=0.9, b2=0.999, e=1e-8, l2=0,
                 vector_l2=False, max_grad_norm=-1, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0 <= warmup:
            raise ValueError("Invalid warmup: {}".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {}".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {}".format(b2))
        if not 0.0 <= e:
            raise ValueError("Invalid epsilon value: {}".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, l2=l2, vector_l2=vector_l2,
                        max_grad_norm=max_grad_norm)
        super(OpenAIAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['b1'], group['b2']

                state['step'] += 1

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['e'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                schedule_fct = SCHEDULES[group['schedule']]
                lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Add weight decay at the end (fixed version)
                if (len(p.size()) > 1 or group['vector_l2']) and group['l2'] > 0:
                    p.data.add_(-lr_scheduled * group['l2'], p.data)

        return loss


class LayerNorm(nn.Module):
    """Construct a layernorm module in the OpenAI style
    (epsilon inside the square root)."""

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state
        #  from Block to Attention to keep identical to TF implem]
        assert n_state % cfg.n_head == 0
        self.register_buffer(
            'b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.b + -1e9 * (1 - self.b)
        # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.b[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -1e9 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = cfg.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.afn]
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg.n_embd
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class TransformerModel(nn.Module):
    """ Transformer model """

    def __init__(self, cfg, vocab=40990, n_ctx=512):
        super(TransformerModel, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg.n_embd)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        block = Block(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block)
                                for _ in range(cfg.n_layer)])

        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x):
        x = x.view(-1, x.size(-2), x.size(-1))
        e = self.drop(self.embed(x))
        # Add the position information to the input embeddings
        h = e.sum(dim=2)
        for block in self.h:
            h = block(h)
        return h


class LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model, cfg, trunc_and_reshape=True):
        super(LMHead, self).__init__()
        self.n_embd = cfg.n_embd
        embed_shape = model.embed.weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight  # Tied weights
        self.trunc_and_reshape = trunc_and_reshape  # XD

    def forward(self, h):
        # Truncated Language modeling logits (we remove the last token)
        h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd) \
            if self.trunc_and_reshape else h  # XD
        lm_logits = self.decoder(h_trunc)
        return lm_logits


class MultipleChoiceHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, clf_token, cfg):
        super(MultipleChoiceHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token

        # To reproduce the noise_shape parameter of TF implementation
        self.dropout = nn.Dropout2d(cfg.clf_pdrop)
        self.linear = nn.Linear(cfg.n_embd, 1)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        # Classification logits
        clf_h = h.view(-1, self.n_embd)
        flat = x[..., 0].contiguous().view(-1)
        clf_h = clf_h[flat == self.clf_token, :]
        clf_h = clf_h.view(-1, x.size(1), self.n_embd, 1)
        # This double transposition is there to replicate the behavior
        # of the noise_shape argument in the tensorflow
        # implementation.  For more details, see
        # https://github.com/huggingface/pytorch-openai-transformer-lm/issues/11
        clf_h = self.dropout(clf_h.transpose(1, 2)).transpose(1, 2)
        clf_h = clf_h.contiguous().view(-1, self.n_embd)
        clf_logits = self.linear(clf_h)

        return clf_logits.view(-1, x.size(1))


class ClfHead(nn.Module):
    """Classification Head for the transformer

    TODO: test this class."""

    def __init__(self, clf_token, cfg, n_class):
        super(ClfHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = nn.Dropout(cfg.clf_pdrop)
        self.linear = nn.Linear(cfg.n_embd, n_class)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        clf_h = h.view(-1, self.n_embd)             # B*E x n_embed
        flat = x[..., 0].contiguous().view(-1)      # B*E
        clf_h = clf_h[flat == self.clf_token, :]    # last slice
        clf_h = self.dropout(clf_h)
        clf_logits = self.linear(clf_h)             # B x n_embed

        return clf_logits


class SimilarityHead(nn.Module):
    """ Similarity Head for the transformer

        TODO: test this class."""

    def __init__(self, clf_token, cfg):
        super(SimilarityHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = nn.Dropout(cfg.clf_pdrop)
        self.linear = nn.Linear(cfg.n_embd, 1)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        sim_h = h.view(-1, self.n_embd)
        flat = x[..., 0].contiguous().view(-1)
        sim_h = sim_h[flat == self.clf_token, :]
        sim_h = self.dropout(sim_h)
        sim_h = sim_h.sum(dim=1)
        sim_logits = self.linear(sim_h)

        return sim_logits


# XD
class LMModel(nn.Module):
    """ Transformer with language model head only """

    def __init__(self, cfg, vocab=40990, n_ctx=512, return_probs=False):
        super(LMModel, self).__init__()
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.lm_head = LMHead(self.transformer, cfg, trunc_and_reshape=False)
        self.return_probs = return_probs
        if self.return_probs:
            pos_emb_mask = torch.zeros(1, 1, vocab)
            pos_emb_mask[:, :, -n_ctx:] = -1e12
            self.register_buffer('pos_emb_mask', pos_emb_mask)

    def forward(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        if self.return_probs:
            lm_logits = F.softmax(lm_logits + self.pos_emb_mask, dim=-1)
        return lm_logits


class DoubleHeadModel(nn.Module):
    """ Transformer with language model and task specific heads """

    def __init__(self, cfg, clf_token, task_head_type, vocab=40990, n_ctx=512):
        super(DoubleHeadModel, self).__init__()
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.lm_head = LMHead(self.transformer, cfg)
        if isinstance(task_head_type, str):
            if task_head_type == 'multiple_choice':
                self.task_head = MultipleChoiceHead(clf_token, cfg)
            elif task_head_type == 'similarity':
                self.task_head = SimilarityHead(clf_token, cfg)
            elif task_head_type == 'inference':
                # the three classes correspond to entailment, contradiction and neutral.
                self.task_head = ClfHead(clf_token, cfg, 3)
            else:
                raise ValueError("task_head_type is expected to be 'multiple_choice' "
                                 "'similarity', 'inference' or ('classification', n_class) "
                                 f"got {task_head_type}.")
        elif isinstance(task_head_type, collections.abc.Sequence) and len(task_head_type) == 2 and \
                task_head_type[0] == 'classification':
            n_class = task_head_type[1]
            self.task_head = ClfHead(clf_token, cfg, n_class)
        else:
            raise ValueError("task_head_type is expected to be 'multiple_choice' "
                             "'similarity', 'inference' or ('classification', n_class) "
                             f"got {task_head_type}.")

    def forward(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        task_logits = self.task_head(h, x)

        return lm_logits, task_logits


class ClassificationLossCompute:
    """A Loss compute and train function for classification tasks."""

    def __init__(self, lm_criterion, clf_criterion, lm_coef, opt=None):
        self.lm_criterion = lm_criterion
        self.clf_criterion = clf_criterion
        self.lm_coef = lm_coef
        self.opt = opt

    def __call__(self, X, Y, M, clf_logits,
                 lm_logits=None, only_return_losses=False):
        # Language modeling loss
        if lm_logits is not None:
            x_shifted = X[:, 1:, 0].contiguous().view(-1)
            M = M.view(-1, M.size(-1))
            lm_losses = self.lm_criterion(lm_logits, x_shifted)
            lm_losses = lm_losses.view(X.size(0), X.size(-2) - 1)
            lm_losses = lm_losses * M[:, 1:]
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        # Classification loss
        clf_losses = self.clf_criterion(clf_logits, Y)
        if only_return_losses:
            return (clf_losses, lm_losses) if lm_logits is not None \
                else clf_losses

        if self.lm_coef > 0 and lm_logits is not None:
            train_loss = clf_losses.sum() + self.lm_coef * lm_losses.sum()
        else:
            train_loss = clf_losses.sum()
        train_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()


class ClassifierModel(nn.Module):
    def __init__(self, args, cfg, clf_token, n_classes=2):
        super(ClassifierModel, self).__init__()

        # prepare the model: Transformer + Classification Head
        self.transformer = TransformerModel(cfg)
        self.task_head = ClfHead(clf_token, cfg, n_classes)

    def make_loss_func(self, args, n_updates_total):
        # prepare loss function
        criterion = nn.CrossEntropyLoss(reduce=False)
        self.optim = self._build_optimizer(args,
                                           n_updates_total)
        self.loss = ClassificationLossCompute(criterion,
                                              criterion,
                                              args.lm_coef,
                                              self.optim)

    def _build_optimizer(self, args, n_updates_total):
        return OpenAIAdam(self.parameters(),
                          lr=args.lr,
                          schedule=args.lr_schedule,
                          warmup=args.lr_warmup,
                          t_total=n_updates_total,
                          b1=args.b1,
                          b2=args.b2,
                          e=args.e,
                          l2=args.l2,
                          vector_l2=args.vector_l2,
                          max_grad_norm=args.max_grad_norm)

    def load_pretrained(self, args):
        load_openai_pretrained_model(self.transformer,
                                     path=args.weights,
                                     path_names=args.weights)

    def forward(self, x):
        h = self.transformer(x)
        task_logits = self.task_head(h, x)
        return task_logits

    def train_batch(self, x_batch, y_batch):
        self.train()
        clf_logits = self.forward(x_batch)
        # TODO: Propagating error signal through the clf-head
        # but not through the transformer.
        return self.loss(None, y_batch, None,
                         clf_logits, lm_logits=None)


def load_openai_pretrained_model(model, n_ctx=-1, n_special=-1, n_transfer=12,
                                 n_embd=768, path='./model/',
                                 path_names='./'):
    # Load weights from TF model
    logger.info("Loading Transformer weights...")
    names = json.load(open(path_names + 'parameters_names.json'))
    shapes = json.load(open(path + 'params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(path + 'params_{}.npy'.format(n))
                   for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape)
                   for param, shape in zip(init_params, shapes)]
    if n_ctx > 0:
        init_params[0] = init_params[0][:n_ctx]
    if n_special > 0:
        init_params[0] = np.concatenate(
            [init_params[1],
             (np.random.randn(n_special, n_embd) * 0.02).astype(np.float32),
             init_params[0]
             ], 0)
    else:
        init_params[0] = np.concatenate(
            [init_params[1],
             init_params[0]
             ], 0)
    del init_params[1]
    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1 + n_transfer * 12
    init_params = [arr.squeeze() for arr in init_params]

    try:
        assert model.embed.weight.shape == init_params[0].shape
    except AssertionError as e:
        e.args += (model.embed.weight.shape, init_params[0].shape)
        raise

    model.embed.weight.data = torch.from_numpy(init_params[0])

    for name, ip in zip(names[1:n_transfer], init_params[1:n_transfer]):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ":0"
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == ip.shape
        except AssertionError as e:
            e.args += (pointer.shape, ip.shape)
            raise
        pointer.data = torch.from_numpy(ip)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DEFAULT_CONFIG = dotdict({
    'n_embd': 768,
    'n_head': 12,
    'n_layer': 12,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'afn': 'gelu',
    'clf_pdrop': 0.1})
