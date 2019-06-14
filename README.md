# Comment removal

This repo explores the `comment removal` prediction task using a
sentence embedding mechanism followed by a classifier of choice.

More specifically encoding of `reddit comments` using
[LASER](https://github.com/facebookresearch/LASER) as inputs to different
classifiers (`mlp`, `svm` or `random forest`).

## Dataset

We use the [Reddit comment removal dataset](https://www.kaggle.com/areeves87/rscience-popular-comment-removal)

### Content
The dataset is a csv of about 30k reddit comments made in `/r/science`
between Jan 2017 and June 2018. 10k of the comments were removed by
moderators; the original text for these comments was recovered using the pushshift.io API.
Each comment is a top-level reply to the parent post and has a comment score of 14 or higher.

The dataset comes from Google BigQuery, Reddit, and Pushshift.io.

## Structure

```bash
# tree -L 3 -I "*.pyc|*cache*|*init*"
.
├── comment_removal
├── data
│   ├── reddit_test.csv.zip
│   └── reddit_train.csv.zip
├── requirements.txt
├── scripts
│   └── download_models.sh
├── setup.cfg
└── workdir

```

## How To


### Installation

First download the pretrained models and additional external code:
```bash
    ./scripts/init.sh
```

And follow the instructions in `external/pyBPE` to install the `pyBPE` tool.

Then, install the python dependencies:
```bash
    pip install -r requirements.txt
```
