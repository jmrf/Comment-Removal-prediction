import logging
import pandas as pd


logger = logging.getLogger(__name__)


def read_csv(file_path):
    return pd.read_csv(file_path,
                       encoding="ISO-8859-1")


class RedditDataLoader():

    VALID_KEYS = {'BODY', 'REMOVED'}
    VALID_SETS = set()

    def __init__(self, train_file=None, test_file=None):
        self.train_file = train_file
        self.test_file = test_file
        self.train_df = None
        self.test_df = None
        self._load_dataset()

    def _load_dataset(self):
        """ Loads train and test dataset from CSV files """
        try:
            # Trainig data
            if self.train_file is not None:
                logger.info(
                    "Loading training data from: {}".format(self.train_file))
                self.train_df = read_csv(self.train_file)
                logger.info("Train labels count:\n{}".format(
                    self.train_df['REMOVED'].value_counts())
                )
                self.VALID_SETS.add('train')

            # Test data
            if self.test_file is not None:
                logger.info(
                    "Loading testing data from: {}".format(self.test_file))
                self.test_df = read_csv(self.test_file)
                logger.info("Test labels count:\n{}".format(
                    self.test_df['REMOVED'].value_counts())
                )
                self.VALID_SETS.add('test')
        except Exception as e:
            logger.error("Error while reading Stance dataset!")
            logger.exception(e)

    def get(self, key, set='train'):
        if key not in self.VALID_KEYS:
            raise ValueError("{} is not a valid dataset field. "
                             "Valid fields: {}".format(key, self.VALID_KEYS))

        if set not in self.VALID_SETS:
            raise ValueError("{} is not a valid set. "
                             "Must be one of {}".format(key, self.VALID_SETS))

        if set == 'train':
            return self.train_df[key].tolist()

        if set == 'test':
            return self.test_df[key].tolist()

        return []
