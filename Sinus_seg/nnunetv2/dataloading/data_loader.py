from abc import ABCMeta, abstractmethod
from builtins import object
import warnings
from collections import OrderedDict
from warnings import warn
import numpy as np

class SlimDataLoaderBase(object):
    def __init__(self, data, batch_size, number_of_threads_in_multithreaded=None):
        """
        Slim version of DataLoaderBase (which is now deprecated). Only provides very simple functionality.

        You must derive from this class to implement your own DataLoader. You must overrive self.generate_train_batch()

        If you use our MultiThreadedAugmenter you will need to also set and use number_of_threads_in_multithreaded. See
        multithreaded_dataloading in examples!

        :param data: will be stored in self._data. You can use it to generate your batches in self.generate_train_batch()
        :param batch_size: will be stored in self.batch_size for use in self.generate_train_batch()
        :param number_of_threads_in_multithreaded: will be stored in self.number_of_threads_in_multithreaded.
        None per default. If you wish to iterate over all your training data only once per epoch, you must coordinate
        your Dataloaders and you will need this information
        """
        __metaclass__ = ABCMeta
        self.number_of_threads_in_multithreaded = number_of_threads_in_multithreaded
        self._data = data
        self.batch_size = batch_size
        self.thread_id = 0

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    @abstractmethod
    def generate_train_batch(self):
        '''override this
        Generate your batch from self._data .Make sure you generate the correct batch size (self.BATCH_SIZE)
        '''
        pass


class DataLoader(SlimDataLoaderBase):
    def __init__(self, data, batch_size, num_threads_in_multithreaded=1, seed_for_shuffle=None, return_incomplete=False,
                 shuffle=True, infinite=False, sampling_probabilities=None):
        """

        :param data: will be stored in self._data for use in generate_train_batch
        :param batch_size: will be used by get_indices to return the correct number of indices
        :param num_threads_in_multithreaded: num_threads_in_multithreaded necessary for synchronization of dataloaders
        when using multithreaded augmenter
        :param seed_for_shuffle: for reproducibility
        :param return_incomplete: whether or not to return batches that are incomplete. Only applies is infinite=False.
        If your data has len of 34 and your batch size is 32 then there return_incomplete=False will make this loader
        return only one batch of shape 32 (omitting 2 of your training examples). If return_incomplete=True a second
        batch with batch size 2 will be returned.
        :param shuffle: if True, the order of the indices will be shuffled between epochs. Only applies if infinite=False
        :param infinite: if True, each batch contains randomly (uniformly) sampled indices. An unlimited number of
        batches is returned. If False, DataLoader will iterate over the data only once
        :param sampling_probabilities: only applies if infinite=True. If sampling_probabilities is not None, the
        probabilities will be used by np.random.choice to sample the indexes for each batch. Important:
        sampling_probabilities must have as many entries as there are samples in your dataset AND
        sampling_probabilitiesneeds to sum to 1
        """
        super(DataLoader, self).__init__(data, batch_size, num_threads_in_multithreaded)
        self.infinite = infinite
        self.shuffle = shuffle
        self.return_incomplete = return_incomplete
        self.seed_for_shuffle = seed_for_shuffle
        self.rs = np.random.RandomState(self.seed_for_shuffle)
        self.current_position = None
        self.was_initialized = False
        self.last_reached = False
        self.sampling_probabilities = sampling_probabilities

        # when you derive, make sure to set this! We can't set it here because we don't know what data will be like
        self.indices = None

    def reset(self):
        assert self.indices is not None

        self.current_position = self.thread_id * self.batch_size

        self.was_initialized = True

        # no need to shuffle if we are returning infinite random samples
        if not self.infinite and self.shuffle:
            self.rs.shuffle(self.indices)

        self.last_reached = False

    def get_indices(self):
        # if self.infinite, this is easy
        if self.infinite:
            return np.random.choice(self.indices, self.batch_size, replace=True, p=self.sampling_probabilities)

        if self.last_reached:
            self.reset()
            raise StopIteration

        if not self.was_initialized:
            self.reset()

        indices = []

        for b in range(self.batch_size):
            if self.current_position < len(self.indices):
                indices.append(self.indices[self.current_position])

                self.current_position += 1
            else:
                self.last_reached = True
                break

        if len(indices) > 0 and ((not self.last_reached) or self.return_incomplete):
            self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size
            return indices
        else:
            self.reset()
            raise StopIteration

    @abstractmethod
    def generate_train_batch(self):
        '''
        make use of self.get_indices() to know what indices to work on!
        :return:
        '''
        pass