# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Classes for datasets and batches."""
import os

from magenta.models.coconet import lib_mask
from magenta.models.coconet import lib_pianoroll
from magenta.models.coconet import lib_util
import numpy as np
import tensorflow.compat.v1 as tf

import pickle

class Dataset(lib_util.Factory):
  """Class for retrieving different datasets."""

  def __init__(self, basepath, hparams, fold, embedding_name=None):
    """Initialize a `Dataset` instance.

    Args:
      basepath: path to directory containing dataset npz files.
      hparams: Hyperparameters object.
      fold: data subset, one of {train,valid,test}.

    Raises:
      ValueError: if requested a temporal resolution shorter then that available
          in the dataset.
    """
    self.basepath = basepath
    self.hparams = hparams
    self.fold = fold

    if self.shortest_duration != self.hparams.quantization_level:
      raise ValueError("The data has a temporal resolution of shortest "
                       "duration=%r, requested=%r" %
                       (self.shortest_duration,
                        self.hparams.quantization_level))

    # Update the default pitch ranges in hparams to reflect that of dataset.
    hparams.pitch_ranges = [self.min_pitch, self.max_pitch]
    hparams.shortest_duration = self.shortest_duration
    self.encoder = lib_pianoroll.get_pianoroll_encoder_decoder(hparams)
    # data_path = os.path.join(tf.resource_loader.get_data_files_path(),
    #                          self.basepath, "%s.npz" % self.name)
    data_path = os.path.join(tf.resource_loader.get_data_files_path(),
                             self.basepath, self.name)
    print("Loading data from", data_path)
    # with tf.gfile.Open(data_path, "rb") as p:
    #   self.data = np.load(p, allow_pickle=True, encoding="latin1")[fold]

    # load slices dataset
    with open(data_path, "rb") as f:
      self.data = pickle.load(f)[fold]

    self.embedding_name = embedding_name
    if self.embedding_name:
      embedding_path = os.path.join(tf.resource_loader.get_data_files_path(), self.basepath, self.embedding_name)
      with open(embedding_path, "rb") as f:
        self.embeddings = pickle.load(f)
      # self.embeddings = np.zeros(self.embeddings.shape)
      # for i in range(len(self.embeddings)):
      #   self.embeddings[i][1] = 1
      # print(self.embeddings[20])
      # print(self.embeddings[20].sum())
      # exit()

  @property
  def name(self):
    return self.hparams.dataset

  @property
  def num_examples(self):
    return len(self.data)

  @property
  def num_pitches(self):
    return self.max_pitch + 1 - self.min_pitch

  def get_sequences(self):
    """Return the raw collection of examples."""
    return self.data

  def get_pianorolls(self, sequences=None):
    """Turn sequences into pianorolls.

    Args:
      sequences: the collection of sequences to convert. If not given, the
          entire dataset is converted.

    Returns:
      A list of multi-instrument pianorolls, each shaped
          (duration, pitches, instruments)
    """
    if sequences is None:
      sequences = self.get_sequences()
    return list(map(self.encoder.encode, sequences))

  def get_featuremaps(self, sequences=None):
    """Turn sequences into features for training/evaluation.

    Encodes sequences into randomly cropped and masked pianorolls, and returns
    a padded Batch containing three channels: the pianorolls, the corresponding
    masks and their lengths before padding (but after cropping).

    Args:
      sequences: the collection of sequences to convert. If not given, the
          entire dataset is converted.

    Returns:
      A Batch containing pianorolls, masks and piece lengths.
    """
    if sequences is None:
      sequences = self.get_sequences()

    pianorolls = []
    masks = []

    for sequence in sequences:
      pianoroll = self.encoder.encode(sequence, self.embeddings)
      pianoroll = lib_util.random_crop(pianoroll, self.hparams.crop_piece_len)
      mask = lib_mask.get_mask(
          self.hparams.maskout_method,
          pianoroll.shape,
          separate_instruments=self.hparams.separate_instruments,
          blankout_ratio=self.hparams.corrupt_ratio)
      pianorolls.append(pianoroll)
      masks.append(mask)
    (pianorolls, masks), lengths = lib_util.pad_and_stack(pianorolls, masks)
    assert pianorolls.ndim == 4 and masks.ndim == 4
    assert pianorolls.shape == masks.shape
    # print('test')
    # assert not np.any(np.isnan(pianorolls))
    # print(not np.any(np.isnan(pianorolls)))
    # import pickle
    # with open('pianorolls_test', 'wb') as f:
    #   pickle.dump(pianorolls, f)
    # with open('batch_test', 'wb') as f:
    #   pickle.dump(Batch(pianorolls=pianorolls, masks=masks, lengths=lengths), f)
    return Batch(pianorolls=pianorolls, masks=masks, lengths=lengths)

  def update_hparams(self, hparams):
    """Update subset of Hyperparameters pertaining to data."""
    for key in "num_instruments min_pitch max_pitch qpm".split():
      setattr(hparams, key, getattr(self, key))


def get_dataset(basepath, hparams, fold, embedding_name):
  """Factory for Datasets."""
  return Dataset.make(hparams.dataset, basepath, hparams, fold, embedding_name)


class Jsb16thSeparated(Dataset):
  key = "Jsb16thSeparated"
  min_pitch = 36
  max_pitch = 81
  shortest_duration = 0.125
  num_instruments = 4
  qpm = 60

# class embedding_test(Dataset):
#   key = "embedding_test"
#   min_pitch = 0
#   max_pitch = 999
#   shortest_duration = 0.125
#   num_instruments = 4
#   qpm = 60

# class embedding_test_20int(Dataset):
#   key = "embedding_test_20int"
#   min_pitch = 0
#   max_pitch = 19
#   shortest_duration = 0.125
#   num_instruments = 1000
#   qpm = 60

class midi_dataset_1000(Dataset):
  key = "midi_dataset_1000"
  min_pitch = 0
  max_pitch = 127
  shortest_duration = 0.125
  num_instruments = 1
  qpm = 60


class TestData(Dataset):
  key = "TestData"
  min_pitch = 0
  max_pitch = 127
  shortest_duration = 0.125
  num_instruments = 4
  qpm = 60


class Batch(object):
  """A Batch of training/evaluation data."""

  keys = set("pianorolls masks lengths".split())

  def __init__(self, **kwargs):
    """Initialize a Batch instance.

    Args:
      **kwargs: data dictionary. Must have three keys "pianorolls", "masks",
          "lengths", each corresponding to a model placeholder. Each value
          is a sequence (i.e. a batch) of examples.
    """
    assert set(kwargs.keys()) == self.keys
    assert all(
        len(value) == len(list(kwargs.values())[0])
        for value in kwargs.values())
    self.features = kwargs

  def get_feed_dict(self, placeholders):
    """Zip placeholders and batch data into a feed dict.

    Args:
      placeholders: placeholder dictionary. Must have three keys "pianorolls",
          "masks" and "lengths".

    Returns:
      A feed dict mapping the given placeholders to the data in this batch.
    """
    assert set(placeholders.keys()) == self.keys
    return dict((placeholders[key], self.features[key]) for key in self.keys)

  def batches(self, **batches_kwargs):
    """Iterate over sub-batches of this batch.

    Args:
      **batches_kwargs: kwargs passed on to lib_util.batches.

    Yields:
      An iterator over sub-Batches.
    """
    keys, values = list(zip(*list(self.features.items())))
    for batch in lib_util.batches(*values, **batches_kwargs):
      yield Batch(**dict(lib_util.eqzip(keys, batch)))
