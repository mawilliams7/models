import tensorflow as tf
import tensorflow.feature_column as fc 

import os
import sys
import pandas

import matplotlib.pyplot as plt
from IPython.display import clear_output


_CSV_COLUMNS = [
	'year', 'severity', 'access', 'impact',
	'exploit', 'products', 'vendors', 'problems',
]

_CSV_COLUMN_DEFAULTS = [[0], ['LOW'], ['NETWORK'], [0],
                        [0], [''], [''], ['Other']]

_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train': 80000,
    'validation': 14713,
}
tf.enable_eager_execution()
print("Hello")

train_file = "data/nvd_data.csv"
test_file = "data/nvd_test.csv"

def build_model_columns():
	"""Builds a set of wide and deep feature columns."""
	year = tf.feature_column.numeric_column('year')
	impact = tf.feature_column.numeric_column('impact')
	exploit = tf.feature_column.numeric_column('exploit')
	severity = tf.feature_column.categorical_column_with_vocabulary_list(
		'severity',
			['LOW', 'MEDIUM', 'HIGH'])
	access = tf.feature_column.categorical_column_with_vocabulary_list(
		'access',
			['NETWORK', 'LOCAL', 'ADJACENT'])
	products = tf.feature_column.categorical_column_with_hash_bucket(
		'products', hash_bucket_size = _HASH_BUCKET_SIZE)
	vendors = tf.feature_column.categorical_column_with_hash_bucket(
		'vendors', hash_bucket_size = _HASH_BUCKET_SIZE)

	wide_columns = [
		severity, 
		access, 
		tf.feature_column.indicator_column(products),
		tf.feature_column.indicator_column(vendors),
	]
	
	deep_columns = [
		year,
		impact,
		exploit,
		tf.feature_column.indicator_column(severity),
		tf.feature_column.indicator_column(access),
		tf.feature_column.indicator_column(products),
		tf.feature_column.indicator_column(vendors),
	]
	
	return wide_columns, deep_columns

def input_fn(data_file, num_epochs, shuffle, batch_size):
	"""Generate an input function for the estimator."""
	assert tf.gfile.Exists(data_file), (
    '%s not found. Please make sure you have run nvd_dataset.py and '
    'set the --data_dir argument to the correct path.' % data_file)

	def parse_csv(value):
		tf.logging.info('Parsing {}'.format(data_file))
		columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
		features = dict(zip(_CSV_COLUMNS, columns))
		labels = features.pop('problems')
		return features, labels

	dataset = tf.data.TextLineDataset(data_file)

	 if shuffle:
		dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

	dataset = dataset.map(parse_csv, num_parallel_calls=5)

	dataset = dataset.repeat(num_epochs)
	dataset = dataset.batch(batch_size)
	return dataset

