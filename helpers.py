'''
Helper functions for the HomesiteQuote Kaggle Competition. 
'''
# coding: utf-8
# Imports
import os
import sys
import csv
import time 
import random
import pickle 
import datetime
import logging 
import argparse
import logaugment

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# machine learning
from sklearn import preprocessing
from sklearn import metrics

def processNanAndLabels(train_fn, test_fn, read_cache, write_cache, drop_na_cols, chunksize):
	logging.info("processing missing data and labels")
	clean_columns_fn = 'clean_columns.p'
	all_columns_fn = 'all_columns.p'
	label_dict_fn = 'label_dict.p'
	columns_fn = clean_columns_fn if drop_na_cols else all_columns_fn
	if read_cache:
		try:
			columns = pickle.load(open(columns_fn, 'rb'))
			label_dict = pickle.load(open(label_dict_fn, 'rb'))
			logging.info("loaded cache, columns size: %s", len(columns))
			return [columns, label_dict] 
		except:
			logging.error("failed to load cache")
	df_small = pd.read_csv(train_fn, nrows = 2, thousands = ",")
	columns = set(df_small.columns)
	label_dict = {}

	for f in df_small.columns:
		if df_small[f].dtype == 'object':
			label_dict[f] = set()
	del label_dict['Original_Quote_Date']

	for chunk in pd.read_csv(train_fn, chunksize = chunksize, thousands = ","):
		for f in chunk.columns:
			if drop_na_cols:
				if -1 in chunk[f].values or chunk[f].isnull().values.any():
					columns.discard(f)
			if label_dict.has_key(f):
				label_dict[f].update(np.unique(list(chunk[f].values)))
			
		logging.info("columns left: %s", len(columns))

	for chunk in pd.read_csv(test_fn, chunksize = chunksize, thousands = ","):
		for f in chunk.columns:
			if drop_na_cols:
				if -1 in chunk[f].values or chunk[f].isnull().values.any():
					columns.discard(f)
			if label_dict.has_key(f):
				label_dict[f].update(np.unique(list(chunk[f].values)))
		logging.info("columns left: %s", len(columns))

	for key in label_dict.keys():
		lbl = preprocessing.LabelEncoder()
		lbl.fit(np.array(list(label_dict[key])))
		label_dict[key] = lbl

	if write_cache:
		pickle.dump(columns, open(columns_fn, 'wb'))
		pickle.dump(label_dict, open(label_dict_fn, 'wb'))
	return [columns, label_dict] 

def cleanDf(df, drop_na_cols):
	if not drop_na_cols:
		df.replace(-1,float('nan'))
	if ('Original_Quote_Date' in df.columns):
		df['Year']  = df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
		df['Month'] = df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
		df['Week']  = df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))
		df.drop(['Original_Quote_Date'], axis=1, inplace=True)
	return df

def handleCategories(df, label_dict):
	for f in df.columns:
		if label_dict.has_key(f):
			df[f] = label_dict[f].transform(list(df[f].values))
	return df

def splitTrainTest(df, train_rows, test_rows):
	train_df = df.drop(test_rows)
	test_df = df.drop(train_rows)
	X_train = train_df.drop(['QuoteConversion_Flag'], axis=1)
	Y_train = train_df['QuoteConversion_Flag']
	X_test = test_df.drop(['QuoteConversion_Flag'], axis=1)
	Y_test = test_df['QuoteConversion_Flag']
	return [X_train, Y_train, X_test, Y_test]

def score(y_pred, y_correct):
	return metrics.roc_auc_score(y_correct, y_pred)

def process_record(record):
	now = datetime.datetime.utcnow()
	try:
		delta = now - process_record.now
	except AttributeError:
		delta = 0
	process_record.now = now
	try:
		formatted = '{}s'.format(delta.total_seconds())
	except AttributeError:
		formatted = '0s'
	return {'time_since_last': formatted}

def initializeLogger(logLevel):
	if(logLevel):
		logger = logging.getLogger()
		handler = logging.StreamHandler()
		formatter = logging.Formatter("%(levelname)s %(time_since_last)s: %(message)s")
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		logger.setLevel(getattr(logging, logLevel))
		logaugment.add(logger, process_record)