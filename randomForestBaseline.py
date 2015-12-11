'''
This script was used for the Homesite Quote Kaggle Competition. 
Sets up preprocesing options, cross-validated tuning of the RandomForestClassifier, and submission. 
'''
# coding: utf-8
import time 
start = time.time()
import os
import sys
import csv
import random
import logging

import pandas as pd
from pandas import Series,DataFrame

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn import preprocessing
from sklearn import metrics
from sklearn import ensemble

from helpers import *

#Constants 
total_training_rows = 260753

#preprocessing options 
drop_na_cols = True #cannot set this to False. TODO: use Imputer class to handle missing values

#RandomForestClassifier params
n_estimators = 300 #minimal improvement past 300 
max_depth = 100
max_features = 25 
min_samples_split = 20
min_samples_leaf = 1
criterion = "gini" 
class_weight = None 

#Cross validation params 
train_size = 120000
test_size = 20000
cv_iterations = 1
optimize_param = 'min_samples_leaf'
grid = [1]
seed = 60

#Submission options
generate_submission = False

#caching/speed/memory options 
read_cache = True
write_cache = True
read_csv_chunksize = 45000 #low memory enviroment 
max_read_rows = 150000 #low memory environment 

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level", default = "ERROR")
args = parser.parse_args()
initializeLogger(args.logLevel)

rel_dir = os.getcwd()
train_fn = os.path.join(rel_dir, "train.csv")
test_fn = os.path.join(rel_dir, "test.csv")
if not (os.path.isfile(train_fn)):
	logging.error("File does not exist: %s", train_fn)
if not (os.path.isfile(test_fn)):
	logging.error("File does not exist: %s", test_fn)
read_rows = train_size + test_size
if (read_rows > max_read_rows):
	logging.error("Trying to read to many rows. Reduce train + test size to less than max_read_rows: %s", max_read_rows)
	sys.exit()

[columns, label_dict] = processNanAndLabels(train_fn, test_fn, read_cache, write_cache, drop_na_cols, read_csv_chunksize)

logging.info("starting cv")
for val in grid:  
	cv_scores_is = []
	cv_scores_os = []
	for i in range(0, cv_iterations):
		classifier = ensemble.RandomForestClassifier(n_estimators = n_estimators, criterion = criterion,  max_features = max_features, random_state = seed+i, max_depth = max_depth, class_weight= class_weight, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
		classifier.set_params(**{optimize_param : val})
		random.seed(seed)
		skip_rows = random.sample(range(1, total_training_rows + 1), total_training_rows - read_rows)
		df = pd.read_csv(train_fn, skiprows = skip_rows, usecols = columns, thousands = ",")
		rows_shuffled = range(read_rows)
		test_rows = rows_shuffled[:test_size]
		train_rows = rows_shuffled[test_size:]
		df = cleanDf(df, drop_na_cols)
		df = handleCategories(df, label_dict)
		[X_train, Y_train, X_test, Y_test] = splitTrainTest(df, train_rows, test_rows)
		classifier.fit(X_train, Y_train)
		Y_train_pred = classifier.predict_proba(X_train)[:,1]
		Y_test_pred = classifier.predict_proba(X_test)[:,1]
		Y_test_correct = np.array(Y_test.tolist())
		Y_train_correct = np.array(Y_train.tolist())
		cv_scores_is.append(str(round(score(Y_train_pred, Y_train_correct),3)))
		cv_scores_os.append(str(round(score(Y_test_pred, Y_test_correct),3)))
		logging.info("finished fit: %s", i)
	print(optimize_param + ": " + str(val))
	print("in-sample: " + ' '.join(cv_scores_is))
	print("out-sample: " + ' '.join(cv_scores_os) + "\n")

if generate_submission: 
	test_cols = columns
	test_cols.discard('QuoteConversion_Flag')
	submit_chunk =  pd.read_csv(test_fn, chunksize = read_csv_chunksize , usecols = test_cols,  thousands = ",")
	submission = pd.DataFrame()
	submission["QuoteNumber"] = []
	submission["QuoteConversion_Flag"] = []
	submission.to_csv('homesite.csv', index=False)
	for submit_df in submit_chunk:
		submit_df = cleanDf(submit_df, drop_na_cols)
		submit_df = handleCategories(submit_df, label_dict)
		submission = pd.DataFrame()
		submission["QuoteNumber"] = submit_df["QuoteNumber"]
		Y_submit_pred = classifier.predict_proba(submit_df)[:,1]
		submission["QuoteConversion_Flag"] = Y_submit_pred
		submission.to_csv('homesite.csv', mode = 'a', header = False, index=False)

print("Runtime: " + str(time.time() - start) + " s")
