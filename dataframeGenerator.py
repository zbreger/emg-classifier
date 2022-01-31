import sys
sys.path.append( '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder, PowerTransformer, StandardScaler
  )
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import pandas as pd

f = open('data/EMG_data_for_gestures-master/31/1_raw_data_11-15_11.04.16.txt', 'r')
f = f.readlines()

for count, val in enumerate(f):
    row = val.split("\t")
    row[-1] = row[-1][:-1]
    f[count] = row
df = pd.DataFrame(f[1:], columns=f[0])
df.to_csv("motionData.csv")