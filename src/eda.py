from pandas import DataFrame
import pandas as pd
from sklearn.utils import shuffle


file= 'data/credit-data.csv'
fraud_data: DataFrame = pd.read_csv(file)
fraud_data: DataFrame = shuffle(fraud_data)
