import kagglehub

# Download latest version
path = kagglehub.dataset_download("ziya07/plant-health-data")

print("Path to dataset files:", path)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

df = pd.read_csv(path+"plant_health_data.csv")