import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pickle
import os


current_path = os.path.dirname(os.path.abspath(__file__))
print(current_path)


file = "Superstore.csv"
df = pd.read_csv(file)
df.drop(['Row ID', 'Order ID', "Ship Mode", "Ship Date", 'Customer ID', 'Customer Name', "Segment", 'Country', 'City',
         "State", 'Postal Code', 'Region', 'Product ID', "Category", "Product Name"], axis=1, inplace=True)
df['Order Month'] = pd.DatetimeIndex(df['Order Date']).month
df.drop("Order Date", axis=1, inplace=True)
df['Order_mnth_qtr'] = df['Order Month'].map(lambda x: (x - 1) // 3 + 1)
df.drop("Order Month", axis=1, inplace=True)
df_with_dummies = pd.get_dummies(df, columns=["Sub-Category", "Order_mnth_qtr"], prefix=["Category", "Order Quarter"],
                                 drop_first=False)
df_with_dummies["Discounts"] = df["Sales"] * df["Discount"]
df_with_dummies.drop("Discount", axis=1, inplace=True)

numeric = df_with_dummies[['Sales', 'Quantity', 'Profit', 'Discounts']]
categoric = df_with_dummies[['Category_Accessories', 'Category_Appliances', 'Category_Art',
                             'Category_Binders', 'Category_Bookcases', 'Category_Chairs',
                             'Category_Copiers', 'Category_Envelopes', 'Category_Fasteners',
                             'Category_Furnishings', 'Category_Labels', 'Category_Machines',
                             'Category_Paper', 'Category_Phones', 'Category_Storage',
                             'Category_Supplies', 'Category_Tables', 'Order Quarter_1', 'Order Quarter_2',
                             'Order Quarter_3', 'Order Quarter_4']]
categor_arr = categoric.to_numpy()
numer_arr = numeric.to_numpy()
num_for_scale = numer_arr[:, [0, 1, 3]]
scaler = StandardScaler()
print(num_for_scale.shape)
num_scaled = scaler.fit_transform(num_for_scale)
print(num_scaled.shape)
num_scaled_arr = np.concatenate((numer_arr[:, [2]], num_scaled), axis=1)
print(num_scaled_arr.shape)
df_arr = np.concatenate((num_scaled_arr, categor_arr), axis=1)

idx_OUT_column = [0]
idx_IN_columns = [i for i in range(np.shape(df_arr)[1]) if i not in idx_OUT_column]
X = df_arr[:, idx_IN_columns]
Y = df_arr[:, [0]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)
# Save model and scaler to disk
pickle.dump(lm, open('finalized_model.pkl', 'wb'))
pickle.dump(scaler, open("scaling.pkl", "wb"))

