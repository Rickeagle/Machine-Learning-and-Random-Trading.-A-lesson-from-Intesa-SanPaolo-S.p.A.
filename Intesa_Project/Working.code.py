# Riccardo Negrisoli
# AI and ML 



# To correctly run the code with no errors PLEASE 
# use the command : pip install arch 





from IPython import get_ipython
get_ipython().magic('reset -sf')
get_ipython().magic('clear')
#%%
#pip install yfinance pandas openpyxl
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import yfinance as yf
import pandas as pd
from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, roc_curve, classification_report
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay


from yahooquery import Ticker
import os

#%%  Section 0
## Relevant Information and preliminary analysis of Intesa
# FETCH FUNDAMENTAL DATA FOR INTESA SAN PAOLO

# Create a Ticker object for Intesa San Paolo (ticker "ISP.MI")
ISP = Ticker("ISP.MI")

# Retrieve and print the ESG scores
print("ESG Scores:")
print(ISP.esg_scores)

# Retrieve and print key statistics
print("\nKey Statistics:")
print(ISP.key_stats)

# Retrieve and print the summary profile
print("\nSummary Profile:")
print(ISP.summary_profile)

# Retrieve and print institutional ownership details
print("\nInstitutional Ownership:")
print(ISP.institution_ownership)

# Retrieve and print fund ownership details
print("\nFund Ownership:")
print(ISP.fund_ownership)

# Retrieve and print the quarterly balance sheet by specifying frequency="q"
print("\nQuarterly Balance Sheet:")
print(ISP.balance_sheet(frequency="q"))

# Retrieve and print the cash flow statement
print("\nCash Flow:")
print(ISP.cash_flow())

# Retrieve and print the income statement
print("\nIncome Statement:")
print(ISP.income_statement())

#%% IMPORTING DATA

# Setting my working directory 
os.chdir(r"C:\Users\Utente\Desktop\WESTMINSTER\AI and ML\Intesa_Project")
folder_path = r"C:\Users\Utente\Desktop\WESTMINSTER\AI and ML\Intesa_Project\Figures"

# Download Intesa Sanpaolo data
ticker = "ISP.MI" ### Intesa San Paolo S.p.A. is traded on the Milan Stock Exchange with the ticker "ISP.MI"
start_date = "2004-01-01"
end_date = "2024-12-31"

ISPdata = yf.download(ticker, start=start_date, end=end_date)
ISPdata.dropna(inplace=True)

# If columns is a MultiIndex (e.g., top level is "ISP.MI"), drop that level
if isinstance(ISPdata.columns, pd.MultiIndex):
    ISPdata.columns = ISPdata.columns.droplevel(1)

# Move the date index into a normal column
ISPdata.reset_index(inplace=True)

# Keep only the columns you want
ISPdata = ISPdata[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Export to Excel without the index
FileName = "Intesa_stock_data.xlsx"
ISPdata.to_excel(FileName, index=False)

print(f"Stock data saved to {FileName}")


#%% Section 2 : Feature Engeneering

# Create new features based on the stock prices
ISPdata['H-L'] = ISPdata['High'] - ISPdata['Low']
ISPdata['O-C'] = ISPdata['Close'] - ISPdata['Open']

# Calculate moving averages for the Close price using a 1-day lag to avoid lookahead bias
ISPdata['3day MA'] = ISPdata['Close'].shift(1).rolling(window=3).mean()
ISPdata['10day MA'] = ISPdata['Close'].shift(1).rolling(window=10).mean()
ISPdata['30day MA'] = ISPdata['Close'].shift(1).rolling(window=30).mean()

# Calculate the rolling standard deviation of the Close price over a 5-day window
ISPdata['Std_dev'] = ISPdata['Close'].rolling(window=5).std()

# Create a binary target variable: 1 if the next day's Close is higher than today's Close, else 0
ISPdata['Price_Rise'] = np.where(ISPdata['Close'].shift(-1) > ISPdata['Close'], 1, 0)

# Drop any rows with missing values from shifting/rolling operations
ISPdata = ISPdata.dropna()

# Display the first few rows of the enhanced dataset
print(ISPdata.head())
ISPdata.index.name = "Date"
ISPdata.to_excel(FileName, index=False)
print(f"Stock data saved to {FileName}")


#%% Extra section 2.1: ANALYSIS OF THE VARIANCE

# Calculate returns (ensure the series is stationary)
returns = np.log(ISPdata['Close'] / ISPdata['Close'].shift(1)).dropna()
#Checking for stationarity
plt.figure(figsize=(10, 5))
plt.plot(returns.index, returns, label='Returns')
plt.axhline(0, color='red', linestyle='--')
plt.title('Daily Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.savefig(os.path.join(folder_path, "fig0.ClosingVol.Relationship.png"))
plt.show()

ISPdata['Returns']= returns

# Fit a GARCH(1,1) model with t-distribution
model = arch_model(returns, vol='Garch', p=1, q=1, dist='t', rescale=True)
fit = model.fit(disp='off')

# Compute conditional variance (volatility squared)
ISPdata = ISPdata.iloc[1:]  # Align with returns dropna
ISPdata['GARCH_t_variance'] = fit.conditional_volatility ** 2

print(fit.summary())
ISPdata.to_excel(FileName, index=False)
print(f"Stock data saved to {FileName}")
fig, ax1 = plt.subplots(figsize=(10, 6))

# Create a figure and an axis (ax1) for the first plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Daily Returns on the left y-axis
color1 = 'tab:blue'
ax1.set_xlabel('Date', color=color1)
ax1.set_ylabel('Daily Returns', color=color1)
ax1.plot(ISPdata['Date'], ISPdata['Returns'], color=color1, label='Daily Returns')
ax1.tick_params(axis='y', labelcolor=color1)

# Create a second axis (ax2) sharing the same x-axis
ax2 = ax1.twinx()

# Plot GARCH Variance on the right y-axis
color2 = 'tab:red'
ax2.set_ylabel('GARCH Variance', color=color2)
ax2.plot(ISPdata['Date'], ISPdata['GARCH_t_variance'], color=color2, label='GARCH Variance')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Daily Log-Returns and GARCH Conditional Variance')
fig.tight_layout()
plt.legend()
plt.savefig(os.path.join(folder_path, "fig20.Garch&Returns.png"))
plt.show()




#%%  PLOTS

# Figure 1: Plot for Closing Price and Volume
fig, ax1 = plt.subplots(figsize=(12, 6))
color = 'tab:blue'
ax1.set_xlabel('Date', color=color)
ax1.set_ylabel('Closing Price', color=color)
ax1.plot(ISPdata['Date'], ISPdata['Close'], label='Closing Price', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Volume', color=color)
ax2.plot(ISPdata['Date'], ISPdata['Volume'], label='Volume', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Closing Price and Volume Relationship')
fig.tight_layout()
plt.savefig(os.path.join(folder_path, "fig1.ClosingVol.Relationship.png"))
plt.show()

# Figure 2: Plot for 3day, 10day, 30day MAs
plt.figure(figsize=(10, 8))
plt.plot(ISPdata['Date'], ISPdata['3day MA'], label='3day MA')
plt.plot(ISPdata['Date'], ISPdata['10day MA'], label='10day MA')
plt.plot(ISPdata['Date'], ISPdata['30day MA'], label='30day MA')
plt.title('Moving Averages Over Time')
plt.xlabel('Date')
plt.ylabel('Moving Averages')
plt.legend()
plt.savefig(os.path.join(folder_path, "fig2.MA.png"))
plt.show()

# Figure 3: Plots for O-C, H-L, and Std_dev
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(ISPdata['Date'], ISPdata['O-C'])
axes[0].set_title('O-C')
axes[1].plot(ISPdata['Date'], ISPdata['H-L'])
axes[1].set_title('H-L')
axes[2].plot(ISPdata['Date'], ISPdata['Std_dev'])
axes[2].set_title('Std_dev')

plt.tight_layout()
plt.savefig(os.path.join(folder_path, "fig3.Comparisons.png"))
plt.show()

# Figure 4: 3day, 10day, 30day MA histogram & density
fig, axes = plt.subplots(1, 3, figsize=(15, 3))
sns.histplot(data=ISPdata, x='3day MA', kde=True, stat='density', ax=axes[0])
axes[0].set_title('3day MA')

sns.histplot(data=ISPdata, x='10day MA', kde=True, stat='density', ax=axes[1])
axes[1].set_title('10day MA')

sns.histplot(data=ISPdata, x='30day MA', kde=True, stat='density', ax=axes[2])
axes[2].set_title('30day MA')

plt.tight_layout()
plt.savefig(os.path.join(folder_path, "fig4.Hist&Density.png"))
plt.show()

# Figure 5: Price Rise & Fall Count Plot + Scatter
chart = sns.FacetGrid(ISPdata, col='Price_Rise')
chart.map(sns.histplot, 'Close')
plt.savefig(os.path.join(folder_path, "figure5_hist.png"))
plt.show()

chart = sns.FacetGrid(ISPdata, col='Price_Rise')
chart.map(plt.scatter, 'Close', 'Std_dev')
plt.savefig(os.path.join(folder_path, "figure5_scatter.png"))
plt.show()

#Fig 6: Corr Matrix
corr_matrix = ISPdata.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.savefig(os.path.join(folder_path, "fig6.CorrMatrix.png"))
plt.show()



#%%  
ISPdata.info()
ISPdata.describe()
ISP.summary_profile

ISP.institution_ownership

ISP.fund_ownership
ISP.history()

#%%  MACHINE LEARNING

feature_cols = ['Volume', 'H-L', 'O-C','Std_dev', '3day MA']
target_col = 'Price_Rise'

X = ISPdata[feature_cols]

Y = ISPdata[target_col]

Y



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15,random_state=240)
X_train

# Standardize features by removing the mean and scaling to unit variance.
# StandardScaler transfers data to standard normally distributed data: Gaussian with zero mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regr = LinearRegression()
regr.fit(X_train, Y_train)
Y_pred = regr.predict(X_test)
print ("MAE", mean_absolute_error(Y_test, Y_pred))

#%%
#  Logistic Regression

model_logistic = LogisticRegression(random_state=101)
model_logistic.fit(X_train, Y_train)

# Predict on the test set
Y_pred_logistic = model_logistic.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(Y_test, Y_pred_logistic))


# Extra Trees
model_extra_trees = ExtraTreesClassifier(random_state=101)
model_extra_trees.fit(X_train, Y_train)

# Predict on the test set
Y_pred_extra = model_extra_trees.predict(X_test)
print("Extra Trees Classification Report:")
print(classification_report(Y_test, Y_pred_extra))

 ## LOGISTIC REGRESSION PERFORMS BETTER!!!
 
 
#%%  #N-fold Cross Validation

#Cross Validation for Logistic Regression
accuracy_scores_log = cross_val_score(model_logistic, X, Y, cv=5, scoring='accuracy')
print(f"Logistic Regression - Mean Accuracy: {accuracy_scores_log.mean():.2f}")
print(f"Logistic Regression - Standard Deviation: {accuracy_scores_log.std():.2f}")

#Cross Validation for Extra Trees
accuracy_scores_et = cross_val_score(model_extra_trees, X, Y, cv=5, scoring='accuracy')
print(f"Extra Trees - Mean Accuracy: {accuracy_scores_et.mean():.2f}")
print(f"Extra Trees - Standard Deviation: {accuracy_scores_et.std():.2f}")

## WE ARE PERFORMING NO BETTER THAN RANDOM GUESSING



# --------------------------------------
#%% Prediction of Price Rise Using Logistic Regression on X_test Data

# Re-train or re-use model_extra_trees
model_logistic = LogisticRegression(random_state=101)
model_logistic.fit(X_train, Y_train)

Y_pred_logistic_test = model_logistic.predict(X_test)
classification_rep = classification_report(Y_test, Y_pred_logistic_test)
print("Final Logistic Regression Classification Report on X_test:")
print(classification_rep)


#%% Confusion MAtrix for logistic
matrix = ConfusionMatrixDisplay.from_estimator(model_logistic, X_test, Y_test)  
plt.title('Confusion Matrix')
plt.savefig(os.path.join(folder_path, "fig6.ConfMatrix.png"))
plt.show()


#%% Confusion MAtrix for Extra Trees
matrix = ConfusionMatrixDisplay.from_estimator(model_extra_trees, X_test, Y_test)  
plt.title('Confusion Matrix')
plt.savefig(os.path.join(folder_path, "fig7.ConfMatrix2.png"))
plt.show()

#%%   ROC Curve for Logistic Regression
log_disp = RocCurveDisplay.from_estimator(model_logistic, X_test, Y_test)
plt.title("ROC Curve - Logistic")
plt.savefig(os.path.join(folder_path, "roc_curve.Logistic.png"))
plt.show()



#%%  BAYESIAN NAIVE


Gauss_Model = GaussianNB()
Gauss_Model.fit(X_train, Y_train)
Y_pred = Gauss_Model.predict(X_test)
target_names=["Down", "Up"]

print (classification_report(Y_test, Y_pred))

# Evaluate the model by means of a Confusion Matrix

matrix = ConfusionMatrixDisplay.from_estimator(Gauss_Model, X_test, Y_test, display_labels=target_names)  
plt.title('Confusion Matrix')
plt.savefig(os.path.join(folder_path, "fig8.ConfMatrix3.png"))
plt.show()

#%%  FEATURE IMPORTANCE  LOGISTIC
classifier = LogisticRegression(random_state=101)
classifier.fit(X_train, Y_train)

# Now retrieve the coefficients from the fitted classifier
importance = classifier.coef_[0]



feature_names=X.columns


indices = np.argsort(importance)
range1 = range(len(importance[indices]))
plt.figure()
plt.title("Logistic Regression Feature Importance")
plt.barh(range1,importance[indices])
plt.yticks(range1, feature_names[indices])
plt.ylim([-1, len(range1)])
plt.savefig(os.path.join(folder_path,"fig9.Logistic.FeatureImportance.png"))
plt.show()

#%% MARKET AND RETURN STRATEGIES WITH LOGISTIC

#  Data Preprocessing
ISPdata['Y_pred_logistic_test'] = np.nan
ISPdata.iloc[len(ISPdata) - len(Y_pred_logistic_test):, -1] = Y_pred_logistic_test  # Fill the last rows with predictions
trade_ISPdata = ISPdata.dropna()  # Drop rows without a prediction

# Computation of Market Returns
trade_ISPdata['Tomorrows Returns'] = 0.
trade_ISPdata['Tomorrows Returns'] = np.log(trade_ISPdata['Close'] / trade_ISPdata['Close'].shift(1))
trade_ISPdata['Tomorrows Returns'] = trade_ISPdata['Tomorrows Returns'].shift(-1)

# Strategy Returns based on Y_pred
trade_ISPdata['Strategy Returns'] = 0.
trade_ISPdata['Strategy Returns'] = np.where(
    trade_ISPdata['Y_pred_logistic_test'] == True,
    trade_ISPdata['Tomorrows Returns'],
    -trade_ISPdata['Tomorrows Returns']
)


# ######Cumulative Market and Strategy Returns

#Computation of cumulative market and strategy returns
trade_ISPdata['Cumulative Market Returns'] = np.cumsum(trade_ISPdata['Tomorrows Returns'])
trade_ISPdata['Cumulative Strategy Returns'] = np.cumsum(trade_ISPdata['Strategy Returns'])

#Plot of cumulative market and strategy returns based on Y_pred
plt.figure(figsize=(10, 3))
plt.plot(trade_ISPdata['Cumulative Market Returns'], color='red', label='Market Returns')
plt.plot(trade_ISPdata['Cumulative Strategy Returns'], color='blue', label='Strategy Returns')
plt.title("Cumulative Market vs. Strategy Returns")
plt.legend()
plt.savefig(os.path.join(folder_path,"fig10.MktStrategy.png"))
plt.show()



#%% FEATURES IMPORTANCE AND MARKET STRATEGY RETURNS USING NAIVE BAYES

# FEATURE "IMPORTANCE" with GaussianNB
# ===============================
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, Y_train)

# Naive Bayes does not provide coefficients; as a proxy we use the class‑conditional means.
# (This only makes sense if your task is binary. Here we assume class 1 is the "positive" class.)
importance = nb_classifier.theta_[1]  
feature_names = X.columns

# Sort the "importance" values for plotting
indices = np.argsort(importance)
range1 = range(len(importance))

plt.figure(figsize=(10, 6))
plt.title("GaussianNB - Feature Means for Class 1")
plt.barh(range1, importance[indices])
plt.yticks(range1, feature_names[indices])
plt.xlabel("Mean Value")
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "fig9.GaussianNB_FeatureMeans.png"))
plt.show()

# ===============================
# MARKET AND RETURN STRATEGIES WITH GAUSSIAN NAIVE BAYES
# ===============================

# Get predictions on the test set from the NB model
Y_pred_nb = nb_classifier.predict(X_test)
print("GaussianNB Classification Report on X_test:")
print(classification_report(Y_test, Y_pred_nb))

# Data Preprocessing: Add NB predictions to ISPdata.
# Here we assume that ISPdata originally has all the data in order and that the predictions
# correspond to the last len(Y_pred_nb) rows.
ISPdata['Y_pred_nb'] = np.nan
ISPdata.iloc[len(ISPdata) - len(Y_pred_nb):, -1] = Y_pred_nb  # Fill last rows with predictions
trade_ISPdata = ISPdata.dropna()  # Drop rows without a prediction

# Compute Market Returns as log returns
trade_ISPdata['Tomorrows Returns'] = np.log(trade_ISPdata['Close'] / trade_ISPdata['Close'].shift(1))
trade_ISPdata['Tomorrows Returns'] = trade_ISPdata['Tomorrows Returns'].shift(-1)

# Compute Strategy Returns based on the NB predictions:
# If Y_pred_nb is True (or 1), take the market return; else, take the negative.
trade_ISPdata['Strategy Returns'] = np.where(
    trade_ISPdata['Y_pred_nb'] == True,  # adjust if your positive class is represented differently
    trade_ISPdata['Tomorrows Returns'],
    -trade_ISPdata['Tomorrows Returns']
)

# Compute cumulative returns
trade_ISPdata['Cumulative Market Returns'] = np.cumsum(trade_ISPdata['Tomorrows Returns'])
trade_ISPdata['Cumulative Strategy Returns'] = np.cumsum(trade_ISPdata['Strategy Returns'])

# Plot the cumulative returns
plt.figure(figsize=(10, 3))
plt.plot(trade_ISPdata['Cumulative Market Returns'], color='red', label='Market Returns')
plt.plot(trade_ISPdata['Cumulative Strategy Returns'], color='blue', label='Strategy Returns')
plt.title("Cumulative Market vs. Strategy Returns (GaussianNB)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "fig10.MktStrategy_GaussianNB.png"))
plt.show()

#%%   ANALYSIS OF VARIANCE WITH ML

# Use the median of GARCH_t_variance as the threshold
vol_threshold = ISPdata['GARCH_t_variance'].median()
ISPdata['HighVolatility'] = (ISPdata['GARCH_t_variance'] > vol_threshold).astype(int)

ISPdata['Lag1_Return'] = ISPdata['Return'].shift(1)
ISPdata['Lag2_Return'] = ISPdata['Return'].shift(2)
ISPdata['HighVolatility_Tomorrow'] = ISPdata['HighVolatility'].shift(-1)

# Drop rows with missing values caused by pct_change and shifting
data_ml = ISPdata.dropna(subset=['Return', 'Lag1_Return', 'Lag2_Return', 'Volume', 'HighVolatility_Tomorrow'])

# Define Feature Matrix X and Target Vector Y
featuresNew = ['Return', 'Lag1_Return', 'Lag2_Return', 'Volume','H-L','O-C', '3day MA']
targetNew = 'HighVolatility_Tomorrow'
X = data_ml[featuresNew]
Y = data_ml[targetNew]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

# Train a Logistic Regression Model for Volatility Regime Prediction
modelGarch = LogisticRegression(random_state=101)
modelGarch.fit(X_train, Y_train)

# Predict on test set
Y_pred = modelGarch.predict(X_test)
correct_predictions = (Y_pred == Y_test)
print("Classification Report:")
print(classification_report(Y_test, Y_pred))
#  Plot and Save the Confusion Matrix
plt.figure(figsize=(6, 6))
cm_disp = ConfusionMatrixDisplay.from_estimator(modelGarch, X_test, Y_test)
plt.title("Confusion Matrix - Volatility Regime Prediction")
plt.savefig(os.path.join(folder_path, "Volatility_Regime_ConfusionMatrix.png"))
plt.show()

#  Plot the ROC Curve

plt.figure(figsize=(6, 6))
roc_disp = RocCurveDisplay.from_estimator(modelGarch, X_test, Y_test)
plt.title("ROC Curve - Volatility Regime Prediction")
plt.savefig(os.path.join(folder_path, "Volatility_Regime_ROC_Curve.png"))
plt.show()


modelGarch.fit(X_train, Y_train)
importance = modelGarch.coef_[0]
feature_names=X.columns


indices = np.argsort(importance)
range1 = range(len(importance[indices]))
plt.figure()
plt.title("Logistic Regression Feature Importance")
plt.barh(range1,importance[indices])
plt.yticks(range1, feature_names[indices])
plt.ylim([-1, len(range1)])
plt.savefig(os.path.join(folder_path,"fig30.Variance.FeatureImportance.png"))
plt.show()

# Count the number of correct and incorrect predictions.
accuracy_counts = correct_predictions.value_counts()
print("Prediction Correctness Counts:\n", accuracy_counts)


plt.figure(figsize=(6,4))
accuracy_counts.plot(kind='bar', color=['green', 'red'])
plt.title("Prediction Accuracy (Correct vs. Incorrect)")
plt.xlabel("Prediction Correctness (True=Correct, False=Incorrect)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "Volatility_Tomorrow_Prediction_Accuracy.png"))
plt.show()






























    