import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.ensemble import IsolationForest

def get_anomaly_data(df):
    # Detect anomalies in Delivery Delay and Lost Sales Cost
    model = IsolationForest(contamination=0.02)
    features = df[['Delivery_Delay', 'Lost_Sales_Cost', 'Fill_Rate']]
    df['is_anomaly'] = model.fit_predict(features)
    return df[df['is_anomaly'] == -1]

def get_association_rules(df):
    # Prepare data for Association Rule Mining (e.g., Disruption types and Transport modes)
    basket = pd.get_dummies(df[['Disruption_Type', 'Transportation_Mode']])
    frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    return rules
