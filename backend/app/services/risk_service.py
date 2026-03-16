import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def classify_risk_logic(revenues):

    labels = []

    for value in revenues:

        if value < 1000:
            labels.append("Low")
        elif value < 5000:
            labels.append("Medium")
        else:
            labels.append("High")

    encoder = LabelEncoder()

    y = encoder.fit_transform(labels)

    model = XGBClassifier()

    model.fit(revenues.reshape(-1,1),y)

    predictions = model.predict(revenues.reshape(-1,1))

    return predictions