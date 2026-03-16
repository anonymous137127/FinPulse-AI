import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def classify_risk_xgb_logic(revenues):

    revenues = np.array(revenues).reshape(-1,1)

    # create labels
    labels = []

    for r in revenues:
        value = r[0]

        if value < 1000:
            labels.append("Low")
        elif value < 5000:
            labels.append("Medium")
        else:
            labels.append("High")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        revenues, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    predictions = model.predict(revenues)
    probabilities = model.predict_proba(revenues)

    return model, encoder, predictions, probabilities, accuracy