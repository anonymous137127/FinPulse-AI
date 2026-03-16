import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def forecast_revenue_logic(revenues):

    chunk_size = max(1, len(revenues)//6)

    monthly_totals = []

    for i in range(6):

        start = i * chunk_size
        end = start + chunk_size

        monthly_totals.append(sum(revenues[start:end]))

    monthly_totals = np.array(monthly_totals)

    X = np.arange(len(monthly_totals)).reshape(-1,1)

    y = monthly_totals

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

    model = LinearRegression()

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    accuracy = r2_score(y_test,y_pred)

    prediction = model.predict([[len(monthly_totals)]])[0]

    return prediction, accuracy