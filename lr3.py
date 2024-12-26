import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor


data = pd.read_csv("KAG_energydata_complete.csv")
X = data.drop('Appliances', axis=1)
y = data["Appliances"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_min, y_max = y_train.min(), y_train.max()
learning_rates = [0.05, 0.1]
depths = [6, 8]
iterations = [500, 1000]
results = []

for lr in learning_rates:
    for depth in depths:
        for iter_count in iterations:
            print(f"Моделювання з параметрами: Learning rate: {lr}, Depth: {depth}, Iterations: {iter_count}")
            model = CatBoostRegressor(
                learning_rate=lr,
                depth=depth,
                iterations=iter_count,
                verbose=0,
                random_seed=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            from math import sqrt
            rmse = sqrt(mean_squared_error(y_test, y_pred))
            nrmse = rmse / (y_max - y_min)
            results.append([lr, depth, iter_count, rmse, nrmse])


results_df = pd.DataFrame(results, columns=["Learning Rate", "Depth", "Iterations", "RMSE", "NRMSE"])
results_df.to_csv("catboost_regression_results.csv", index=False)