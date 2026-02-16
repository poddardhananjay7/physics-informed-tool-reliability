import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(42)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    import numpy as np
    import matplotlib.pyplot as plt

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n--- {name} ---")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    # Save scatter plot only for best model
    if "Proxy-Damage" in name:

        plt.figure()
        plt.scatter(y_test, preds, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 linestyle="--")
        plt.xlabel("True RUL")
        plt.ylabel("Predicted RUL")
        plt.title("Predicted vs True RUL (Physics-Informed Proxy)")
        plt.savefig("docs/rul_prediction_scatter.png")
        plt.close()



def train_rul_models():

    df = pd.read_csv("data/rul_dataset.csv")
    assert "proxy_damage" in df.columns, "proxy_damage missing in dataset"

    groups = df["run_id"]
    y = df["RUL"]

    splitter = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(df, y, groups))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    y_train = train_df["RUL"]
    y_test = test_df["RUL"]

    # -------------------------
    # ML-only features
    # -------------------------
    ml_features = ["rms", "peak", "damping", "force_amp"]
    proxy_features = ["rms", "peak", "damping", "force_amp", "proxy_damage"]

    X_train_ml = train_df[ml_features]
    X_test_ml = test_df[ml_features]

    # -------------------------
    # Physics-informed features
    # -------------------------
    hybrid_features = ["rms", "peak", "damping", "force_amp", "damage", "wear"]

    X_train_hybrid = train_df[hybrid_features]
    X_test_hybrid = test_df[hybrid_features]

    # -------------------------
    # Linear baseline
    # -------------------------
    evaluate_model(
        "Linear ML-only",
        LinearRegression(),
        X_train_ml,
        X_test_ml,
        y_train,
        y_test
    )

    # -------------------------
    # Random Forest ML-only
    # -------------------------
    evaluate_model(
        "Random Forest ML-only",
        RandomForestRegressor(n_estimators=100, random_state=42),
        X_train_ml,
        X_test_ml,
        y_train,
        y_test
    )

    # -------------------------
    # Random Forest Proxy
    # -------------------------
    evaluate_model(
        "Random Forest Proxy-Damage (Observable Physics)",
        RandomForestRegressor(n_estimators=100, random_state=42),
        train_df[proxy_features],
        test_df[proxy_features],
        y_train,
        y_test
    )

    # -------------------------
    # Random Forest Hybrid
    # -------------------------
    evaluate_model(
        "Random Forest Hybrid (Physics-Informed)",
        RandomForestRegressor(n_estimators=100, random_state=42),
        X_train_hybrid,
        X_test_hybrid,
        y_train,
        y_test
    )
    

    rf_hybrid = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_hybrid.fit(X_train_hybrid, y_train)

    importances = rf_hybrid.feature_importances_
    feature_names = hybrid_features
    


    plt.figure()
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Hybrid Model Feature Importance")
    plt.savefig("docs/feature_importance.png")
    plt.show()


if __name__ == "__main__":
    train_rul_models()
