"""E-commerce return reduction: which order attributes drive returns? Logistic regression + odds ratios."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(7)


def make_orders(n=8000):
    df = pd.DataFrame({
        "category": RNG.choice(["Apparel", "Footwear", "Electronics", "Home"], n, p=[0.4, 0.2, 0.2, 0.2]),
        "price": RNG.lognormal(7, 0.6, n).round(0),
        "discount_pct": RNG.choice([0, 10, 20, 40, 60], n),
        "delivery_days": RNG.integers(1, 10, n),
        "size_chart_viewed": RNG.integers(0, 2, n),
        "is_cod": RNG.integers(0, 2, n),
    })
    logit = (-2.2 + df.category.map({"Apparel": 0.9, "Footwear": 0.7, "Electronics": -0.3, "Home": -0.5})
             + 0.015 * df.discount_pct + 0.12 * df.delivery_days
             - 0.6 * df.size_chart_viewed + 0.5 * df.is_cod)
    df["returned"] = (RNG.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def main():
    df = make_orders()
    print("Return rate by category:\n", df.groupby("category").returned.mean().round(3).to_string(), "\n")

    X = pd.get_dummies(df.drop(columns="returned"), columns=["category"], drop_first=True, dtype=float)
    X["log_price"] = np.log(X.pop("price"))
    X_tr, X_te, y_tr, y_te = train_test_split(X, df.returned, test_size=0.25, random_state=7)
    model = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    print(f"ROC-AUC: {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.3f}\n")

    odds = pd.Series(np.exp(model.coef_[0]), index=X.columns).sort_values(ascending=False)
    print("Odds ratios (>1 raises return likelihood):\n", odds.round(2).to_string())


if __name__ == "__main__":
    main()
