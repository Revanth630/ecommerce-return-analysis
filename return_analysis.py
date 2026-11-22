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
    plot(df.groupby("category").returned.mean().sort_values(), odds.sort_values())


def plot(rate, odds):
    """Two panels: return rate by category, and odds ratios from the model."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#fcfcfb",
                               gridspec_kw={"width_ratios": [1, 1.4]})
    a.barh(rate.index, rate.values * 100, color="#2a78d6", height=0.55)
    for y, v in enumerate(rate.values):
        a.text(v * 100 + 0.8, y, f"{v:.0%}", va="center")
    a.set_xlim(0, 50)
    a.set_title("Return rate by category", loc="left", fontweight="bold")
    names = (odds.index.str.replace("category_", "category: ").str.replace("_", " ")
             .str.replace("is cod", "cash on delivery"))
    b.barh(names, odds.values - 1, left=1, height=0.55,
           color=["#eb6834" if v > 1 else "#2a78d6" for v in odds.values])
    b.axvline(1, color="#0b0b0b", lw=0.8)
    for y, v in enumerate(odds.values):
        b.text(v + (0.02 if v > 1 else -0.02), y, f"{v:.2f}", va="center",
               ha="left" if v > 1 else "right")
    b.set_xlim(0, 1.9)
    b.set_xlabel("odds ratio, 1 = no effect (orange raises returns, blue lowers them)",
                 color="#52514e")
    b.set_title("What drives a return", loc="left", fontweight="bold")
    for ax in (a, b):
        ax.set_facecolor("#fcfcfb")
        ax.tick_params(length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("return_drivers.svg", facecolor="#fcfcfb")
    fig.savefig("return_drivers.png", dpi=110, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
