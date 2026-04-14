import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
MAX_ITER = 10000
ZERO_TOL = 1e-5


def load_data(path="data/telecom_churn.csv"):
    df = pd.read_csv(path)

    X = df.drop(columns=["customer_id", "churned"])
    y = df["churned"]

    return X, y


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore"
                ),
                categorical_cols,
            ),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    feature_names = list(numeric_cols)

    if categorical_cols:
        cat_features = (
            preprocessor.named_transformers_["cat"]
            .get_feature_names_out(categorical_cols)
            .tolist()
        )
        feature_names.extend(cat_features)

    return feature_names


def fit_regularization_paths(X_processed, y, feature_names, C_values):
    settings = {
        "l1": "saga",
        "l2": "lbfgs",
    }

    all_results = []

    for penalty, solver in settings.items():
        coef_matrix = []

        for C in C_values:
            model = LogisticRegression(
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=MAX_ITER,
                random_state=RANDOM_STATE,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                model.fit(X_processed, y)

            coef_matrix.append(model.coef_[0])

        coef_matrix = np.array(coef_matrix)

        penalty_df = pd.DataFrame(coef_matrix, columns=feature_names)
        penalty_df["C"] = C_values
        penalty_df["penalty"] = penalty

        penalty_df = penalty_df.melt(
            id_vars=["C", "penalty"],
            var_name="feature",
            value_name="coefficient"
        )

        all_results.append(penalty_df)

    coef_df = pd.concat(all_results, ignore_index=True)
    return coef_df


def find_l1_zero_order(coef_df, feature_names, C_values):
    l1_subset = coef_df[coef_df["penalty"] == "l1"].copy()

    zero_order = []

    # scan from large C -> small C
    C_desc = sorted(C_values, reverse=True)

    for feature in feature_names:
        feature_path = (
            l1_subset[l1_subset["feature"] == feature]
            .sort_values("C", ascending=False)
            .reset_index(drop=True)
        )

        coeffs = feature_path["coefficient"].to_numpy()
        c_vals = feature_path["C"].to_numpy()

        first_zero_C = None

        for i in range(1, len(coeffs)):
            was_nonzero = not np.isclose(coeffs[i - 1], 0.0, atol=ZERO_TOL)
            is_zero_now = np.isclose(coeffs[i], 0.0, atol=ZERO_TOL)

            if was_nonzero and is_zero_now:
                first_zero_C = c_vals[i]
                break

        final_abs_coef = abs(coeffs[0])  # coefficient at largest C (weakest regularization)

        zero_order.append(
            {
                "feature": feature,
                "first_zero_C": first_zero_C,
                "final_abs_coef": final_abs_coef,
            }
        )

    zero_order_df = pd.DataFrame(zero_order)
    zero_order_df = zero_order_df.sort_values(
        by=["first_zero_C", "final_abs_coef"],
        ascending=[True, False],
        na_position="last"
    ).reset_index(drop=True)

    return zero_order_df


def plot_paths(coef_df, feature_names, zero_order_df, C_values, output_path="regularization_paths.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # highlight strongest features at largest C to reduce clutter
    l2_large_c = coef_df[
        (coef_df["penalty"] == "l2") & (np.isclose(coef_df["C"], max(C_values)))
    ].copy()
    top_features = (
        l2_large_c.assign(abs_coef=l2_large_c["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)["feature"]
        .head(8)
        .tolist()
    )

    for ax, penalty in zip(axes, ["l1", "l2"]):
        subset = coef_df[coef_df["penalty"] == penalty]

        for feature in feature_names:
            feature_data = subset[subset["feature"] == feature].sort_values("C")

            if feature in top_features:
                ax.plot(
                    feature_data["C"],
                    feature_data["coefficient"],
                    linewidth=2,
                    alpha=0.95,
                    label=feature
                )
            else:
                ax.plot(
                    feature_data["C"],
                    feature_data["coefficient"],
                    linewidth=1,
                    alpha=0.25
                )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel("C (log scale)")
        ax.set_title(f"Logistic Regression Coefficient Paths ({penalty.upper()})")

    axes[0].set_ylabel("Coefficient value")

    # annotate first 5 zeroed-out features on L1 plot
    # Annotate the first 5 features that get zeroed under L1
    l1_subset = coef_df[coef_df["penalty"] == "l1"]
    top_zeroed = zero_order_df.dropna(subset=["first_zero_C"]).head(5)

    short_names = {
        "contract_type_Two year": "contract: two year",
        "contract_type_One year": "contract: one year",
        "payment_method_Mailed check": "payment: mailed check",
        "internet_service_Fiber optic": "internet: fiber optic",
        "monthly_charges": "monthly charges",
        "total_charges": "total charges",
        "has_partner": "has partner",
        "gender_Male": "male",
        "num_support_calls": "support calls",
        "tenure": "tenure",
    }

    y_positions = [0.90, 0.82, 0.74, 0.66, 0.58]

    for ((_, row), y_pos) in zip(top_zeroed.iterrows(), y_positions):
        feature = row["feature"]
        C_zero = row["first_zero_C"]

        point = l1_subset[
            (l1_subset["feature"] == feature) & (np.isclose(l1_subset["C"], C_zero))
        ].iloc[0]

        label = short_names.get(feature, feature)

        axes[0].annotate(
            label,
            xy=(C_zero, point["coefficient"]),
            xycoords="data",
            xytext=(0.03, y_pos),
            textcoords="axes fraction",
            ha="left",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    fig.suptitle("Regularization Explorer: Telecom Churn", fontsize=14)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    X, y = load_data()

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)
    C_values = np.logspace(-3, 2, 20)

    coef_df = fit_regularization_paths(X_processed, y, feature_names, C_values)
    zero_order_df = find_l1_zero_order(coef_df, feature_names, C_values)

    print("\nFeatures zeroed out first under L1:")
    print(zero_order_df.dropna(subset=["first_zero_C"]).head(10).to_string(index=False))

    plot_paths(coef_df, feature_names, zero_order_df, C_values)


if __name__ == "__main__":
    main()