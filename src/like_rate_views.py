from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess


# ustawienia analizy
LOWESS_FRAC = 0.4854
LOWESS_ITERATIONS = 3
BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 12_345
TOP_VIDEOS = 7


# ścieżki względem głównego folderu projektu
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "public" / "views_like_rate_public.csv"
FIGURES_DIR = PROJECT_DIR / "figures"


def load_data() -> pd.DataFrame:
    required_columns = {"video_id", "views", "likes", "like_rate"}

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku danych: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Brakuje wymaganych kolumn: {missing}")

    data = data.loc[:, ["video_id", "views", "likes", "like_rate"]].copy()
    data["views"] = pd.to_numeric(data["views"], errors="raise")
    data["likes"] = pd.to_numeric(data["likes"], errors="raise")
    data["like_rate"] = pd.to_numeric(data["like_rate"], errors="raise")

    if data["video_id"].duplicated().any():
        raise ValueError("Kolumna video_id zawiera duplikaty")

    if len(data) != 103:
        raise ValueError(f"Oczekiwano 103 obserwacji, otrzymano {len(data)}")

    if (data["views"] <= 0).any():
        raise ValueError("Liczba wyświetleń musi być dodatnia")

    if (data["like_rate"] <= 0).any():
        raise ValueError("Like Rate musi być dodatni")

    # zaokrąglenie odtwarza dokładność wcześniejszego widoku MariaDB
    data["like_rate"] = data["like_rate"].round(4)
    data["like_rate_pct"] = 100 * data["like_rate"]
    data["log10_views"] = np.log10(data["views"])

    return data


def fit_elasticity(data: pd.DataFrame) -> dict[str, object]:
    model_data = data.loc[
        (data["views"] > 0) & (data["like_rate"] > 0)
    ].copy()

    model_data["ln_views"] = np.log(model_data["views"])
    model_data["ln_like_rate"] = np.log(model_data["like_rate"])

    explanatory = sm.add_constant(model_data["ln_views"])
    explained = model_data["ln_like_rate"]

    model = sm.OLS(explained, explanatory).fit(
        cov_type="HC3",
        use_t=True,
    )

    alpha = float(model.params["const"])
    beta = float(model.params["ln_views"])
    beta_ci = model.conf_int().loc["ln_views"]

    views_grid = np.geomspace(
        model_data["views"].min(),
        model_data["views"].max(),
        400,
    )
    prediction_pct = 100 * np.exp(
        alpha + beta * np.log(views_grid)
    )

    return {
        "model": model,
        "alpha": alpha,
        "beta": beta,
        "beta_ci_lower": float(beta_ci.iloc[0]),
        "beta_ci_upper": float(beta_ci.iloc[1]),
        "beta_p_value": float(model.pvalues["ln_views"]),
        "r_squared": float(model.rsquared),
        "n": len(model_data),
        "change_for_10_pct": 100 * (1.10**beta - 1),
        "change_for_doubling": 100 * (2**beta - 1),
        "views_grid": views_grid,
        "prediction_pct": prediction_pct,
    }


def calculate_spearman(data: pd.DataFrame) -> dict[str, float | int]:
    views = data["views"].to_numpy()
    like_rate = data["like_rate"].to_numpy()
    rho, p_value = spearmanr(views, like_rate)

    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_rho = np.empty(BOOTSTRAP_SAMPLES)
    sample_size = len(data)

    for iteration in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, sample_size, size=sample_size)
        bootstrap_rho[iteration] = spearmanr(
            views[indices],
            like_rate[indices],
        ).statistic

    bootstrap_rho = bootstrap_rho[~np.isnan(bootstrap_rho)]
    ci_lower, ci_upper = np.percentile(bootstrap_rho, [2.5, 97.5])

    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n": sample_size,
    }


def calculate_lowess(data: pd.DataFrame) -> np.ndarray:
    return lowess(
        endog=data["like_rate_pct"],
        exog=data["log10_views"],
        frac=LOWESS_FRAC,
        it=LOWESS_ITERATIONS,
        return_sorted=True,
    )


def print_results(
    label: str,
    elasticity: dict[str, object],
    spearman: dict[str, float | int],
) -> None:
    print(f"\n{label}")
    print("-" * len(label))

    print("\nModel elastyczności log-log")
    print("ln(Like Rate) = alpha + beta * ln(Views) + epsilon")
    print(f"alpha = {elasticity['alpha']:.4f}")
    print(f"beta = {elasticity['beta']:.4f}")
    print(
        "95% CI dla beta = "
        f"[{elasticity['beta_ci_lower']:.4f}; "
        f"{elasticity['beta_ci_upper']:.4f}]"
    )
    print(f"p-value dla beta = {elasticity['beta_p_value']:.6g}")
    print(f"R^2 = {elasticity['r_squared']:.4f}")
    print(f"n = {elasticity['n']}")
    print(
        "Zmiana Like Rate przy wzroście Views o 10% = "
        f"{elasticity['change_for_10_pct']:.2f}%"
    )
    print(
        "Zmiana Like Rate przy podwojeniu Views = "
        f"{elasticity['change_for_doubling']:.2f}%"
    )

    print("\nKorelacja rang Spearmana")
    print(f"rho_S = {spearman['rho']:.4f}")
    print(
        "95% bootstrap CI = "
        f"[{spearman['ci_lower']:.4f}; {spearman['ci_upper']:.4f}]"
    )
    print(f"p-value = {spearman['p_value']:.6g}")
    print(f"n = {spearman['n']}")


def save_plot(
    data: pd.DataFrame,
    elasticity: dict[str, object],
    smooth: np.ndarray,
    title: str,
    output_name: str,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 7))
    plt.scatter(
        data["views"],
        data["like_rate_pct"],
        s=45,
        alpha=0.65,
        color="#2878B5",
        edgecolor="white",
        linewidth=0.5,
        label="TikToki",
    )
    plt.plot(
        10 ** smooth[:, 0],
        smooth[:, 1],
        color="#D62728",
        linewidth=3,
        label="LOWESS",
    )
    plt.plot(
        elasticity["views_grid"],
        elasticity["prediction_pct"],
        color="#2CA02C",
        linewidth=2.5,
        linestyle="--",
        label=f"Model elastyczności (beta = {elasticity['beta']:.3f})",
    )

    plt.xscale("log")
    plt.xlabel("Liczba wyświetleń — skala logarytmiczna")
    plt.ylabel("Like Rate [%]")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / output_name,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def analyze_sample(
    data: pd.DataFrame,
    label: str,
    title: str,
    output_name: str,
) -> None:
    elasticity = fit_elasticity(data)
    spearman = calculate_spearman(data)
    smooth = calculate_lowess(data)

    print_results(label, elasticity, spearman)
    save_plot(data, elasticity, smooth, title, output_name)


def main() -> None:
    full_data = load_data()

    analyze_sample(
        data=full_data,
        label="Pełna próba",
        title=f"Like Rate a liczba wyświetleń (n = {len(full_data)})",
        output_name="views_like_rate_lowess.png",
    )

    top_seven = full_data.nlargest(TOP_VIDEOS, "views").sort_values(
        "views",
        ascending=False,
    )

    print("\nWyłączone TOP 7 według liczby wyświetleń")
    print(
        top_seven[["video_id", "views", "like_rate"]].to_string(
            index=False
        )
    )

    data_without_top = full_data.drop(index=top_seven.index).copy()
    data_without_top.reset_index(drop=True, inplace=True)

    analyze_sample(
        data=data_without_top,
        label="Analiza odporności bez TOP 7",
        title=(
            "Like Rate a liczba wyświetleń — bez TOP 7 "
            f"(n = {len(data_without_top)})"
        ),
        output_name="views_like_rate_without_top7.png",
    )

    print(f"\nWykresy zapisano w: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
