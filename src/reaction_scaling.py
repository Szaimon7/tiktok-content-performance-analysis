from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ustawienia analizy
EXPECTED_ROWS = 103
TOP_VIDEOS = 7
BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 12_345


# ścieżki względem głównego folderu projektu
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "public" / "reaction_scaling_public.csv"
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"


# kolejność reakcji na wykresie i w tabelach
REACTIONS = {
    "likes": "Polubienia",
    "comments": "Komentarze",
    "shares": "Udostępnienia",
    "active_reactions": "Komentarze + udostępnienia",
}


def load_data() -> pd.DataFrame:
    required_columns = {
        "video_id",
        "views",
        "likes",
        "comments",
        "shares",
    }

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku danych: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Brakuje wymaganych kolumn: {missing}")

    data = data.loc[
        :, ["video_id", "views", "likes", "comments", "shares"]
    ].copy()

    for column in ["views", "likes", "comments", "shares"]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if len(data) != EXPECTED_ROWS:
        raise ValueError(
            f"Oczekiwano {EXPECTED_ROWS} obserwacji, otrzymano {len(data)}"
        )

    if data["video_id"].duplicated().any():
        raise ValueError("Kolumna video_id zawiera duplikaty")

    if data[["views", "likes", "comments", "shares"]].isna().any().any():
        raise ValueError("Dane zawierają brakujące wartości")

    if (data[["views", "likes", "comments", "shares"]] <= 0).any().any():
        raise ValueError("Wszystkie liczniki muszą być dodatnie")

    data["active_reactions"] = data["comments"] + data["shares"]

    return data


def fit_scaling_model(
    data: pd.DataFrame,
    reaction: str,
) -> dict[str, float | int]:
    model_data = data.loc[
        (data["views"] > 0) & (data[reaction] > 0)
    ].copy()

    model_data["ln_views"] = np.log(model_data["views"])
    model_data["ln_reaction"] = np.log(model_data[reaction])

    explanatory = sm.add_constant(model_data[["ln_views"]])
    explained = model_data["ln_reaction"]

    model = sm.OLS(explained, explanatory).fit(
        cov_type="HC3",
        use_t=True,
    )

    alpha = float(model.params["const"])
    gamma = float(model.params["ln_views"])
    gamma_ci = model.conf_int().loc["ln_views"]

    # test proporcjonalnego skalowania reakcji względem wyświetleń
    gamma_equals_one_test = model.t_test("ln_views = 1")
    test_statistic = float(
        np.asarray(gamma_equals_one_test.tvalue).squeeze()
    )
    test_p_value = float(
        np.asarray(gamma_equals_one_test.pvalue).squeeze()
    )

    return {
        "n": len(model_data),
        "alpha": alpha,
        "gamma": gamma,
        "gamma_ci_lower": float(gamma_ci.iloc[0]),
        "gamma_ci_upper": float(gamma_ci.iloc[1]),
        "test_gamma_equals_one_t": test_statistic,
        "p_value_gamma_equals_one": test_p_value,
        "r_squared": float(model.rsquared),
        "reaction_change_10_pct": 100 * (1.10**gamma - 1),
        "rate_change_10_pct": 100 * (1.10 ** (gamma - 1) - 1),
        "reaction_change_doubling_pct": 100 * (2**gamma - 1),
        "rate_change_doubling_pct": 100 * (2 ** (gamma - 1) - 1),
    }


def analyze_sample(
    data: pd.DataFrame,
    sample_name: str,
    sample_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for reaction, reaction_label in REACTIONS.items():
        result = fit_scaling_model(data, reaction)
        rows.append(
            {
                "sample": sample_name,
                "sample_label": sample_label,
                "reaction": reaction,
                "reaction_label": reaction_label,
                **result,
            }
        )

    return pd.DataFrame(rows)


def calculate_bootstrap_differences(
    data: pd.DataFrame,
    sample_name: str,
    sample_label: str,
    random_seed: int,
) -> pd.DataFrame:
    comparisons = [
        ("shares", "likes"),
        ("shares", "comments"),
    ]
    reaction_names = list(REACTIONS)
    log_views = np.log(data["views"].to_numpy(dtype=float))
    log_reactions = np.column_stack(
        [
            np.log(data[reaction].to_numpy(dtype=float))
            for reaction in reaction_names
        ]
    )
    centered_views = log_views - log_views.mean()
    centered_reactions = log_reactions - log_reactions.mean(axis=0)
    point_gammas = (
        centered_views @ centered_reactions
    ) / np.sum(centered_views**2)

    rng = np.random.default_rng(random_seed)
    sample_size = len(data)
    bootstrap_gammas = np.empty(
        (BOOTSTRAP_SAMPLES, len(reaction_names)),
        dtype=float,
    )

    for iteration in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, sample_size, size=sample_size)
        sampled_views = log_views[indices]
        sampled_reactions = log_reactions[indices]
        centered_views = sampled_views - sampled_views.mean()
        denominator = np.sum(centered_views**2)

        if denominator == 0:
            bootstrap_gammas[iteration, :] = np.nan
            continue

        centered_reactions = (
            sampled_reactions - sampled_reactions.mean(axis=0)
        )
        bootstrap_gammas[iteration, :] = (
            centered_views @ centered_reactions
        ) / denominator

    valid_rows = ~np.isnan(bootstrap_gammas).any(axis=1)
    bootstrap_gammas = bootstrap_gammas[valid_rows]
    reaction_positions = {
        reaction: position
        for position, reaction in enumerate(reaction_names)
    }

    rows: list[dict[str, object]] = []

    for first_reaction, second_reaction in comparisons:
        differences = (
            bootstrap_gammas[:, reaction_positions[first_reaction]]
            - bootstrap_gammas[:, reaction_positions[second_reaction]]
        )
        ci_lower, ci_upper = np.percentile(differences, [2.5, 97.5])
        lower_tail = (
            np.count_nonzero(differences <= 0) + 1
        ) / (len(differences) + 1)
        upper_tail = (
            np.count_nonzero(differences >= 0) + 1
        ) / (len(differences) + 1)
        approximate_p_value = min(
            1.0,
            2 * min(lower_tail, upper_tail),
        )

        first_position = reaction_positions[first_reaction]
        second_position = reaction_positions[second_reaction]
        point_difference = (
            point_gammas[first_position] - point_gammas[second_position]
        )

        rows.append(
            {
                "sample": sample_name,
                "sample_label": sample_label,
                "first_reaction": first_reaction,
                "second_reaction": second_reaction,
                "gamma_difference": float(point_difference),
                "bootstrap_ci_lower": float(ci_lower),
                "bootstrap_ci_upper": float(ci_upper),
                "approximate_p_value": approximate_p_value,
                "bootstrap_samples": len(differences),
            }
        )

    return pd.DataFrame(rows)


def print_results(
    results: pd.DataFrame,
    differences: pd.DataFrame,
    top_seven: pd.DataFrame,
) -> None:
    columns = [
        "reaction_label",
        "gamma",
        "gamma_ci_lower",
        "gamma_ci_upper",
        "p_value_gamma_equals_one",
        "reaction_change_doubling_pct",
        "rate_change_doubling_pct",
        "r_squared",
        "n",
    ]

    for sample_label in results["sample_label"].drop_duplicates():
        print(f"\n{sample_label}")
        print("-" * len(sample_label))
        sample_results = results.loc[
            results["sample_label"] == sample_label,
            columns,
        ].copy()
        print(sample_results.to_string(index=False, float_format="%.4f"))

    print("\nPorównania elastyczności w bootstrapie")
    print("--------------------------------------")
    print(differences.to_string(index=False, float_format="%.4f"))

    print("\nTOP 7 według liczby wyświetleń")
    print("--------------------------------")
    print(
        top_seven[["video_id", "views"]].to_string(index=False)
    )


def save_results(
    results: pd.DataFrame,
    differences: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        RESULTS_DIR / "reaction_elasticities.csv",
        index=False,
        float_format="%.8f",
    )
    differences.to_csv(
        RESULTS_DIR / "reaction_elasticity_differences.csv",
        index=False,
        float_format="%.8f",
    )


def save_plot(results: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    reaction_order = list(REACTIONS)
    reaction_labels = [REACTIONS[reaction] for reaction in reaction_order]
    base_positions = np.arange(len(reaction_order), dtype=float)
    sample_styles = {
        "full": {
            "offset": -0.11,
            "color": "#2878B5",
            "marker": "o",
            "label": "Pełna próba",
        },
        "without_top7": {
            "offset": 0.11,
            "color": "#D62728",
            "marker": "s",
            "label": "Bez TOP 7",
        },
    }

    figure, axis = plt.subplots(figsize=(10, 6.5))

    for sample_name, style in sample_styles.items():
        sample_results = (
            results.loc[results["sample"] == sample_name]
            .set_index("reaction")
            .loc[reaction_order]
        )
        gammas = sample_results["gamma"].to_numpy()
        lower_errors = (
            gammas - sample_results["gamma_ci_lower"].to_numpy()
        )
        upper_errors = (
            sample_results["gamma_ci_upper"].to_numpy() - gammas
        )

        axis.errorbar(
            gammas,
            base_positions + style["offset"],
            xerr=np.vstack([lower_errors, upper_errors]),
            fmt=style["marker"],
            markersize=7,
            capsize=4,
            linewidth=1.8,
            color=style["color"],
            label=style["label"],
        )

    axis.axvline(
        1,
        color="#444444",
        linestyle="--",
        linewidth=1.8,
        label="Skalowanie proporcjonalne",
    )
    axis.set_yticks(base_positions)
    axis.set_yticklabels(reaction_labels)
    axis.invert_yaxis()
    axis.set_xlabel("Elastyczność względem liczby wyświetleń")
    axis.set_title("Skalowanie reakcji względem liczby wyświetleń")
    axis.grid(True, axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "reaction_elasticities.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    full_data = load_data()
    top_seven = full_data.nlargest(TOP_VIDEOS, "views").sort_values(
        "views",
        ascending=False,
    )
    data_without_top = full_data.drop(index=top_seven.index).copy()
    data_without_top.reset_index(drop=True, inplace=True)

    full_results = analyze_sample(
        full_data,
        sample_name="full",
        sample_label="Pełna próba",
    )
    reduced_results = analyze_sample(
        data_without_top,
        sample_name="without_top7",
        sample_label="Bez TOP 7",
    )
    results = pd.concat(
        [full_results, reduced_results],
        ignore_index=True,
    )

    full_differences = calculate_bootstrap_differences(
        full_data,
        sample_name="full",
        sample_label="Pełna próba",
        random_seed=RANDOM_SEED,
    )
    reduced_differences = calculate_bootstrap_differences(
        data_without_top,
        sample_name="without_top7",
        sample_label="Bez TOP 7",
        random_seed=RANDOM_SEED + 1,
    )
    differences = pd.concat(
        [full_differences, reduced_differences],
        ignore_index=True,
    )

    save_results(results, differences)
    save_plot(results)
    print_results(results, differences, top_seven)

    print(f"\nWyniki zapisano w: {RESULTS_DIR}")
    print(f"Wykres zapisano w: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
