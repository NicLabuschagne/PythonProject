import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
import pandas as pd

# Create plotting distributions to visualize various types of distributions within the DF.

# Generates a pair plot for specified features and displays it.

def plot_feature_distribution(df: pl.DataFrame, features: List[str])-> None:

# Transform the Polars df into Pandas for use of Seaborn

    pd_df = df.select(features).to_pandas()

# Create the pair plot

    sns.pairplot(pd_df,diag_kind="kde")
    plt.suptitle("Feature Distribution", y=1.05)
    plt.show()
    plt.close()

# Create a correlation matrix to identify feature importance to target variable.

def display_feature_corr(
        df: pl.DataFrame,
        target_col: str = "close_log_return") -> pl.DataFrame:

# Arguments:
    # df: The Polars Df containing all features and the target.
    # target_col: the name of the target column for the correlation.
# returns a Polars data frame containing the correlation values.

        corr_df = df.corr()

        target_corr = corr_df.select(pl.col(target_col))
        target_corr = target_corr.with_columns(pl.Series(name="Feature",values=corr_df.columns))

        target_corr = target_corr.select([pl.col("Feature"),
        pl.col(target_col).abs().alias("Abs_correlation")]).sort("Abs_correlation",descending=True)

        return target_corr


# Create a equity curve for cummulated log returns of predicted model

import polars as pl
import matplotlib.pyplot as plt

def plot_cum_trade_log_returns(
        df: pl.DataFrame,
        strategy_col: str = "equity_curve",
        benchmark_col: str = None) -> None:

    # Select the target column and convert to a pandas series for plotting.
    plot_data = df.select(pl.col(strategy_col)).to_series().to_pandas()
    plt.figure(figsize=(10, 5))
    plot_data.plot(
        kind='line',
        label='Model Strategy (Sharpe 1.8)',
        linewidth=2,
        color='orange'
    )

    if benchmark_col is not None:
        benchmark_df = df.select(pl.col(benchmark_col)).to_series().to_pandas()

        benchmark_df.plot(
            kind='line',
            label=benchmark_col,
            linewidth=2,
            color='lightblue',
            ax=plt.gca()
        )
    # 3. Add titles and labels
    plt.title("Cumulative Trade Log Returns: Strategy vs Benchmark", fontsize=16, y=1.03)
    plt.xlabel("Trade Number / Time Index")
    plt.ylabel("Cumulative Log Return")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
