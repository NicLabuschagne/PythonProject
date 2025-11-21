import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

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

