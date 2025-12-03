import polars as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

from torchvision.transforms.v2.functional import to_tensor

# Set Seed for reproducibility

# In model_development.py - Final set_seeds function

def set_seeds(seed: int = 99):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 1. Standard PyTorch Seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2. STRICT Determinism Flags (CRUCIAL for stability)
    torch.use_deterministic_algorithms(True)

    # Optional: Disables backend acceleration which can introduce non-determinism
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Initializing Train Test Split

def time_split(
        df: pl.DataFrame,
        train_size: float = 0.75):
    split_idx = int(len(df) * train_size)
    return df[:split_idx], df[split_idx:].clone()

# Arguments:
    # df: polars data frame
    # train_size: 0.75 with test size being 0.25
    # function will return and split according to split percentage


# Create tensors from Polars Data Frame

# In model_development.py

def to_tensor(x, dtype=None) -> torch.Tensor:
    # 1. Convert Polars/NumPy to PyTorch Tensor
    t = torch.tensor(x.to_numpy(), dtype=torch.float32 if dtype is None else dtype)

    # 2. Check if the tensor is 1D (only has one dimension in its shape)
    # This happens when you have only one feature (N,)
    if t.ndim == 1:
        # Reshape to (N, 1) - N rows, 1 column (feature)
        t = t.reshape(-1, 1)

    return t


def train_test_split(
        df: pl.DataFrame,
        features: list,
        target: str,
        train_size: float = 0.75) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1. Drop NaNs first
    df = df.drop_nulls()

    # 2. Perform the ts split on the Polars DataFrame
    df_train, df_test = time_split(df, train_size)

    # 3. Extract features and targets and convert them to PyTorch tensors

    # Training Data
    X_train = to_tensor(df_train[features])
    y_train = to_tensor(df_train[target]).reshape(-1, 1)  # Reshape for compatibility with PyTorch losses

    # Testing Data
    X_test = to_tensor(df_test[features])
    y_test = to_tensor(df_test[target]).reshape(-1, 1)

    return X_train, X_test, y_train, y_test

#############################################################################

# Model Development

# Define Model Class for the Linear Regression model

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        # nn.Linear(input_size, 1) is a linear regression model
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Define Model Class for the Neural Network model

class TradingModel(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(TradingModel, self).__init__()

        # We use nn.Sequential to stack the layers
        self.layer_stack = nn.Sequential(
            # Hidden Layer 1: 64 neurons
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Hidden Layer 2: 32 neurons
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output Layer
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.layer_stack(x)


# MODEL DEFINITION ###

def train_model(
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        input_size: int, # Number of Features
        model_type: str = "linear",  # This can be linear or nn
        lr: float = 0.0005,
        epochs: int = 5000,
        print_freq: int = 500,
) -> nn.Module:  # Return type is the general nn.Module

    # 1. MODEL INSTANTIATION
    if model_type.lower() == "linear":
        model = LinearRegressionModel(input_size=input_size)
        print("Model: Linear Regression")

    elif model_type.lower() == "nn":
        model = TradingModel(input_size=input_size)
        print("Model: Neural Network (MLP)")

    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'linear' or 'nn'.")

    # 2. DEFINITIONS FOR LOSS AND OPTIMIZER
    criterion = nn.L1Loss() # This can be experimented with for performance
    # Hyperparameter
    optimizer = optim.Adam(model.parameters(), lr=lr) # This can be experimented with for performance
    #  optimizer = optim.Adam(model.parameters(), lr=lr)
    # 3. TRAINING LOOP
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # training loop logic (zero_grad, forward, loss, backward)
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % print_freq == 0:
            print(f"Epoch: {epoch:04d} | Loss: {loss.item():.6f}")

    print("\nTraining Complete.")
    #print("Final weight:", model.weight.data)
    #print("Final bias:", model.bias.data)
    return model

# Evaluate Model Performance denoted by y_hat

def evaluate_model(model: nn.Module, X_test: torch.Tensor) -> torch.Tensor:
    """Generates predictions on the test set."""
    # Set the model to evaluation mode (crucial for Dropoutlayers)
    model.eval()

    # Disable gradient calculations (saves memory and speeds things up)
    with torch.no_grad():
        # Generate predictions
        y_hat = model(X_test)

    return y_hat

# Create Polars DataFrame with trade results

def get_trade_results(
        y_hat: torch.Tensor,
        y_test: torch.Tensor) -> pl.DataFrame:
    """
    Calculates key trading metrics and constructs the equity curve.

    Args:
        y_hat (torch.Tensor): Model's predictions (e.g., predicted log returns).
        y_test (torch.Tensor): True target values (actual log returns).

    Returns:
        pl.DataFrame: DataFrame containing predictions, actual returns, signals,
                      trade returns, and the cumulative equity curve.
    """

    # Ensure the tensors are on the CPU and converted to NumPy arrays
    # then squeeze out any single dimensions
    y_hat_np = y_hat.cpu().detach().numpy().squeeze()
    y_test_np = y_test.cpu().detach().numpy().squeeze()

    # 1. Create the Polars DataFrame
    trade_results = pl.DataFrame({
        'y_hat': y_hat_np,
        'y': y_test_np,
    })

    # 2. Calculate signal and win/loss
    trade_results = trade_results.with_columns(
        # is_won: True if the sign of prediction matches the sign of actual return
        (pl.col('y_hat').sign() == pl.col('y').sign()).alias('is_won'),
        # signal: The direction the model predicted (-1 or 1)
        pl.col('y_hat').sign().alias('signal'),
    )

    # 3. Calculate Trade Return and Equity Curve
    trade_results = trade_results.with_columns([
        # Trade Log Return: The actual return (y) multiplied by the signal (1 or -1)
        (pl.col('signal') * pl.col('y')).alias('trade_log_return'),
    ]).with_columns([
        # Equity Curve: The cumulative sum of all trade log returns
        pl.col('trade_log_return').cum_sum().alias('equity_curve'),
    ])

    # 4. Calculate Buy and Hold (B&H) Metrics
    trade_results = trade_results.with_columns(
        # B&H Log Return is simply the actual return (y)
        pl.col("y").alias("B_H_log_return")

    ).with_columns(
        # B&H Equity Curve: Cumulative sum of the B&H Log Return
        pl.col("B_H_log_return").cum_sum().alias("B_H_equity_curve")

    ).with_columns(
        # Drawdown (calculated on the Strategy's Equity Curve)
        (pl.col("equity_curve") - pl.col("equity_curve").cum_max()).alias("drawdown_log")
    )

    return trade_results

# Create a performance metrics Data Frame

def get_strategy_metrics(
        trade_results_df: pl.DataFrame,
        # NOTE: This factor should be the square root of the number of periods per year!
        annual_factor_sqrt: float = 15.87,
        risk_free_rate: float = 0.0
) -> pl.DataFrame:



    strategy_returns = trade_results_df['trade_log_return']
    volatility = strategy_returns.std()
    mean_return = strategy_returns.mean()

    # --- 2. SHARPE RATIO CALCULATION ---
    # Annualized Sharpe = ((Mean Return - Risk Free Rate) / Volatility) * Annualization Factor (Sqrt)
    if volatility == 0:
        sharpe_ratio = 0.0
    else:

        sharpe_ratio = ((mean_return - risk_free_rate) / volatility) * annual_factor_sqrt

    # METRICS
    total_log_return = trade_results_df['trade_log_return'].sum()
    # MDD calculation using Pandas
    max_drawdown = trade_results_df['drawdown_log'].min()
    drawdown_pct = np.exp(max_drawdown)-1
    compound_ret = np.exp(total_log_return)


    # --- 4. ASSEMBLE RESULTS ---

    metrics = pl.DataFrame({
        'Total_Return': [total_log_return],
        'Annualized_Sharpe': [sharpe_ratio],
        'Max_Drawdown': [-max_drawdown],
        'Drawdown_Pct': [drawdown_pct],
        'Win_Rate': [trade_results_df['is_won'].mean()],
        'Total_Trades': [len(trade_results_df)],
        "Compound Return": [compound_ret],
    })

    return metrics.with_columns(
        # Calmar Ratio: Total Return / Max Drawdown (Absolute Value)
        (pl.col('Total_Return') / pl.col('Max_Drawdown').abs()).alias('Calmar_Ratio')
    )