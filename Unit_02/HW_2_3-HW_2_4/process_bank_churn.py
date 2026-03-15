import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_data(
    raw_df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw DataFrame into train and validation sets with stratification.

    Args:
        raw_df: Raw DataFrame with all columns.
        test_size: Fraction of data to use for validation.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df).
    """
    return train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[target_col],
    )


def encode_categorical(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """Encode categorical columns using OneHotEncoder.

    The encoder is fit only on training data.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        categorical_cols: List of categorical columns to encode.

    Returns:
        Tuple of (train_encoded_df, val_encoded_df, fitted_encoder).
    """
    encoder = OneHotEncoder(
        drop="if_binary",
        sparse_output=False,
        handle_unknown="ignore",
    )

    train_encoded = encoder.fit_transform(train_df[categorical_cols])
    val_encoded = encoder.transform(val_df[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    train_encoded_df = pd.DataFrame(train_encoded, index=train_df.index, columns=encoded_cols)
    val_encoded_df = pd.DataFrame(val_encoded, index=val_df.index, columns=encoded_cols)

    train_result = train_df.drop(columns=categorical_cols).join(train_encoded_df)
    val_result = val_df.drop(columns=categorical_cols).join(val_encoded_df)

    return train_result, val_result, encoder


def scale_numeric(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale numeric columns using StandardScaler.

    The scaler is fit only on training data.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        numeric_cols: List of numeric columns to scale.

    Returns:
        Tuple of (train_scaled_df, val_scaled_df, fitted_scaler).
    """
    scaler = StandardScaler()

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])

    return train_df, val_df, scaler


def preprocess_data(
    raw_df: pd.DataFrame,
    exclude_cols: list[str],
    target_col: str,
    scale_numeric_features: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], StandardScaler | None, OneHotEncoder]:
    """Full preprocessing pipeline for the bank churn dataset.

    Performs train/val split, categorical encoding, and (optionally) numeric
    scaling. All transformers are fit only on training data.

    Args:
        raw_df: Raw DataFrame (typically train.csv after set_index).
        scale_numeric_features: Whether to scale numeric features. Use True
            for logistic regression, False for decision trees.
        test_size: Fraction of data to use for validation.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of:
            - X_train: Training features.
            - train_targets: Training labels.
            - X_val: Validation features.
            - val_targets: Validation labels.
            - input_cols: Original feature column names (before encoding).
            - scaler: Fitted StandardScaler or None.
            - encoder: Fitted OneHotEncoder.
    """
    input_cols = [col for col in raw_df.columns if col not in exclude_cols]

    train_df, val_df = split_data(
        raw_df, 
        target_col=target_col, 
        test_size=test_size, 
        random_state=random_state
        )

    X_train = train_df[input_cols].copy()
    X_val = val_df[input_cols].copy()
    train_targets = train_df[target_col]
    val_targets = val_df[target_col]

    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

    X_train, X_val, encoder = encode_categorical(X_train, X_val, categorical_cols)

    scaler = None
    if scale_numeric_features:
        X_train, X_val, scaler = scale_numeric(X_train, X_val, numeric_cols)

    return X_train, train_targets, X_val, val_targets, input_cols, scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: list[str],
    encoder: OneHotEncoder,
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    """Preprocess new data (e.g. test.csv) using already fitted transformers.

    Args:
        new_df: New DataFrame (e.g. test.csv after set_index).
        input_cols: Original feature column names returned by preprocess_data().
        encoder: Fitted OneHotEncoder returned by preprocess_data().
        scaler: Fitted StandardScaler returned by preprocess_data(), or None.

    Returns:
        Processed DataFrame ready for prediction.
    """
    X = new_df[input_cols].copy()

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    # Encode categorical features
    encoded = encoder.transform(X[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded, index=X.index, columns=encoded_cols)
    X = X.drop(columns=categorical_cols).join(encoded_df)

    # Scale numeric features (if scaler is provided)
    if scaler is not None:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X