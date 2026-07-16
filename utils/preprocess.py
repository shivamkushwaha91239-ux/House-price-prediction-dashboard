"""
Preprocessing Module
--------------------
This module contains functions for cleaning and preprocessing
the California Housing dataset.
"""

import pandas as pd
import numpy as np


def clean_data(df):
    """
    Clean the housing dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill missing values in numerical columns
    numerical_columns = df.select_dtypes(include=np.number).columns

    df[numerical_columns] = df[numerical_columns].fillna(
        df[numerical_columns].median()
    )

    return df


def encode_data(df):
    """
    Encode categorical columns.
    """

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()

    df["ocean_proximity"] = encoder.fit_transform(
        df["ocean_proximity"]
    )

    return df, encoder


def split_features_target(df):
    """
    Split dataset into features and target.
    """

    X = df.drop("median_house_value", axis=1)

    y = df["median_house_value"]

    return X, y