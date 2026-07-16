"""
Visualization Module
--------------------
This module contains visualization functions
for the House Price Prediction project.
"""

import plotly.express as px


def plot_histogram(df, column):
    """
    Create Histogram
    """
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=f"Histogram of {column}"
    )
    return fig


def plot_boxplot(df, column):
    """
    Create Box Plot
    """
    fig = px.box(
        df,
        y=column,
        title=f"Box Plot of {column}"
    )
    return fig


def plot_scatter(df, x_axis, y_axis):
    """
    Create Scatter Plot
    """
    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color="median_house_value",
        title=f"{x_axis} vs {y_axis}"
    )
    return fig


def plot_heatmap(df):
    """
    Create Correlation Heatmap
    """
    corr = df.select_dtypes(include="number").corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    return fig


def plot_target_distribution(df):
    """
    Distribution of Target Variable
    """
    fig = px.histogram(
        df,
        x="median_house_value",
        nbins=40,
        title="House Price Distribution"
    )

    return fig


def plot_income_vs_price(df):
    """
    Median Income vs House Price
    """
    fig = px.scatter(
        df,
        x="median_income",
        y="median_house_value",
        color="median_house_value",
        title="Median Income vs House Price"
    )

    return fig


def plot_ocean_proximity(df):
    """
    Ocean Proximity Count
    """
    fig = px.histogram(
        df,
        x="ocean_proximity",
        title="Ocean Proximity Distribution"
    )

    return fig