import pandas as pd
import matplotlib.pyplot as plt 
from typing import List, Optional, Dict, Any


def plot_pie_chart(
    df: pd.DataFrame,
    column_name: str,
    labels: List[str],
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: tuple = (8, 6),
    autopct: str = '%1.1f%%',
    explode: Optional[List[float]] = None,
    shadow: bool = False,
    startangle: int = 90,
    additional_kwargs: Optional[Dict[str, Any]] = None
) -> None:
    """
    Plot a pie chart of the distribution of values in a DataFrame column.

    Parameters:
        df (pd.DataFrame): The input data frame.
        column_name (str): The name of the column to analyze.
        labels (List[str]): List of category labels to include. Data not in this list will be ignored.
        title (Optional[str]): Chart title. Defaults to a formatted title from the column name.
        colors (Optional[List[str]]): Colors for each pie slice. If None, the 'Pastel1' colormap is used.
        figsize (tuple): Figure size, defaults to (8, 6).
        autopct (str): Format string for displaying percentages, defaults to '%1.1f%%'.
        explode (Optional[List[float]]): Offsets for the pie slices. If provided, must match the number of labels.
        shadow (bool): If True, draws a shadow under the pie, defaults to False.
        startangle (int): Starting angle of the pie, defaults to 90.
        additional_kwargs (Optional[Dict[str, Any]]): Any additional keyword arguments to pass to plt.pie().

    Returns:
        None
    """
    # Compute counts for categories, ensuring the order via reindex
    category_counts = df[column_name].value_counts().reindex(labels, fill_value=0)

    plt.figure(figsize=figsize, constrained_layout=True)

    # Use custom colors if provided; else use matplotlib's Pastel1 colors
    if colors is None:
        colors = plt.cm.Pastel1.colors

    # Prepare additional keyword arguments if any
    pie_kwargs = additional_kwargs.copy() if additional_kwargs else {}
    
    plt.pie(
        category_counts,
        labels=category_counts.index,
        autopct=autopct,
        colors=colors,
        explode=explode,
        shadow=shadow,
        startangle=startangle,
        **pie_kwargs
    )
    
    # Set title for the chart
    if title is None:
        title = f'Distribution of {column_name.replace("_", " ").title()}'
    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_bar_chart(
    df: pd.DataFrame,
    column_name: str,
    labels: List[str],
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: tuple = (8, 6),
    additional_kwargs: Optional[Dict[str, Any]] = None
) -> None:
    """
    Plot a bar chart of the distribution of values in a DataFrame column.

    Parameters:
        df (pd.DataFrame): The input data frame.
        column_name (str): The name of the column to analyze.
        labels (List[str]): List of category labels to include. Data not in this list will be ignored.
        title (Optional[str]): Chart title. Defaults to a formatted title from the column name.
        colors (Optional[List[str]]): Colors for each bar. If None, a default matplotlib color cycle is used.
        figsize (tuple): Figure size, defaults to (8, 6).
        additional_kwargs (Optional[Dict[str, Any]]): Any additional keyword arguments to pass to plt.bar().

    Returns:
        None
    """
    # Compute counts for categories, ensuring the order via reindex
    category_counts = df[column_name].value_counts().reindex(labels, fill_value=0)

    plt.figure(figsize=figsize, constrained_layout=True)
    
    # Set default color if none provided
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Prepare additional keyword arguments if any
    bar_kwargs = additional_kwargs.copy() if additional_kwargs else {}

    bars = plt.bar(
        x=category_counts.index,
        height=category_counts.values,
        color=colors[:len(category_counts)],
        **bar_kwargs
    )

    # Annotate bars with counts (optional)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f'{height}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center',
            va='bottom'
        )

    # Set a default title if not provided
    if title is None:
        title = f'Distribution of {column_name.replace("_", " ").title()}'
    plt.title(title)

    plt.xlabel(column_name.replace("_", " ").title())
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()