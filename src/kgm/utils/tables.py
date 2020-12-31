"""Tools for creating and formatting tables."""
from typing import Callable, Sequence

import pandas

from kgm.utils.common import argparse_bool


def normalize_boolean(
    df: pandas.DataFrame,
    column: str,
    false_name: str = "no",
    true_name: str = "yes",
    normalize_from_string: bool = True,
) -> pandas.Series:
    """Normalize a boolean column in-place."""
    values = df[column]
    if normalize_from_string:
        values = values.apply(argparse_bool)
    return values.apply(lambda x: true_name if x else false_name)


def normalize_similarity(
    df: pandas.DataFrame,
    cls_column_name: str = "params.similarity.cls",
    transformation_column_name: str = "params.similarity.transformation",
) -> pandas.Series:
    """Normalize similarity by combining cls and transformation."""
    similarity = df[cls_column_name].str.lower()

    # transformation only active if similarity in {l1, l2}
    trans_mask = similarity.str.match("l[1-2]")
    similarity[trans_mask] = similarity[trans_mask] + " (" + df[trans_mask][transformation_column_name] + ")"

    return similarity


def normalize_embedding_norm(
    df: pandas.DataFrame,
    embedding_norm_method_column: str = "params.model.embedding_norm",
    embedding_norm_mode_column: str = "params.model.embedding_norm_mode",
) -> pandas.Series:
    """Normalize embedding normalization by combining norm and norm-mode."""
    norm = df[embedding_norm_method_column].apply(lambda v: v.rsplit(".", maxsplit=1)[-1])
    mode = df[embedding_norm_mode_column].apply(lambda v: v.rsplit(".", maxsplit=1)[-1])
    normalization = (norm + "_" + mode).str.replace("none_none", "none")
    return normalization.apply(dict(
        none="never",
        l1_initial="l1 initial",
        l1_every_forward="l1 always",
        l2_initial="l2 initial",
        l2_every_forward="l2 always",
    ).__getitem__)


def normalize_subset(
    df: pandas.DataFrame,
    column: str,
    drop_suffix: str = "-15k-v2",
) -> pandas.Series:
    """Normalize subset name in place."""

    def _normalize_subset(name: str) -> str:
        return name.lower().replace('_', '-').replace(drop_suffix, "")

    return df[column].apply(_normalize_subset)


def latex_bold(text: str) -> str:
    """Format text in bold font using Latex."""
    return rf"\textbf{{{text}}}"


def highlight_max(
    data: pandas.Series,
    float_formatter: Callable[[float], str] = "{:2.2f}".format,
    highlighter: Callable[[str], str] = latex_bold,
    larger_is_better: bool = True,
) -> pandas.Series:
    """Highlight best value in each column."""
    best = data.max() if larger_is_better else data.min()
    is_best = data == best
    data = data.apply(float_formatter).str.replace("nan", "")
    data[is_best] = data[is_best].apply(highlighter)
    return data


def format_ablation_table(
    df: pandas.DataFrame,
    cell_value: str,
    groups: Sequence[str],
    ablation_name_column_name: str = "parameter",
    ablation_value_column_name: str = "value",
    float_formatter: Callable[[float], str] = "{:2.2f}".format,
    highlighter: Callable[[str], str] = latex_bold,
    larger_is_better: bool = True,
) -> pandas.DataFrame:
    """
    Format an ablation table.

    :param df:
        The dataframe obtained from ablation. In particular it has to have the ablation name/value columns.
    :param cell_value:
        The value column.
    :param groups:
        The groups to use for the columns of the table.
    :param ablation_name_column_name:
        The name of the the ablation parameter name column.
    :param ablation_value_column_name:
        The name of the the ablation parameter value column.
    :param float_formatter:
        The formatter to use for formatting float values.
    :param highlighter:
        The highlighting to use for the best entries in each group.
    :param larger_is_better:
        Whether larger values are better.

    :return:
        A table with groups as columns, ablation parameter name/value as index, and highlighting of the best entry in
        each ablation parameter name - group combination.
    """
    keep_columns = [ablation_name_column_name, ablation_value_column_name, *groups, cell_value]

    # highlight within groups of ablation table
    out = []
    for _, values in df.groupby(by=[ablation_name_column_name, *groups]):
        values = values[keep_columns].copy()
        values[cell_value] = highlight_max(
            data=values[cell_value],
            float_formatter=float_formatter,
            highlighter=highlighter,
            larger_is_better=larger_is_better,
        )
        out.append(values)
    if len(out) == 0:
        df = pandas.DataFrame(columns=keep_columns)
    else:
        df = pandas.concat(out, ignore_index=True)

    # escape under-score
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace("_", r"\_")

    # pivot table
    df = df.set_index([ablation_name_column_name, ablation_value_column_name, *groups])[cell_value]
    for _ in range(len(groups)):
        df = df.unstack()
    return df


def combine_tables(
    *tables: pandas.DataFrame,
    sep: str = "/",
) -> pandas.DataFrame:
    """
    Element-wise combination of tables.

    :param tables:
        The tables.
    :param sep:
        The separator to use.

    :return:
        The combined table, a dataframe with the same index and columns as the input table, and string elements.
    """
    if not tables:
        raise ValueError("You must at least provide one table.")
    if len(tables) < 2:
        return tables[0]

    def combination(first: pandas.Series, second: pandas.Series):
        """Concat the string representations of both series and insert the separator in between."""
        return first.astype(str) + sep + second.astype(str)

    first, *other = tables
    return first.combine(other=combine_tables(*other, sep=sep), func=combination)
