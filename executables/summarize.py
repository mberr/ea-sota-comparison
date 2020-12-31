"""Script to (re-)generate all tables of the paper."""
import argparse
import logging
import pathlib
from typing import Callable, Optional

import mlflow
import pandas
from mlflow.entities import ViewType

from kgm.data import SIDES, get_dataset_by_name
from kgm.utils.mlflow_utils import ablation, get_metric_history, get_results_from_hpo
from kgm.utils.tables import combine_tables, format_ablation_table, normalize_boolean, normalize_embedding_norm, normalize_similarity, normalize_subset


def latex_bold(text: str) -> str:
    """Format text in bold font using Latex."""
    return rf"\textbf{{{text}}}"


def highlight_max(
    data: pandas.Series,
    float_formatter: Callable[[float], str] = "{:2.2f}".format,
    highlighter: Callable[[str], str] = latex_bold,
) -> pandas.Series:
    """Highlight maximum value in each column."""
    is_max = data == data.max()
    data = data.apply(float_formatter).str.replace("nan", "")
    data[is_max] = data[is_max].apply(highlighter)
    return data


def _dataset_table(output_root: pathlib.Path) -> None:
    datasets = [
        ("dbp15k_jape", "zh_en"),
        ("dbp15k_jape", "ja_en"),
        ("dbp15k_jape", "fr_en"),
        ("wk3l15k", "en_de"),
        ("wk3l15k", "en_fr"),
        ("openea", "EN_DE_15K_V2"),
        ("openea", "EN_FR_15K_V2"),
        ("openea", "D_Y_15K_V2"),
        ("openea", "D_W_15K_V2"),
    ]
    g_data = []
    for dataset_name, subset_name in datasets:
        dataset = get_dataset_by_name(dataset_name=dataset_name, subset_name=subset_name)
        num_exclusives = dataset.num_exclusives
        for side, graph in dataset.graphs.items():
            g_data.append([
                dataset_name.split("_")[0],
                "-".join(map(str.lower, subset_name.split("_")[:2])),
                subset_name.split('_')[SIDES.index(side)].lower(),
                graph.num_entities,
                graph.num_relations,
                graph.num_triples,
                graph.num_entities - num_exclusives[side],
                num_exclusives[side],
            ])
    g_df = pandas.DataFrame(
        data=g_data,
        columns=[
            "dataset",
            "subset",
            "graph",
            "num_entities",
            "num_relations",
            "num_triples",
            "num_entities_shared",
            "num_entities_exclusive",
        ]
    )
    g_df.to_csv(output_root / "graphs.tsv", sep="\t")

    # prepare latex tables
    _int_formatter = "{:,}".format

    def _escape_underscore(value):
        if isinstance(value, str):
            return value.replace("_", r"\_")
        return value

    g_rename = dict(
        num_entities=r"$|\mathcal{E}|$",
        num_relations=r"$|\mathcal{R}|$",
        num_triples=r"$|\mathcal{T}|$",
        num_entities_shared=r"$|\mathcal{A}|$",
        num_entities_exclusive=r"$|\mathcal{X}|$",
    )
    g_index = ["dataset", "subset", "graph"]
    with (output_root / "datasets.tex").open(mode="wt") as tex_file:
        tex_file.write(
            g_df.applymap(
                _escape_underscore
            ).set_index(
                g_index
            ).rename(
                columns=g_rename,
            ).to_latex(
                escape=False,
                formatters={
                    col: _int_formatter
                    for col in g_rename.values()
                },
            )
        )


def _result_table(output_root: pathlib.Path, force: bool = False, tracking_uri: str = "http://localhost:5000") -> None:
    mlflow.set_tracking_uri(tracking_uri)

    data = [
        # Zero-Shot
        _get_runs_from_mlflow(
            experiment_name="zero_shot",
            method_name="zero_shot",
            dataset_column="params.dataset",
            subset_column="params.subset",
            init_column="params.init",
            metric_column="metrics.test.hits_at_1",
        ),
        # GCN-Align
        _get_runs_from_mlflow_from_hpo(
            tracking_uri=tracking_uri,
            experiment_name="gcn_align",
            method_name="gcn_align",
            dataset_column="params.data.dataset",
            subset_column="params.data.subset",
            init_column="params.model.node_embedding_init_method",
            force=force,
        ),
        # RDGCN
        _get_runs_from_mlflow_from_hpo(
            tracking_uri=tracking_uri,
            experiment_name="rdgcn",
            method_name="rdgcn",
            dataset_column="params.data.dataset",
            subset_column="params.data.subset",
            init_column="params.model.node_embedding_init_method",
            force=force,
        ),
        # DGMC
        _get_runs_from_mlflow_from_hpo(
            tracking_uri=tracking_uri,
            experiment_name="dgmc",
            method_name="dgmc",
            dataset_column="params.data.dataset",
            subset_column="params.data.subset",
            init_column="params.model.init",
            force=force,
        ),
    ]

    df = pandas.concat(data, axis=0, ignore_index=True)

    # normalize dataset
    df["dataset"] = df["dataset"].str.replace("_", "")
    # normalize subset
    df["subset"] = df["subset"].apply(lambda s: '-'.join(s.split('_')[:2]).lower())
    # normalize init
    rename = dict(
        bert_precomputed="BERT",
        rdgcn_precomputed="Wu",
        xu_precomputed="Xu",
        openea_rdgcn_precomputed="Sun",
    )
    df["init"] = df["init"].apply(rename.__getitem__)
    df.to_csv(output_root / "results.tsv", sep="\t", index=False)

    index = ["method", "subset", "init"]
    metric = "H@1"
    order = [
        ("zero_shot", "Zero Shot"),
        ("gcn_align", r"\acrshort{gcnalign}"),
        ("rdgcn", r"\acrshort{rdgcn}"),
        ("dgmc", r"\acrshort{dgmc}"),
    ]
    _translate_methods = dict(order)
    position = {
        k: v
        for v, (_, k) in enumerate(order)
    }
    df["method"] = df["method"].apply(_translate_methods.__getitem__)
    for dataset, group in df.groupby(by="dataset"):
        table = group.groupby(by=index).agg({metric: "mean"}).unstack().unstack().applymap(lambda v: 100 * v)
        table = table.iloc[sorted(range(len(table)), key=lambda i: position[table.index[i]])].copy()
        column_format = "l" + "*{" + str(len(table.columns)) + r"}{@{\extracolsep{\fill}}r}"
        table = table.apply(highlight_max, float_formatter="{:2.2f}".format, highlighter=latex_bold)
        with (output_root / f"results_{dataset}.tex").open(mode="wt") as tex_file:
            string = table.to_latex(column_format=column_format, escape=False)
            string = string.replace(r"}{l}{", r"}{c}{")
            string = string.split("\n")
            string = "\n".join(line for line in string if not line.startswith("method") and metric not in line)
            string = string.replace(r"\begin{tabular}", r"\begin{tabular*}{\textwidth}")
            string = string.replace(r"\end{tabular}", r"\end{tabular*}")
            tex_file.write(string)


def _get_runs_from_mlflow_from_hpo(
    tracking_uri: str,
    experiment_name: str,
    method_name: str,
    dataset_column: str = "params.dataset",
    subset_column: str = "params.subset",
    init_column: str = "params.init",
    metric_column: str = "eval.test.hits_at_1",
    validation_metric_column: str = "eval.validation.hits_at_1",
    buffer_to: Optional[str] = None,
    force: bool = False,
) -> pandas.DataFrame:
    exp_id = mlflow.get_experiment_by_name(name=experiment_name)
    if exp_id is None:
        raise ValueError(f"{experiment_name} does not exist at MLFlow instance at {tracking_uri}")
    exp_id = exp_id.experiment_id
    prefix = "metrics."
    if metric_column.startswith(prefix):
        metric_column = metric_column[len(prefix):]
    if validation_metric_column.startswith(prefix):
        validation_metric_column = validation_metric_column[len(prefix):]

    if buffer_to is None:
        buffer_to = pathlib.Path("/tmp") / f"{experiment_name}_{method_name}.tsv"
    if buffer_to.is_file() and not force:
        logging.info(f"Loading from file {buffer_to}")
        history = pandas.read_csv(buffer_to, sep="\t")
    else:
        logging.info(f"Loading from MLFlow {experiment_name} {exp_id}")
        # get full history
        history = get_metric_history(
            tracking_uri=tracking_uri,
            experiment_ids=[exp_id],
            metrics=[metric_column, validation_metric_column],
            convert_to_wide_format=True,
        ).reset_index()
        history.to_csv(buffer_to, sep="\t", index=False)

    # get parameters
    columns = [dataset_column, subset_column, init_column]
    runs: pandas.DataFrame = mlflow.search_runs(
        experiment_ids=[exp_id],
        run_view_type=ViewType.ACTIVE_ONLY,
    ).loc[:, ["run_id"] + columns]

    # merge
    df = runs.merge(right=history, how="inner", on="run_id")

    # get best configuration & epoch according to validation_metric
    data = []
    for _, info in df.groupby(by=[dataset_column, subset_column, init_column]):
        first = info.sort_values(by=validation_metric_column, ascending=False).head(n=1)
        data.append(first)
    columns += [metric_column]
    translation = dict(zip(columns, [
        "dataset",
        "subset",
        "init",
        "H@1",
    ]))
    df = pandas.concat(data).loc[:, columns].rename(columns=translation).copy()

    # save method
    df["method"] = method_name
    return df


def _get_runs_from_mlflow(
    experiment_name: str,
    method_name: str,
    dataset_column: str = "params.dataset",
    subset_column: str = "params.subset",
    init_column: str = "params.init",
    metric_column: str = "metrics.test.hits_at_1",
    filter_string: Optional[str] = None,
    drop_duplicates: bool = False,
) -> pandas.DataFrame:
    zero_shot_exp_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id
    columns = [dataset_column, subset_column, init_column, metric_column]
    translation = dict(zip(columns, [
        "dataset",
        "subset",
        "init",
        "H@1",
    ]))
    runs: pandas.DataFrame = mlflow.search_runs(
        experiment_ids=[zero_shot_exp_id],
        filter_string=filter_string,
        run_view_type=ViewType.ACTIVE_ONLY,
    ).loc[:, columns].rename(columns=translation).copy()
    runs["method"] = method_name
    if drop_duplicates:
        logging.warning("Non-deterministically dropping duplicates!")
        runs = runs.drop_duplicates(subset=["dataset", "subset", "init"], keep="last")
    return runs


def _normalize_boolean_(
    df: pandas.DataFrame,
    column: str,
    false_name: str = "no",
    true_name: str = "yes",
) -> None:
    """Normalize a boolean column in-place."""
    df[column] = df[column].apply(lambda x: true_name if x else false_name)


def _normalize_similarity(df: pandas.DataFrame) -> None:
    """Normalizes similarity by combining cls and transformation."""
    df["params.similarity"] = (df["params.similarity.cls"] + "_" + df["params.similarity.transformation"])

    # transformation only active if similarity in {l1, 2}
    unused_transformation_mask = ~(df["params.similarity"].str.startswith("l1_") | df["params.similarity"].str.startswith("l2_"))
    df.loc[unused_transformation_mask, "params.similarity"] = df.loc[unused_transformation_mask, "params.similarity"].apply(lambda v: v.replace("_bound_inverse", "").replace("_negative", ""))

    def _normalize(name):
        name = name.split("_")
        if len(name) == 1:
            return name[0]
        else:
            return f"{name[0]} ({' '.join(name[1:])})"

    df["params.similarity"] = df["params.similarity"].apply(_normalize)


def _normalize_embedding_norm(df: pandas.DataFrame) -> None:
    """Normalize embedding normalization by combining norm and norm-mode."""
    norm = df["params.model.embedding_norm"].apply(lambda v: v.rsplit(".", maxsplit=1)[-1])
    mode = df["params.model.embedding_norm_mode"].apply(lambda v: v.rsplit(".", maxsplit=1)[-1])
    df["params.embedding_norm"] = (norm + "_" + mode).str.replace("none_none", "none")
    _translate = dict(
        none="never",
        l2_initial="initial",
        l2_every_forward="always",
    )
    df["params.embedding_norm"] = df["params.embedding_norm"].apply(_translate.__getitem__)


def _normalize_subset(df: pandas.DataFrame, column: str, drop_suffix: str = "-15k-v2") -> None:
    """Normalize subset name in place."""

    def normalize_subset(name: str) -> str:
        return name.lower().replace('_', '-').replace(drop_suffix, "")

    df[column] = df[column].apply(normalize_subset)


def _rdgcn_ablation_table(
    output_root: pathlib.Path,
    force: bool = False,
    tracking_uri: str = "http://localhost:5000",
) -> None:
    """Generate ablation table."""
    filter_string = """params.data.dataset='openea'"""
    validation_key = "eval.validation.hits_at_1"
    test_key = "eval.test.hits_at_1"
    experiment_name = "rdgcn"

    output_root = output_root / "ablation"
    output_root.mkdir(exist_ok=True, parents=True)

    # Get results from HPO
    df = get_results_from_hpo(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        validation_metric_column=validation_key,
        smaller_is_better=False,
        additional_metrics=[test_key],
        buffer_root=output_root,
        force=force,
        filter_string=filter_string,
    )

    # normalize columns
    df["normalization"] = normalize_embedding_norm(
        df=df,
        embedding_norm_method_column="model.embedding_norm",
        embedding_norm_mode_column="model.embedding_norm_mode",
    )
    df["subset"] = normalize_subset(df=df, column="data.subset")
    df["similarity"] = normalize_similarity(
        df=df,
        cls_column_name="similarity.cls",
        transformation_column_name="similarity.transformation",
    )
    df["GCN layers"] = df["model.num_gcn_layers"]
    df["interaction layers"] = df["model.num_interactions"]
    df["trainable embeddings"] = normalize_boolean(df=df, column="model.trainable_node_embeddings")
    df["hard_negatives"] = df["training.sampler"] == "hard_negative"
    df["hard negatives"] = normalize_boolean(df=df, column="hard_negatives")

    ablation_df = ablation(
        data=df,
        ablation_parameter=[
            "normalization",
            "GCN layers",
            "interaction layers",
            "trainable embeddings",
            "similarity",
            "hard negatives",
        ],
        group_keys="subset",
        sort_by=validation_key,
        sort_ascending=False,
    )

    table = combine_tables(
        *(
            format_ablation_table(df=ablation_df, cell_value=metric, groups=["subset"], float_formatter=lambda x: f"{100 * x:2.2f}")
            for metric in ["eval.validation.hits_at_1", "eval.test.hits_at_1"]
        ),
    )
    with (output_root / "table.tex").open("w") as f:
        f.write(table.to_latex(escape=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("kgm").setLevel(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["results", "datasets", "ablation"], required=True)
    parser.add_argument("--output", default="./out")
    parser.add_argument("--force", default=False, action="store_true")
    parser.add_argument("--tracking_uri", default="http://localhost:5000")
    args = parser.parse_args()
    _output_root = pathlib.Path(args.output).expanduser().absolute()
    _output_root.mkdir(exist_ok=True, parents=True)
    if args.target == "results":
        _result_table(output_root=_output_root, force=args.force, tracking_uri=args.tracking_uri)
    elif args.target == "datasets":
        _dataset_table(output_root=_output_root)
    elif args.target == "ablation":
        _rdgcn_ablation_table(output_root=_output_root, force=args.force, tracking_uri=args.tracking_uri)
