# coding=utf-8
"""Utility methods for MLFlow."""
import hashlib
import itertools
import logging
import math
import os
import pathlib
import platform
import tempfile
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import mlflow
import mlflow.entities
import pandas
import tqdm

from .common import to_dot

logger = logging.getLogger(name=__name__)


def connect_mlflow(
    tracking_uri='http://mlflow.dbs.ifi.lmu.de:5000',
    experiment_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> mlflow.ActiveRun:
    """
    Connect to MLFlow server.

    :param tracking_uri:
        The URI.
    :param experiment_name:
        An optional experiment name to set.
    :param run_id:
        An optional run_id to associate the run with.

    :return:
        The active MLFlow run.
    """
    # connect
    mlflow.set_tracking_uri(tracking_uri)

    # set experiment
    if experiment_name is not None:
        mlflow.set_experiment(experiment_name=experiment_name)

    # start run
    return mlflow.start_run(run_id=run_id)


def log_params_to_mlflow(
    config: Dict[str, Any],
) -> None:
    """Log parameters to MLFlow. Allows nested dictionaries."""
    nice_config = to_dot(config)
    # mlflow can only process 100 parameters at once
    keys = sorted(nice_config.keys())
    batch_size = 100
    for start in range(0, len(keys), batch_size):
        mlflow.log_params({k: nice_config[k] for k in keys[start:start + batch_size]})


def log_metrics_to_mlflow(
    metrics: Mapping[str, Any],
    step: Optional[int] = None,
    prefix: Optional[str] = None,
) -> None:
    """Log metrics to MLFlow. Allows nested dictionaries."""
    nice_metrics = to_dot(metrics, prefix=prefix)
    nice_metrics = {k: float(v) for k, v in nice_metrics.items()}
    mlflow.log_metrics(nice_metrics, step=step)


def query_mlflow(
    tracking_uri: str,
    experiment_id: str,
    params: Dict[str, Union[str, int, float]] = None,
    metrics: Dict[str, Union[str, int, float]] = None,
    tags: Dict[str, Union[str, int, float]] = None
) -> List[mlflow.entities.Run]:
    """Query MLFlow for runs with matching params, metrics and tags."""
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    # Construct query
    q_params = [f'params.{p} = "{v}"' for p, v in to_dot(params).items()] if params else []
    q_metrics = [f'metrics.{m} = "{v}"' for m, v in to_dot(metrics).items()] if metrics else []
    q_tags = [f'tags.{t} = "{v}"' for t, v in tags.items()] if tags else []
    query = ' and '.join([*q_params, *q_metrics, *q_tags])

    return client.search_runs(experiment_id, query)


def get_metric_history_for_runs(
    tracking_uri: str,
    metrics: Union[str, Collection[str]],
    runs: Union[str, Collection[str]],
) -> pandas.DataFrame:
    """
    Get metric history for selected runs.

    :param tracking_uri:
        The URI of the tracking server.
    :param metrics:
        The metrics.
    :param runs:
        The IDs of selected runs.

    :return:
         A dataframe with columns {'run_id', 'key', 'step', 'timestamp', 'value'}.
    """
    # normalize input
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(runs, str):
        runs = [runs]
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    data: List[Tuple[str, str, int, int, float]] = []
    task_list = sorted(itertools.product(metrics, runs))
    n_success = n_error = 0
    with tqdm.tqdm(task_list, unit='metric+task', unit_scale=True) as progress:
        for metric, run in progress:
            try:
                data.extend(
                    (run, measurement.key, measurement.step, measurement.timestamp, measurement.value)
                    for measurement in client.get_metric_history(run_id=run, key=metric)
                )
                n_success += 1
            except IOError as error:
                n_error += 1
                progress.write(f'[Error] {error.strerror}')
            progress.set_postfix(dict(success=n_success, error=n_error))
    return pandas.DataFrame(
        data=data,
        columns=['run_id', 'key', 'step', 'timestamp', 'value']
    )


def get_metric_history(
    tracking_uri: str,
    experiment_ids: Union[int, Collection[int]],
    metrics: Collection[str],
    runs: Optional[Collection[str]] = None,
    convert_to_wide_format: bool = False,
    filter_string: Optional[str] = "",
) -> pandas.DataFrame:
    """
    Get metric history data for experiment(s).

    :param tracking_uri:
        The URI of the tracking server.
    :param experiment_ids:
        The experiments ID(s).
    :param metrics:
        The name of the metrics to retrieve the history for.
    :param runs:
        An optional selection of runs via IDs. If None, get all.
    :param convert_to_wide_format:
        Whether to convert the dataframe from "long" to "wide" format.
    :param filter_string:
        Filter query string, defaults to searching all runs.

    :return:
        A dataframe of results.
    """
    # Normalize runs
    if runs is None:
        runs = get_all_runs_from_experiments(
            tracking_uri=tracking_uri,
            filter_string=filter_string,
            experiment_ids=experiment_ids,
        )
        logger.info(f'Retrieved {len(runs)} runs for experiment(s) {experiment_ids}.')
    df = get_metric_history_for_runs(tracking_uri=tracking_uri, metrics=metrics, runs=runs)
    if convert_to_wide_format:
        df = _convert_metric_history_long_to_wide(history_df=df)
    return df


def _convert_metric_history_long_to_wide(
    history_df: pandas.DataFrame,
) -> pandas.DataFrame:
    """
    Convert ta dataframe of metric history from "long" to "wide" format.

    :param history_df:
        The dataframe in long format.

    :return:
        The dataframe in wide format.
    """
    return history_df.pivot_table(
        index=['run_id', 'step'],
        values='value',
        columns=['key'],
    )


T = TypeVar('T')


def _get_run_information_from_experiments(
    *,
    projection: Callable[[mlflow.entities.Run], T],
    experiment_ids: Union[int, str, Collection[int], Collection[str]],
    selection: Optional[Callable[[mlflow.entities.Run], bool]] = None,
    filter_string: Optional[str] = "",
    tracking_uri: Optional[str] = None,
    client: Optional[mlflow.tracking.MlflowClient] = None,
) -> Collection[T]:
    """
    Collect information for all runs associated with an experiment ID.

    .. note ::
        Exactly one of `tracking_uri` or `client` has to be provided.

    :param projection:
        The projection from an MLFlow run to the desired information.
    :param experiment_ids:
        The experiment IDs.
    :param selection:
        A selection criterion for filtering runs.
    :param filter_string:
        Filter query string, defaults to searching all runs.
    :param tracking_uri:
        The Mlflow tracking URI.
    :param client:
        The Mlflow client.


    :return:
        A collection of information for each run.
    """
    if None not in {tracking_uri, client}:
        raise ValueError('Cannot provide tracking_uri and client.')
    if tracking_uri is not None:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    elif client is None:
        raise ValueError("You must either provide client or tracking_uri.")
    if selection is None:
        def selection(_run: mlflow.entities.Run) -> bool:
            """Keep all."""
            return True
    assert selection is not None

    runs: List[T] = []

    # support for paginated results
    continue_searching = True
    page_token = None

    while continue_searching:
        page_result_list = client.search_runs(
            experiment_ids=_normalize_experiment_ids(experiment_ids),
            filter_string=filter_string,
            page_token=page_token,
        )
        runs.extend(projection(run) for run in page_result_list if selection(run))
        page_token = page_result_list.token
        continue_searching = page_token is not None

    return runs


def _normalize_experiment_ids(experiment_ids: Union[int, str, Collection[int], Collection[str]]) -> List[str]:
    if isinstance(experiment_ids, (int, str)):
        experiment_ids = [experiment_ids]
    return list(map(str, experiment_ids))


def get_all_runs_from_experiments(
    *,
    experiment_ids: Union[int, Collection[int]],
    filter_string: Optional[str] = "",
    tracking_uri: Optional[str] = None,
    client: Optional[mlflow.tracking.MlflowClient] = None,
) -> Collection[str]:
    """
    Collect IDs for all runs associated with an experiment ID.

    .. note ::
        Exactly one of `tracking_uri` or `client` has to be provided.

    :param experiment_ids:
        The experiment IDs.
    :param filter_string:
        Filter query string, defaults to searching all runs.
    :param tracking_uri:
        The Mlflow tracking URI.
    :param client:
        The Mlflow client.


    :return:
        A collection of run IDs.
    """

    def _get_run_id(
        run: mlflow.entities.Run,
    ) -> str:
        """Extract the run UUID."""
        return run.info.run_uuid

    return _get_run_information_from_experiments(
        projection=_get_run_id,
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        tracking_uri=tracking_uri,
        client=client
    )


def get_params_from_experiments(
    *,
    experiment_ids: Union[int, Collection[int]],
    filter_string: str = "",
    selection: Optional[Callable[[mlflow.entities.Run], bool]] = None,
    tracking_uri: Optional[str] = None,
    client: Optional[mlflow.tracking.MlflowClient] = None,
) -> pandas.DataFrame:
    """
    Collect run parameters for all runs associated with an experiment ID.

    .. note ::
        Exactly one of `tracking_uri` or `client` has to be provided.

    :param experiment_ids:
        The experiment IDs.
    :param filter_string:
        A filter for runs.
    :param selection:
        A selection criterion for filtering runs.
    :param tracking_uri:
        The Mlflow tracking URI.
    :param client:
        The Mlflow client.


    :return:
        A dataframe with an index of `run_uuid`s and one column per parameter.
    """

    def _get_run_params(
        run: mlflow.entities.Run,
    ) -> Mapping[str, str]:
        """Extract the run parameters."""
        result = dict(run.data.params)
        result['run_id'] = run.info.run_uuid
        return result

    return pandas.DataFrame(data=_get_run_information_from_experiments(
        projection=_get_run_params,
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        selection=selection,
        tracking_uri=tracking_uri,
        client=client,
    )).set_index(keys='run_id')


def _sort_key(x: Mapping[str, Any]) -> str:
    return hashlib.md5((';'.join(f'{k}={x}' for k, v in x.items()) + ';' + str(platform.node()) + ';' + str(os.getenv('CUDA_VISIBLE_DEVICES', '?'))).encode()).hexdigest()


def run_experiments(
    search_list: List[Mapping[str, Any]],
    experiment: Callable[[Mapping[str, Any]], Tuple[Mapping[str, Any], int]],
    num_replicates: int = 1,
    break_on_error: bool = False,
) -> None:
    """
    Run experiments synchronized by MLFlow.

    :param search_list:
        The search list of parameters. Each entry corresponds to one experiment.
    :param experiment:
        The experiment as callable. Takes the dictionary of parameters as input, and produces a result dictionary as well as a final step.
    :param num_replicates:
        The number of replicates to run each experiment.
    :param break_on_error:
        Whether to break on the first error.
    """
    # randomize sort order to avoid collisions with multiple workers
    search_list = sorted(search_list, key=_sort_key)

    n_experiments = len(search_list)
    counter = {
        'error': 0,
        'success': 0,
        'skip': 0,
    }
    for run, params in enumerate(search_list * num_replicates):
        logger.info('================== Run %4d/%4d ==================', run, n_experiments * num_replicates)
        params = dict(**params)

        # Check, if run with current parameters already exists
        query = ' and '.join(list(map(lambda item: f"params.{item[0]} = '{str(item[1])}'", to_dot(params).items())))
        logger.info('Query: \n%s\n', query)

        run_hash = hashlib.md5(query.encode()).hexdigest()
        params['run_hash'] = run_hash
        logger.info('Hash: %s', run_hash)

        existing_runs = mlflow.search_runs(filter_string=f"params.run_hash = '{run_hash}'", run_view_type=mlflow.tracking.client.ViewType.ACTIVE_ONLY)
        if len(existing_runs) >= num_replicates:
            logger.info('Skipping existing run.')
            counter['skip'] += 1
            continue

        mlflow.start_run()

        params['environment'] = {
            'server': platform.node(),
        }

        # Log to MLFlow
        log_params_to_mlflow(params)
        log_metrics_to_mlflow({'finished': False}, step=0)

        # Run experiment
        try:
            final_evaluation, final_step = experiment(params)
            # Log to MLFlow
            log_metrics_to_mlflow(metrics=final_evaluation, step=final_step)
            log_metrics_to_mlflow({'finished': True}, step=final_step)
            counter['success'] += 1
        except Exception as e:  # pylint: disable=broad-except
            logger.error('Error occured.')
            logger.exception(e)
            log_metrics_to_mlflow(metrics={'error': 1})
            counter['error'] += 1
            if break_on_error:
                raise e

        mlflow.end_run()

    logger.info('Ran %d experiments.', counter)


def get_results_from_hpo(
    tracking_uri: str,
    experiment_name: Union[str, Collection[str]],
    validation_metric_column: str = "eval.validation.hits_at_1",
    smaller_is_better: bool = False,
    additional_metrics: Optional[Collection[str]] = None,
    buffer_root: Union[None, str, pathlib.Path] = None,
    force: bool = False,
    filter_string: str = "",
) -> pandas.DataFrame:
    """
    Get results from a HPO logged to MLFlow.

    :param tracking_uri:
        The tracking URI.
    :param experiment_name:
        The experiment_name.
    :param validation_metric_column:
        The validation metric column with which to perform early stopping for each run.
    :param smaller_is_better:
        Whether smaller is better for the validation metric.
    :param additional_metrics:
        Additional metrics.
    :param buffer_root:
        A directory where downloaded data will be buffered to as TSV.
    :param force:
        Whether to enforce re-downloading even if buffered files exist.
    :param filter_string:
        A string to use for filtering runs.

    :return:
        A dataframe with one line per run, all parameters, and columns for the validation metric and all selected metrics.
    """
    if not isinstance(experiment_name, str):
        return pandas.concat([
            get_results_from_hpo(
                tracking_uri=tracking_uri,
                experiment_name=single_experiment_name,
                validation_metric_column=validation_metric_column,
                smaller_is_better=smaller_is_better,
                additional_metrics=additional_metrics,
                buffer_root=buffer_root,
                force=force,
                filter_string=filter_string,
            )
            for single_experiment_name in experiment_name
        ], ignore_index=True)
    mlflow.set_tracking_uri(uri=tracking_uri)
    exp_id = mlflow.get_experiment_by_name(name=experiment_name)
    if exp_id is None:
        raise ValueError(f"{experiment_name} does not exist at MLFlow instance at {tracking_uri}")
    exp_id = exp_id.experiment_id
    logger.info(f"Resolved experiment \"{experiment_name}\": {tracking_uri}/#/experiments/{exp_id}")

    # normalize input
    if additional_metrics is None:
        additional_metrics = []
    # filter out duplicates
    additional_metrics = sorted(set(additional_metrics).difference([validation_metric_column]))
    prefix = "metrics."
    metric_names = [validation_metric_column] + additional_metrics
    metric_names = [
        metric_name[len(prefix):] if metric_name.startswith(prefix) else metric_name
        for metric_name in metric_names
    ]

    # normalize root
    buffer_root = _resolve_experiment_buffer(
        buffer_root=buffer_root,
        exp_id=exp_id,
        experiment_name=experiment_name,
        filter_string=filter_string,
    )

    # load experiment parameters
    params = buffered_load_parameters(
        tracking_uri=tracking_uri,
        exp_id=exp_id,
        buffer_root=buffer_root,
        force=force,
        filter_string=filter_string,
    )
    runs = params["run_id"].tolist()
    logger.info(f"Found {len(runs)} runs.")

    # load metric history
    metrics = buffered_load_metric_history(
        tracking_uri=tracking_uri,
        metric_names=metric_names,
        runs=runs,
        buffer_root=buffer_root,
        force=force,
    )

    # perform early stopping for each run
    metrics = early_stopping_from_metric_history(
        validation_metric=metric_names[0],
        metric_history=metrics,
        smaller_is_better=smaller_is_better,
    )

    # combine parameters and metrics for each run
    return params.merge(right=metrics, how="inner", on="run_id")


def _resolve_experiment_buffer(
    buffer_root: Union[None, str, pathlib.Path],
    exp_id: int,
    experiment_name: str,
    filter_string: str,
) -> pathlib.Path:
    """
    Resolve the buffer root for a given experiment and filter string.

    :param buffer_root:
        The buffer root.
    :param exp_id:
        The experiment ID.
    :param experiment_name:
        The experiment name.
    :param filter_string:
        A filter string.

    :return:
        The absolute path to an existing directory used for buffering results.
    """
    # normalize buffer root
    if buffer_root is None:
        buffer_root = pathlib.Path(tempfile.gettempdir(), "mlflow_buffer")
    else:
        buffer_root = pathlib.Path(buffer_root)
    # use filter string hash to avoid confusing data extracted with different filter strings
    filter_hash = hashlib.sha512(filter_string.encode(encoding="utf8")).hexdigest()[:8]
    # compose buffer directory
    buffer_root = buffer_root.joinpath(f"{exp_id}_{experiment_name}", filter_hash).expanduser().absolute()
    # ensure the directory exists
    buffer_root.mkdir(exist_ok=True, parents=True)
    return buffer_root


def early_stopping_from_metric_history(
    metric_history: pandas.DataFrame,
    validation_metric: str,
    smaller_is_better: bool,
) -> pandas.DataFrame:
    """
    Perform "post-mortem" early stopping given a dataframe of metric history.

    :param metric_history:
        The dataframe of metric history. Columns: {'run_id', 'key', 'step', 'timestamp', 'value'}.
    :param validation_metric:
        The name of the validation metric.
    :param smaller_is_better:
        Whether smaller validation metric values indicate better performance.

    :return: shape: (num_runs, 2 + num_metrics)
        A wide dataframe with columns 'run_id', 'step', and all metrics from the history dataframe.
    """
    # early stopping: get best epoch according to validation_metric
    # in rare cases of exactly equal values, choose smaller epoch
    best = pandas.concat([
        this_metrics.sort_values(
            by=["value", "step"],
            ascending=[smaller_is_better, True],
        ).head(n=1)[["run_id", "step"]]
        for _, this_metrics in metric_history.loc[metric_history["key"] == validation_metric].groupby(by="run_id")
    ])
    metric_history = best.merge(right=metric_history, how="inner", on=["run_id", "step"])
    return _convert_metric_history_long_to_wide(history_df=metric_history).reset_index()


def buffered_load_metric_history(
    tracking_uri: str,
    metric_names: Collection[str],
    runs: Collection[str],
    buffer_root: pathlib.Path,
    force: bool = False,
) -> pandas.DataFrame:
    """
    Load metric histories from MLflow with TSV-file buffering.

    .. note ::
        The buffering is done on a per-metric basis to re-use old buffers when requesting additional metrics.

    :param tracking_uri:
        The tracking URI of a running MLFlow server.
    :param metric_names:
        The names of the metrics of interest.
    :param runs:
        The run IDs for which the metric history shall be retrieved.
    :param buffer_root:
        The root directory for buffering. Must be an existing directory.
    :param force:
        Whether to enforce re-downloading even if buffered files exist.

    :return:
        A dataframe of metric histories in long format, columns: {'run_id', 'key', 'step', 'timestamp', 'value'}.
    """
    metrics = []
    logger.info(f"Loading metric history for {len(metric_names)} metrics.")
    for i, metric_name in enumerate(metric_names, start=1):
        logger.info(f"[{i:3}/{len(metric_names)}] Loading metrics: {metric_name}")
        metrics_path = buffer_root / f"metrics.{metric_name}.tsv"
        if metrics_path.is_file() and not force:
            logger.info(f"Loading from {metrics_path.as_uri()}")
            single_metric = pandas.read_csv(metrics_path, sep="\t")
        else:
            logger.info("Loading from MLFlow")
            single_metric = get_metric_history_for_runs(tracking_uri=tracking_uri, metrics=metric_name, runs=runs)
            single_metric.to_csv(metrics_path, sep="\t", index=False)
            logger.info(f"Saved to {metrics_path.as_uri()}")
        metrics.append(single_metric)
    metrics_df = pandas.concat(metrics)
    logger.info(f"Loaded in total {metrics_df.shape[0]:,} measurements.")
    return metrics_df


def _has_non_empty_metrics(
    run: mlflow.entities.Run,
) -> bool:
    """Return true if the run has at least one finite metric value."""
    metrics = run.data.metrics
    return len(metrics) > 0 and any(map(math.isfinite, metrics.values()))


def buffered_load_parameters(
    tracking_uri: str,
    exp_id: int,
    buffer_root: pathlib.Path,
    force: bool = False,
    filter_string: str = "",
) -> pandas.DataFrame:
    """
    Load parameters from MLflow with TSV-file buffering.

    :param tracking_uri:
        The tracking URI.
    :param exp_id:
        The experiment ID.
    :param buffer_root:
        The root directory for buffering. Must be an existing directory.
    :param force:
        Whether to enforce re-downloading.
    :param filter_string:
        A string to use for filtering runs.

    :return:
        A dataframe of parameters, one row per run. Contains at least the column "run_id".
    """
    # Load parameters
    params_path = buffer_root / "params.tsv"
    if params_path.is_file() and not force:
        logger.info(f"Loading parameters from {params_path.as_uri()}")
        return pandas.read_csv(params_path, sep="\t")

    logger.info(f"Loading parameters from MLFlow {exp_id}")
    params = get_params_from_experiments(
        experiment_ids=exp_id,
        filter_string=filter_string,
        selection=_has_non_empty_metrics,
        tracking_uri=tracking_uri,
    ).reset_index()
    params.to_csv(params_path, sep="\t", index=False)
    logger.info(f"Saved parameters to {params_path.as_uri()}")
    return params


def best_per_group(
    data: pandas.DataFrame,
    group_keys: Union[str, Sequence[str]],
    sort_by: Union[str, Sequence[str]],
    sort_ascending: Union[bool, Sequence[bool]],
    tie_breaker: Sequence[str] = ("step", "run_id"),
    tie_break_ascending: Sequence[bool] = (True, True),
) -> pandas.DataFrame:
    """
    Get the best run for each group.

    :param data:
        The run data.
    :param group_keys:
        The keys by which to group.
    :param sort_by:
        The sort criteria.
    :param sort_ascending:
        The sort order.
    :param tie_breaker:
        The tie breakers (applied if there are multiple runs with exactly same sort keys).
    :param tie_break_ascending:
        The sort order for tie breakers.

    :return:
        A dataframe with the groups as index.
    """
    # input normalization
    if isinstance(group_keys, str):
        group_keys = [group_keys]
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    if isinstance(sort_ascending, bool):
        sort_ascending = [sort_ascending] * len(sort_by)
    assert len(sort_ascending) == len(sort_by)
    # ensure everything is a list
    group_keys = list(group_keys)
    # combine sort criterion and tie breakers
    sort_by = list(sort_by) + list(tie_breaker)
    ascending = list(sort_ascending) + list(tie_break_ascending)
    return pandas.concat(
        [
            group.sort_values(by=sort_by, ascending=ascending).head(n=1)
            for key, group in data.groupby(by=group_keys)
        ]
    ).set_index(keys=group_keys)


def ablation(
    data: pandas.DataFrame,
    ablation_parameter: Union[str, Collection[str]],
    group_keys: Union[str, Sequence[str]],
    sort_by: Union[str, Sequence[str]],
    sort_ascending: Union[bool, Sequence[bool]],
    tie_breaker: Sequence[str] = ("step", "run_id"),
    tie_break_ascending: Sequence[bool] = (True, True),
    name_column_name: str = "parameter",
    value_column_name: str = "value"
) -> pandas.DataFrame:
    """
    Ablation study for one parameter.

    :param data:
        The run data.
    :param ablation_parameter:
        The ablation parameter.
    :param group_keys:
        The keys by which to group.
    :param sort_by:
        The keys by which to sort.
    :param sort_ascending:
        Whether to sort ascending for each sort key.
    :param tie_breaker:
        The keys used to tie break.
    :param tie_break_ascending:
        Whether to sort ascending for each tie breaker key.
    :param name_column_name:
        The column name for the additional column containing the ablation parameter name.
    :param value_column_name:
        The column name for the additional column containing the ablation parameter value.

    :return:
        A dataframe with two additional columns name_column_name and  value_column_name.
    """
    if not {name_column_name, value_column_name}.isdisjoint(data.columns):
        raise ValueError("Either name_column_name or value_column_name occur as column names in the data dataframe.")
    if not isinstance(ablation_parameter, str):
        return pandas.concat([
            ablation(
                data=data,
                ablation_parameter=this_ablation_parameter,
                group_keys=group_keys,
                sort_by=sort_by,
                sort_ascending=sort_ascending,
                tie_breaker=tie_breaker,
                tie_break_ascending=tie_break_ascending,
                name_column_name=name_column_name,
                value_column_name=value_column_name,
            )
            for this_ablation_parameter in ablation_parameter
        ])

    if len(data[ablation_parameter].unique()) > 10:
        logger.warning(f"Ablation of parameter={ablation_parameter} has more than 10 levels.")
    this_out = []
    for parameter_value, group in data.groupby(by=ablation_parameter):
        best = best_per_group(
            data=group,
            group_keys=group_keys,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            tie_breaker=tie_breaker,
            tie_break_ascending=tie_break_ascending,
        ).reset_index()
        best[value_column_name] = parameter_value
        best[name_column_name] = ablation_parameter
        this_out.append(best)
    return pandas.concat(this_out)
