# coding=utf-8
"""Utility methods for MLFlow."""
import itertools
import logging
from typing import Any, Collection, Dict, List, Optional, Union

import mlflow
import mlflow.entities
import pandas
import tqdm
from mlflow.exceptions import MlflowException

from .common import to_dot

logger = logging.getLogger(name=__name__)


def connect_mlflow(
    tracking_uri='http://localhost:5000',
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
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: Optional[str] = None,
) -> None:
    """Log metrics to MLFlow. Allows nested dictionaries."""
    nice_metrics = to_dot(metrics, prefix=prefix)
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


def experiment_name_to_id(
    tracking_uri: str,
    experiment_id: int,
) -> str:
    """Convert an experiment name to experiment ID."""
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    return [exp.name for exp in client.list_experiments() if int(exp.experiment_id) == experiment_id][0]


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
    data = []
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
            except (ConnectionError, MlflowException) as error:
                n_error += 1
                progress.write(f'[Error] {error}')
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
            experiment_ids=experiment_ids
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
    # Normalize input
    if isinstance(experiment_ids, int):
        experiment_ids = [experiment_ids]
    if None not in {tracking_uri, client}:
        raise ValueError('Cannot provide tracking_uri and client.')
    if tracking_uri is not None:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    runs = []

    # support for paginated results
    continue_searching = True
    page_token = None

    while continue_searching:
        page_result_list = client.search_runs(
            experiment_ids=list(map(str, experiment_ids)),
            filter_string=filter_string,
            page_token=page_token
        )
        runs.extend(run.info.run_uuid for run in page_result_list)
        page_token = page_result_list.token
        continue_searching = page_token is not None

    return runs
