from lib.experiment.evaluation.experiment.metrics import AugmentationMetricEvaluation, AugMetricLevel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lib.experiment.augmentation import VisualisationConfig
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind, mannwhitneyu
from typing import Tuple
from pandas import DataFrame
from scipy import stats


class AugmentationTuningVisualizationConfig:
    name: str
    metric: str
    metric_label: str
    label: str
    color: str
    boxplot_color: str
    percentage: bool
    plot_type: str
    subject_display: bool
    relative_to_baseline: bool
    baseline_parameter_value: float
    parameter_rounding: int
    metric_rounding: int

    def __init__(self,
                 name: str = "Augmentation Tuning",
                 metric: str = "accuracy",
                 metric_label: str = "Accuracy",
                 label: str = "Tuning Parameter",
                 color: str = "navy",
                 boxplot_color: str = "lavender",
                 percentage: bool = True,
                 plot_type: str = "boxplot",
                 subject_display: bool = True,
                 relative_to_baseline: bool = True,
                 baseline_parameter_value: float = 0,
                 parameter_rounding: int = 1,
                 metric_rounding: int = 4
                 ):
        self.name = name
        self.metric = metric
        self.metric_label = metric_label
        self.label = label
        self.color = color
        self.boxplot_color = boxplot_color
        self.percentage = percentage
        self.plot_type = plot_type
        self.subject_display = subject_display
        self.relative_to_baseline = relative_to_baseline
        self.baseline_parameter_value = baseline_parameter_value
        self.parameter_rounding = parameter_rounding
        self.metric_rounding = metric_rounding

    @staticmethod
    def from_visualization_config(visual_config: VisualisationConfig):
        return AugmentationTuningVisualizationConfig(
            name=visual_config.name,
            metric=visual_config.metric,
            metric_label=visual_config.metric_label,
            label=visual_config.label,
            color=visual_config.kwargs.get('color', 'navy'),
            boxplot_color=visual_config.kwargs.get('boxplot_color', 'lavender'),
            percentage=visual_config.kwargs.get('percentage', True),
            plot_type=visual_config.kwargs.get('plot_type', 'boxplot'),
            subject_display=visual_config.kwargs.get('subject_display', True),
            relative_to_baseline=visual_config.kwargs.get('relative_to_baseline', True),
            baseline_parameter_value=visual_config.kwargs.get('baseline_parameter_value', 0),
            parameter_rounding=visual_config.kwargs.get('parameter_rounding', 1),
            metric_rounding=visual_config.kwargs.get('metric_rounding', 4)
        )


def generate_tuning_augmentation_df(evaluation: AugmentationMetricEvaluation,
                                    relative: bool | None = None) -> pd.DataFrame:
    if evaluation.augmentation_subject is None:
        raise ValueError("Unable to generate augmentation visualization -> no augmentation tuning results found")

    tuning_results = evaluation.augmentation_subject.results
    info = evaluation.augmentation_subject.info
    assert info.visualisation_config is not None, "No visualisation config found in augmentation tuning results"
    assert tuning_results[0].metrics is not None, "No metrics found in augmentation tuning results"
    assert info.visualisation_config.metric in tuning_results[
        0].metrics, "Visualisation metric not found in augmentation tuning results"

    config = AugmentationTuningVisualizationConfig.from_visualization_config(info.visualisation_config)
    metric = config.metric
    datapoints = [(result.value, result.metrics[metric], result.subject) for result in tuning_results]
    unzipped_datapoints = list(zip(*datapoints))
    data_dict = {
        info.optimization_parameter: unzipped_datapoints[0],
        metric: unzipped_datapoints[1],
        'subject': unzipped_datapoints[2]
    }

    df = pd.DataFrame(data_dict)

    return df


def generate_generic_augmentation_df(evaluation: AugMetricLevel,
                                     relative: bool | None = None) -> pd.DataFrame:

    tuning_results = evaluation.results
    info = evaluation.info
    assert info.visualisation_config is not None, "No visualisation config found in augmentation tuning results"
    assert tuning_results[0].metrics is not None, "No metrics found in augmentation tuning results"
    assert info.visualisation_config.metric in tuning_results[
        0].metrics, "Visualisation metric not found in augmentation tuning results"

    config = AugmentationTuningVisualizationConfig.from_visualization_config(info.visualisation_config)
    metric = config.metric
    datapoints = [(result.value, result.metrics[metric], *result.attributes.values()) for result in tuning_results]
    unzipped_datapoints = list(zip(*datapoints))
    data_dict = {
        info.optimization_parameter: unzipped_datapoints[0],
        metric: unzipped_datapoints[1],
    }
    for i, attribute in enumerate(tuning_results[0].attributes.keys()):
        data_dict[attribute] = unzipped_datapoints[i + 2]

    df = pd.DataFrame(data_dict)

    return df


def get_y_label(config: AugmentationTuningVisualizationConfig) -> str:
    y_label = f'{config.metric_label} relative\nimprovement' if config.relative_to_baseline else config.metric_label
    if config.percentage:
        y_label += ' (%)'
    return y_label


def visualize_augmentation_result(
        evaluation: AugmentationMetricEvaluation,
) -> Tuple[plt.Figure, DataFrame]:
    df = generate_tuning_augmentation_df(evaluation)

    info = evaluation.augmentation_subject.info
    config = AugmentationTuningVisualizationConfig.from_visualization_config(info.visualisation_config)

    strategy_parameter = info.optimization_parameter
    metric = config.metric

    if config.relative_to_baseline:
        baseline = np.mean(df[df[info.optimization_parameter] == config.baseline_parameter_value][metric])
        df[metric] = (df[metric] / baseline - 1)

    mean_df = df.groupby(strategy_parameter).mean()
    argmax_mean = mean_df[metric].argmax()
    median_df = df.groupby(strategy_parameter).median()
    argmax_median = median_df[metric].argmax()
    subjects = df['subject'].unique()

    p_values = []
    if len(subjects) > 1:
        for value in df[strategy_parameter].unique():
            stat_test = mannwhitneyu(df[df[strategy_parameter] == value][metric],
                                     df[df[strategy_parameter] == config.baseline_parameter_value][metric],
                                     alternative='two-sided')
            p_values.append(np.round(stat_test.pvalue, 3))

    df[info.optimization_parameter] = df[info.optimization_parameter].round(config.parameter_rounding)
    df[metric] = df[metric].round(config.metric_rounding)

    mean = mean_df.iloc[argmax_mean][metric] * 100 if config.percentage else mean_df.iloc[argmax_mean][metric]
    median = median_df.iloc[argmax_median][metric] * 100 if config.percentage else median_df.iloc[argmax_median][metric]

    best_value = (mean_df.index[argmax_mean] * (mean_df.shape[0] - 1) / np.max(mean_df.index), mean) \
        if np.max(mean_df.index) > 0 else (0, mean)

    figure, ax = plt.subplots(1, 1, figsize=(16, 12))

    y_lim_offset = 0.01

    if config.percentage:
        df[metric] = df[metric] * 100
        y_lim_offset *= 100
    with sns.axes_style('whitegrid'):
        if config.plot_type == 'pointplot':
            sns.pointplot(data=df, x=strategy_parameter, y=metric, ax=ax, color=config.color)
            ax.plot(best_value[0], best_value[1], marker=".", markersize=18, color='red')
            ax.annotate(f"Best: {np.round(best_value[1], 2)}", best_value, textcoords="offset points",
                        xytext=(35, 10), fontsize=18, ha='center')
        else:
            sns.boxplot(data=df, x=strategy_parameter, y=metric, ax=ax, color=config.boxplot_color,
                        showmeans=True,
                        meanprops={"marker": "D",
                                   "markerfacecolor": "navy",
                                   "markersize": "10",
                                   "alpha": 0.5
                                   }
                        )

            if config.relative_to_baseline:
                ax.plot(0, 0, marker="D", markersize=10, color='navy', alpha=0.5)

            sns.stripplot(data=df, x=strategy_parameter, y=metric, ax=ax,
                          color=None if config.subject_display else 'black',
                          alpha=None if config.subject_display else 0.5,
                          hue='subject' if config.subject_display else None,
                          palette="flare", legend=config.subject_display)
            legend_elements = [Line2D([0], [0], color='black', lw=4, label=f'Best Median: {np.round(median, 2)}'),
                               Line2D([0], [0], marker='D', label=f'Best Average: {np.round(mean, 2)}',
                                      markerfacecolor='navy', markersize=10)]
            ax.legend().remove()

            if config.subject_display:
                legend2 = figure.legend(loc="outside lower left", bbox_to_anchor=(0, 0.04),
                                        ncol=len(df['subject'].unique()), prop={'size': 16})
                plt.gca().add_artist(legend2)

            ax.legend(handles=legend_elements, loc="upper right", ncol=2, prop={'size': 16})

        ax.set_xlabel(config.label, fontdict=dict(fontsize=18), labelpad=50)
        ax.set_ylabel(get_y_label(config), fontdict=dict(fontsize=18))
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)

        ax.set_ylim(df[metric].min() - y_lim_offset, df[metric].max() + y_lim_offset)

        for i, xtick in enumerate(ax.get_xticks()):
            if i < len(p_values):
                y_lim = ax.get_ylim()[0]
                diff = 0.05
                y = y_lim * (1 - diff) + ax.get_yticks()[len(ax.get_yticks()) - 2] * diff
                ax.text(xtick, y, p_values[i], size='x-small', color='k', weight='semibold')

        if config.relative_to_baseline:
            ax.axhline(y=0, xmin=0, xmax=1, ls='--', c='tab:red')

        ax.set_title(config.name, fontsize=22)
        sns.despine()
    plt.tight_layout()
    return figure, df


def visualize_averaged_augmentation_result(evaluation: AugmentationMetricEvaluation,
                                           baseline: float = 0.5) -> plt.Figure:
    df = generate_tuning_augmentation_df(evaluation)

    info = evaluation.augmentation_subject.info
    config = AugmentationTuningVisualizationConfig.from_visualization_config(info.visualisation_config)
    strategy_parameter = info.optimization_parameter
    metric = config.metric

    df = df.groupby(info.optimization_parameter)[metric].mean().reset_index()
    df[info.optimization_parameter] = df[info.optimization_parameter].round(config.parameter_rounding)
    df[metric] = df[metric].round(config.metric_rounding)

    max_value = df[metric].max()
    value_rounding = config.metric_rounding
    max_parameter = df[df[metric] == max_value][info.optimization_parameter].values[0]

    y_lim_offset = 0.01

    if config.percentage:
        df[metric] = df[metric] * 100
        y_lim_offset *= 100
        baseline *= 100
        max_value = max_value * 100
        value_rounding = 2

    figure, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(df[strategy_parameter], df[metric], marker='o', color=config.color, markersize=10)

    ax.set_ylim(baseline - y_lim_offset, df[metric].max() + y_lim_offset)
    ax.axhline(y=baseline, xmin=0, xmax=1, ls='--', c='tab:red')

    ax.set_title(config.name, fontsize=22)
    ax.set_xlabel(config.label, fontdict=dict(fontsize=18))
    ax.set_ylabel(get_y_label(config), fontdict=dict(fontsize=18))

    legend_elements = [Line2D([0], [0], color='black', lw=4,
                              label=f'Best Parameter: {np.round(max_parameter, config.parameter_rounding)}'),
                       Line2D([0], [0], marker='D', label=f'Best Value: {np.round(max_value, value_rounding)}',
                              markerfacecolor='navy', markersize=10)]
    ax.legend().remove()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.14)
    figure.legend(handles=legend_elements, loc="outside lower left", ncol=2, prop={'size': 16})
    return figure


def visualize_ensemble_augmentation_result(evaluation: AugMetricLevel,
                                           ci_dist: str | None = None,
                                           ci: float = 0.95) -> plt.Figure:
    df = generate_generic_augmentation_df(evaluation)

    info = evaluation.info
    config = AugmentationTuningVisualizationConfig.from_visualization_config(info.visualisation_config)
    strategy_parameter = info.optimization_parameter
    metric = config.metric

    df = df.groupby(by=['ensemble', info.optimization_parameter])[metric].mean().reset_index()
    df[info.optimization_parameter] = df[info.optimization_parameter].round(config.parameter_rounding)
    df[metric] = df[metric].round(config.metric_rounding)

    n = len(df['ensemble'].unique())

    if ci_dist is None:
        ci_dist = 'min_max' if n < 10 else 't' if n < 30 else 'z'

    if ci_dist == 'min_max':
        summary_df = df.groupby(strategy_parameter)[metric].agg(['mean', 'count', 'std', 'min', 'max']).reset_index()
        summary_df['ci95_hi'] = summary_df['max']
        summary_df['ci95_lo'] = summary_df['min']
    else:
        if ci_dist == 't':
            test_stat = stats.t.ppf((ci + 1) / 2, n - 1)
        else:
            test_stat = stats.norm.ppf((ci + 1) / 2)

        summary_df = df.groupby(strategy_parameter)[metric].agg(['mean', 'count', 'std']).reset_index()
        summary_df['ci95_hi'] = summary_df['mean'] + test_stat * summary_df['std'] / np.sqrt(summary_df['count'])
        summary_df['ci95_lo'] = summary_df['mean'] - test_stat * summary_df['std'] / np.sqrt(summary_df['count'])



    max_value = summary_df['mean'].max()
    value_rounding = config.metric_rounding
    max_parameter = summary_df[summary_df['mean'] == max_value][info.optimization_parameter].values[0]

    y_lim_offset = 0.01

    if config.percentage:
        summary_df['mean'] = summary_df['mean'] * 100
        summary_df['ci95_lo'] = summary_df['ci95_lo'] * 100
        summary_df['ci95_hi'] = summary_df['ci95_hi'] * 100
        y_lim_offset *= 100
        max_value = max_value * 100
        value_rounding = 2

    figure, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(summary_df[strategy_parameter], summary_df['mean'], marker='o', color=config.color, markersize=10)
    ax.fill_between(summary_df[strategy_parameter], summary_df['ci95_lo'], summary_df['ci95_hi'], color='b', alpha=0.2)

    ax.set_title(config.name, fontsize=22)
    ax.set_xlabel(config.label, fontdict=dict(fontsize=18))
    ax.set_ylabel(get_y_label(config), fontdict=dict(fontsize=18))

    legend_elements = [Line2D([0], [0], color='black', lw=4,
                              label=f'Best Parameter: {np.round(max_parameter, config.parameter_rounding)}'),
                       Line2D([0], [0], marker='D', label=f'Best Value: {np.round(max_value, value_rounding)}',
                              markerfacecolor='navy', markersize=10)]
    ax.legend().remove()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.14)
    figure.legend(handles=legend_elements, loc="outside lower left", ncol=2, prop={'size': 16})
    return figure


def visualize_subject_path_augmentation_result(evaluation: AugmentationMetricEvaluation,
                                               baseline: float = 0.5) -> plt.Figure:
    df = generate_tuning_augmentation_df(evaluation)

    info = evaluation.augmentation_subject.info
    config = AugmentationTuningVisualizationConfig.from_visualization_config(info.visualisation_config)
    strategy_parameter = info.optimization_parameter
    metric = config.metric

    figure, ax = plt.subplots(1, 1, figsize=(12, 9))

    y_lim_offset = 0.01

    if config.percentage:
        df[metric] = df[metric] * 100
        y_lim_offset *= 100
        baseline *= 100
    with sns.axes_style('whitegrid'):
        sns.lineplot(data=df, x=strategy_parameter, y=metric, hue='subject' if config.subject_display else None)

        ax.set_xlabel(config.label, fontdict=dict(fontsize=18))
        ax.set_ylabel(get_y_label(config), fontdict=dict(fontsize=18))
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)

        ax.set_ylim(baseline - y_lim_offset, df[metric].max() + y_lim_offset)
        ax.axhline(y=baseline, xmin=0, xmax=1, ls='--', c='tab:red')
        ax.legend().remove()

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.155)
        figure.legend(loc="outside lower left", ncol=len(df['subject'].unique()),
                      prop={'size': 16}, title='Subjects')

        ax.set_title(config.name, fontsize=22)
        sns.despine()
    return figure
