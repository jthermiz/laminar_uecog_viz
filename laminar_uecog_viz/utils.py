import matplotlib.pyplot as plt


def plot_markers(ax, times, color='r'):
    """Plots vertical markers

    Parameters
    ----------
    ax : matplotlib.Axes.axes
        Axes handle
    times : array_like
        Times to plot vertical marker in seconds
    color : str, optional
        Color of markers, by default 'r'

    Returns
    -------
    matplotlib.Axes.axes
        Axes handle
    """

    y_min, y_max = ax.get_ylim()
    for time in times:
        ax.plot([time, time], [y_min, y_max], color + '--')
    return ax


def check_fig_ax(fig, ax):
    """Checks whether fig and ax are valid plot handles. If so, returns them
    otherwise returns a new fig and ax handle

    Parameters
    ----------
    fig : matplotlib.Figure or None
        Handle or None
    ax : matplotlib.Axes.axes or None
        Handle or None

    Returns
    -------
    tuple
        Figure and axes handles
    """
    if (ax is None) or (fig is None):
        fig, ax = plt.subplots(1, 1)
    return fig, ax


def get_trials(X, t, start_times, stop_times, baseline_start_times, baseline_stop_times):
    """Convert N-D array into N-D + 1 set of trial arrays

    Parameters
    ----------
    X : N-D array
        [description]
    t : 1-D array
        Array of time points in seconds
    start_times : array_like
        Trial start times in seconds
    stop_times : array_like
        Trial stop times in seconds
    baseline_start_times : array_like
        Start times for calculating baseline normalizing statistics in seconds
    baseline_stop_times : array_like
        Stop times for calculating baseline normalizing statistics in seconds

    Raises
    ------
    NotImplementedError
        [description]
    """
    raise NotImplementedError


def plot_trials(X, t, type='shaded_error', stat='median'):
    """Plots central statistic (eg median) across trials along with
    measure of error or single trials

    Parameters
    ----------
    X : 2-D array
        Trial matrix, samples by trials
    t : array_like
        Relative trial time
    type : str, optional
        shaded_error: plots shaded error
        single_trials: plots single trials, by default 'shaded_error'
    stat : str, optional
        median: plots median trial and computes error using MAD
        mean: plots mean trial and computes error using standard error
    """
    raise NotImplementedError
