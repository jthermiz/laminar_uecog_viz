import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np 
import itertools


def plot_markers(ax, times, color='darksalmon'):
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


def axes_grid(rows, cols, figsize=(8, 8), xlabel=None, ylabel=None, plot_type=None, onelabel=True, numbers=None):
    """
    Create grid axes
    label only the bottom ones

    Parameters
    ----------
    rows (int): Number of rows
    cols (int): Number of columns
    figsize (class, optional):The default is (8, 8).
    xlabel (str, optional): Label for x-axis. The default is None.
    ylabel (str, optional): Label for y-axis. The default is None.
    onelabel : TYPE, optional
        DESCRIPTION. The default is True.
    numbers (list, optional): list of channels in correct order of device. The default is None.

    Raises
    ------
    ValueError: `major` must be one of 'row' or 'col'

    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows, cols)
    
    if plot_type is True:
        fig.suptitle('{} Across Channels'.format(plot_type), fontsize = 20, y = 1)
    
    fig.supylabel(ylabel, fontsize = 10)
    fig.supxlabel(xlabel, fontsize = 10)
    
    
    row_col = list(itertools.product(range(rows), range(cols)))

    if numbers is True:
        numbers = range(len(row_col))
    
    for i, (row, col) in enumerate(row_col):
        ax = plt.subplot(gs[row, col])
        plt.tight_layout()
        if numbers is not None:
            # list of ids
            # ax.text(0, 0, numbers[i], color='red', fontsize=8, transform=ax.transAxes)   
            ax.set_title("Channel {}".format(numbers[i]), fontsize = 10)
        if row == rows-1 and (not onelabel or col == 0):
            # Last row and first col (or onelabel == False)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        yield ax


def poly_axes(poly_channel_order, ncols=2, xlabel='Time (ms)', ylabel='mV', plot_type=None):
    """
    Default axes for the polytrode data, displaying in order by depth,
    returned in dataset order (so you can enumerate them and index directly into a datset)

    Parameters
    ----------
    poly_channel_order (list): List of channel order of laminar probe.
    ncols (int): Number of columns. The default is 2.
    xlabel (str, optional): The default is 'Time (ms)'.
    ylabel (str, optional): The default is 'mV'.

    """
    n_ch = len(poly_channel_order)
    nrows = int(np.ceil( float(n_ch) / float(ncols) ))
    # elec = reader.nwb.electrodes
    axes = np.array(
        list(
            axes_grid(
                nrows, ncols,
                xlabel=xlabel, ylabel=ylabel,
                figsize=(1,6), numbers=poly_channel_order, plot_type=plot_type,
            )
        )
    )
    # for ax in axes:
    #     ax.set_xticklabels([])
    #     ax.set_xticks([])
    #     ax.set_yticklabels([])
    #     ax.set_yticks([])

    return axes[np.argsort(poly_channel_order)]


def ecog_axes(ecog_channel_order, ncols=16, xlabel='Time (ms)', ylabel='mV', plot_type=None):
    """
    Default axes for the ECoG data, displaying in order of on the surface,
    returned in dataset order (so you can enumerate them and index directly into a datset)

    Parameters
    ----------
    ecog_channel_order (list): List of channel order of ECoG.
    ncols (int): Number of columns. The default is 16.
    xlabel (str, optional): The default is 'Time (ms)'.
    ylabel (str, optional): The default is 'mV'.
    
    """
    n_ch = len(ecog_channel_order)
    nrows = int(np.ceil( float(n_ch) / float(ncols) ))
    axes = np.array(
        list(
            axes_grid(
                nrows, ncols,
                xlabel=xlabel, ylabel=ylabel,
                figsize=(30,20), numbers=ecog_channel_order, plot_type=plot_type,
            )
        )
    )
    # for ax in axes:
    #     ax.set_xticklabels([])
    #     ax.set_xticks([])
    #     ax.set_yticklabels([])
    #     ax.set_yticks([])

    return axes[np.argsort(ecog_channel_order)]


def get_ch_trials_matrix(signal_data, marker_onsets, channel, pre_buf = 10000, post_buf = 10000):
    
    """
    Creates a trials matrix for one channel
    
    Parameters
    ----------
    signal_data (np.array): signal data (nsamples, nchannels).
    marker_onsets (list): List of trial sitmulus onsets in samples.
    channnel (int): Specific channel you want data for 
    pre_buf (int, optional): Number of samples to pull prior to baseline. Defaults to 10000.
    post_buf (int, optional): Number of samples to pull after. Defaults to 10000.
    
    Returns
    -------
    trials_mat (np.array): Trial matrix for one channel (samples, trials).
    """
    
    nsamples = post_buf + pre_buf
    ntrials = len(marker_onsets)
    trials_mat = np.empty((nsamples, ntrials))
    channel_data = signal_data[:, channel]
    
    for idx, marker in enumerate(marker_onsets):
        start_frame, end_frame = marker - pre_buf, marker + post_buf
        trials_mat[:, idx] = channel_data[int(start_frame):int(end_frame)]
    return trials_mat

def get_all_trials_matrices(signal_data, marker_onsets, channel_order, pre_buf = 10000, post_buf = 10000):
    """
    Python dictionary where the key is the channel and the value is the trial matrix for that channel
    now, instead of calling get_trials_matrix a multiple of times when we want to visualize, we can 
    iterate over the keys in the all_trials matrix 

    Parameters
    ----------
    signal_data (np.array): signal data (nsamples, nchannels).
    marker_onsets (list): List of trial sitmulus onsets in samples.
    channel_order (list): List of channel order on ECoG array or laminar probe.
    pre_buf (int, optional): Number of samples to pull prior to baseline. Defaults to 10000.
    post_buf (int, optional): Number of samples to pull after. Defaults to 10000.

    Returns
    -------
    trials_dict (dict): Trials dictionary that is the length of channel_order. Each channel key has its trial matrix (samples, trials).
    """
    
    all_trials = {}
    for i in np.arange(len(channel_order)):
        one_channel = get_ch_trials_matrix(signal_data, marker_onsets, i)
        all_trials[channel_order[i]] = one_channel
    trials_dict = all_trials
    return trials_dict


def channel_orderer(signal_data, channel_order):
    """Puts the wave data into the order of the channels
    Args: 
    data: signal data in timepoints x channels
    chs (list): the correct order of the channels"""
    shape_wanted = signal_data.shape
    new_data = np.empty((shape_wanted[0], shape_wanted[1]))
    
    for i in np.arange(shape_wanted[1]):
        new_data[:, i] = signal_data[:, (channel_order[i] - 1)]
    return new_data



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
    
    ## Using get_ch_trials_matrix as a template but feel free to change anything!!
    
    # nsamples = post_buf + pre_buf
    # ntrials = len(stim_start_times)
    # trials_mat = np.empty((nsamples, ntrials))
    # channel_data = signal_data[:, channel]
    
    # for idx, onset in enumerate(stim_start_times):
    #     start_frame, end_frame = onset - pre_buf, onset + post_buf
    #     trials_mat[:, idx] = channel_data[int(start_frame):int(end_frame)]
    # return trials_mat
    

def nwb_stim_t(trials_df, fs):
    
    df_s = trials_df[trials_df["sb"] == "s"]
    
    onsets = df_s.iloc[:, [0,2]]
    stim_markers = onsets['start_time'].to_list()
    stim_onsets = [int(x*fs) for x in stim_markers]
    stim_start_times = np.array(stim_onsets)
    
    offsets = df_s.iloc[:, [1,2]]
    stim_mrks = offsets['stop_time'].to_list()
    stim_offsets = [int(x*fs) for x in stim_mrks]
    stim_stop_times = np.array(stim_offsets)
    
    return stim_start_times, stim_stop_times


def nwb_baseline_t(trials_df, fs):

    df_b = trials_df[trials_df["sb"] == "b"]
    
    b_onsets = df_b.iloc[:, [0,2]]
    base_markers = b_onsets['start_time'].to_list()
    base_onsets = [int(x*fs) for x in base_markers]
    base_start_times = np.array(base_onsets)
    
    b_offsets = df_b.iloc[:, [1,2]]
    base_mrks = b_offsets['stop_time'].to_list()
    base_offsets = [int(x*fs) for x in base_mrks]
    base_stop_times = np.array(base_offsets)
    
    return base_start_times, base_stop_times
