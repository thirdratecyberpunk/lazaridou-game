import matplotlib.pyplot as plt
import numpy as np

def display_loss_graph(loss_rate):
    """
    Function to display loss rate over time as a graph

    Parameters
    ----------
    loss_rate
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    # for key, value in comm_succ.items():
        # ax.plot(value, label = key)
    ax.plot(loss_rate)
    ax.set(title="Loss",
            xlabel="Number of rounds",
            ylabel="Loss")
    # ax.set_ylim([0,100])
    # ax.set_yticks(np.arange(0,100,10))
    ax.legend(loc="lower right")
    plt.show()

def display_comm_succ_graph(comm_succ):
    """
    Function to display communication success rate of agents over time as a graph

    Parameters
    ----------
    comm_succ_rate: dictionary of communication success rate
    """

    fig = plt.figure()
    ax = fig.add_subplot()
    for key, value in comm_succ.items():
        ax.plot(value, label = key)
    ax.set(title="Communication success rate of agents (%)",
            xlabel="Number of rounds",
            ylabel="Communication success rate (%)")
    ax.set_ylim([0,100])
    ax.set_yticks(np.arange(0,100,10))
    ax.legend(loc="lower right")
    plt.show()

def plot_figures(figures, nrows = 1, ncols=1):
    """
    Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[ind], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(ind)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    plt.show()
