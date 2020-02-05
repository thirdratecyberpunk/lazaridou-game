import matplotlib.pyplot as plt

def display_accuracy_graph(accuracy):
    """
    Function to display accuracy of agents over time as a graph

    Parameters
    ----------
    accuracy: array of accuracies
    """

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(accuracy)
    ax.set(title="Accuracy of agents",
            xlabel="Number of rounds",
            ylabel="Accuracy")
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
