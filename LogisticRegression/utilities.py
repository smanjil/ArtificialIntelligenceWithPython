
import numpy as np
import matplotlib.pyplot as plt


def visualize_classifier(classifier, X, y):
    # define minimum and maximum values for X and y
    # that will be used in the mesh grid
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    
    # define the step size to use in plotting the mesh grid
    mesh_step_size = 0.01
    
    # define the mesh grid of X and y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))
    
    # run the classifier on the mesh grid
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    
    # reshape the output array
    output = output.reshape(x_vals.shape)
    
    # create a plot
    plt.figure()
    
    # choose a color scheme for the plot
    plt.pcolormesh(x_vals, y_vals, output, cmap = plt.cm.gray)
    
    # overlay the training points on the plot
    plt.scatter(X[:, 0], X[:, 1], c = y, s = 75, edgecolors = 'black', linewidth = 1, cmap = plt.cm.Paired)
    
    # specify the boundaries of the plot
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    
    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))
    
    plt.show()