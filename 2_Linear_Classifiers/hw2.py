import numpy as np
import matplotlib.pyplot as plt

import numpy # import again
import matplotlib.pyplot # import again

import numpy.linalg
import numpy.random


def generate_data(Para1, Para2, seed=0):
    """Generate binary random data

    Para1, Para2: dict, {str:float} for each class,
      keys are mx (center on x axis), my (center on y axis),
               ux (sigma on x axis), ux (sigma on y axis),
               y (label for this class)
    seed: int, seed for NUMPy's random number generator. Not Python's random.

    """
    numpy.random.seed(seed)
    X1 = numpy.vstack((numpy.random.normal(Para1['mx'], Para1['ux'], Para1['N']),
        numpy.random.normal(Para1['my'], Para1['uy'], Para1['N'])))
    X2 = numpy.vstack((numpy.random.normal(Para2['mx'], Para2['ux'], Para2['N']),
        numpy.random.normal(Para2['my'], Para2['uy'], Para2['N'])))
    Y = numpy.hstack(( Para1['y']*numpy.ones(Para1['N']),
        Para2['y']*numpy.ones(Para2['N'])  ))
    X = numpy.hstack((X1, X2))
    X = numpy.transpose(X)
    return X, Y

def plot_mse(X, y, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented
    y: 1-D numpy array

    Examples
    -----------------
    >>> X,y = generate_data(\
            {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
            {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
            seed=10)
    >>> plot_mse(X, y, 'test1.png')
    array([-1.8650779 , -0.03934209,  2.91707992])
    >>> X,y = generate_data(\
            {'mx':1,'my':-2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
            {'mx':-1,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
            seed=10)
    >>> # print (X, y)
    >>> plot_mse(X, y, 'test2.png')
    array([ 0.93061084, -0.01833983,  0.01127093])
    """
    w = np.array([0,0,0]) # just a placeholder

    # your code here
    X1 = X[y == +1]
    X2 = X[y == -1]

    X_t = numpy.transpose(X)
    N = X_t.shape[1]
    X_t = numpy.vstack((X_t, numpy.ones(N))) # augment

    # calculate w
    compound = numpy.matmul(X_t, numpy.transpose(X_t)) #(X^T * X)
    all_but_y = numpy.matmul(numpy.linalg.inv(compound), X_t) # [(X^T *X)^-1]*X^T
    w = numpy.matmul(all_but_y, y)


    # hyperplane
    x_ticks = numpy.array([numpy.min(X[:,0]), numpy.max(X[:,0])])
    y_ticks = -1*(x_ticks * w[0] +w[2])/w[1]
    matplotlib.pyplot.plot(x_ticks, y_ticks)

    X1 = numpy.transpose(X1)
    X2 = numpy.transpose(X2)

    matplotlib.pyplot.plot(X1[0], X1[1], '.b')
    matplotlib.pyplot.plot(X2[0], X2[1], '.r')

    # limit the range of plot to the dataset only
    matplotlib.pyplot.xlim(numpy.min(X[:,0]), numpy.max(X[:,0]))
    matplotlib.pyplot.ylim(numpy.min(X[:,1]), numpy.max(X[:,1]))
    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close('all') # it is important to always clear the plot
    return w
def plot_fisher(X, y, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented
    y: 1-D numpy array

    Examples
    -----------------
    >>> X,y = generate_data(\
            {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
            {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
            seed=10)
    >>> plot_fisher(X, y, 'test3.png')
    array([-1.61707972, -0.0341108 ,  2.54419773])
    >>> X,y = generate_data(\
            {'mx':-1.5,'my':2, 'ux':0.1, 'uy':2, 'y':1, 'N':200}, \
            {'mx':2,'my':-4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
            seed=1)
    >>> plot_fisher(X, y, 'test4.png')
    array([-1.54593468,  0.00366625,  0.40890079])
    """
    # your code here

    w = np.array([0,0,0])

    # Calculate S1
    X1 = X[y == 1]

    m1 = numpy.mean(numpy.transpose(X1), axis=1).reshape(2,1) # Calculate mean

    x_1_t = numpy.transpose(X1) # Transpose x

    Xm1 = numpy.subtract(x_1_t, m1) # Subtract X from M1
    Xm1_T = numpy.transpose(Xm1) # Take transpose of x-m1

    s_1 = numpy.matmul(Xm1, Xm1_T) # (x1-m1)(x1-m1)T

    # Calculate S2
    X2 = X[y == -1]

    m2 = numpy.mean(numpy.transpose(X2), axis=1).reshape(2,1) # Calculate mean

    x_2_t = numpy.transpose(X2)

    Xm2 = numpy.subtract(x_2_t, m2)
    Xm2_T = numpy.transpose(Xm2)

    s_2 = numpy.matmul(Xm2, Xm2_T)

    # Calculate S_w
    s_w = numpy.add(s_1, s_2)
    s_inv = numpy.linalg.inv(s_w)

    mean_diff = m1-m2
    w = numpy.matmul(s_inv, mean_diff)
    w_t = numpy.transpose(w)
    mm = -0.5 * (m1 + m2)
    w_bias = np.matmul(w_t, mm) # our bias
    w = numpy.array([w_t[0][0], w_t[0][1], w_bias[0][0]])

    x_ticks = numpy.array([numpy.min(X[:,0]), numpy.max(X[:,0])])
    y_ticks = -1*(x_ticks * w[0] +w[2])/w[1]

    matplotlib.pyplot.plot(x_ticks, y_ticks)
    matplotlib.pyplot.plot(X1[0], X1[1], '.b')
    matplotlib.pyplot.plot(X2[0], X2[1], '.r')

    # limit the range of plot to the dataset only
    matplotlib.pyplot.xlim(numpy.min(X[:,0]), numpy.max(X[:,0]))
    matplotlib.pyplot.ylim(numpy.min(X[:,1]), numpy.max(X[:,1]))
    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close('all') # it is important to always clear the plot
    return w

if __name__ == "__main__":
    import doctest
    doctest.testmod()



