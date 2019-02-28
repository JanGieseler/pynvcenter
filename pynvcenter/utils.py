import numpy as np

def connect(x):
    """
    x: data that has sudden jumps because of sign changes, connect by reuqiring that the change in slope is minimized

    returns: smoothed x
    """

    x_connected = np.zeros(len(x))

    slope = None
    for i, _x in enumerate(x):

        if i > 0:
            slope = _x - _x_prev
        if i > 1:
            slope_flip = -_x - _x_prev
            if (slope - _slope_prev) ** 2 > (slope_flip - _slope_prev) ** 2:
                slope = slope_flip
                _x = -_x

        x_connected[i] = _x
        _x_prev = _x
        _slope_prev = slope

    return np.array(x_connected)


def connect_2(x, y):
    """
    connects the lines x and y such that the new lines are smoother

    returns: smoothed x and y
    """

    assert len(x) == len(y)
    x_connected = np.zeros([len(x), 2])

    slope = None
    flip = False
    for i, _x in enumerate(zip(x, y)):
        _x = np.array(_x)
        if flip is True:
            _x = _x[::-1]
        if i > 0:
            slope = _x - _x_prev
        if i > 1:
            slope_flip = _x[::-1] - _x_prev
            if np.sum((slope - _slope_prev) ** 2) > np.sum((slope_flip - _slope_prev) ** 2):
                slope = slope_flip
                _x = _x[::-1]
                flip = flip is False  # invert the value of flip

        x_connected[i] = _x
        _x_prev = _x
        _slope_prev = slope

    return x_connected[:, 0], x_connected[:, 1]

def connect_2_flip(x, y, flip_point):
    """

    flips the data in x and y at the index flip_point

    """
    return np.concatenate([x[0:flip_point], y[flip_point:]]), np.concatenate([y[0:flip_point], x[flip_point:]])
