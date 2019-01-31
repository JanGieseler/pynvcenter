import matplotlib.pyplot as plt

from pynvcenter import nv_analysis
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

magnet_parameters = {
    'particle_radius': 19,
    'nv_radius': 67,  # 73.670
    'theta_mag': 0,
    'phi_mag': 60,
    'dipole_height': 80,
    'shot_noise': 0,
    'linewidth': 1e7,
    'n_angle': 48,  # 45 48
    'n_freq': 448,  # 351
    'f_min': 2.62e9,  # , 2.695
    'f_max': 3.120e9,  # , 3.045
    'avrg_count_rate': 1,
    'xo': 0,
    'yo': 0
}

label_map = {'xo': 'x', 'yo': 'y', 'dipole_height': 'z', 'theta_mag': 't', 'phi_mag': 'p',
             'particle_radius':'a', 'nv_radius': 'r'}


def create_image(xo, yo, plot_img=False, particle_radius=20, nv_radius=70, theta_mag=0, phi_mag=45,
                 dipole_height=80, shot_noise=0, linewidth=1e7,
                 n_angle=60, n_freq=300,
                 f_min=2.65e9, f_max=3.15e9,
                 avrg_count_rate=100):
    """
    xo, yo center of the circle
    """

    signal = nv_analysis.esr_2D_map_ring_scan(nv_x=xo, nv_y=yo,
                                              particle_radius=particle_radius, nv_radius=nv_radius, theta_mag=theta_mag,
                                              phi_mag=phi_mag, dipole_height=dipole_height, shot_noise=shot_noise,
                                              linewidth=linewidth, n_angle=n_angle, n_freq=n_freq,
                                              f_min=f_min, f_max=f_max, avrg_count_rate=avrg_count_rate,
                                              return_data=True, show_plot=plot_img)

    return signal


def worker_function(parameters, pbar=None):
    img = create_image(**parameters)  # calculate the image
    if pbar:
        pbar.update()  # update the progress
    return img  # return the image


def generate_data(n_data, parameters=None, n_jobs=2):

    """




    :param n_data:
    :param parameters: parameters such as magnet_parameters as a dictionary, if single value, this value is fixed if tupple then first value is mean and second the range
    :param n_jobs:
    :param random_labels:
    :return:
    """
    max_displacement = 20  ## maximum offset from the center
    # positive and negative values
    # positions = pd.DataFrame(max_displacement * (np.random.random((n_data, 2)) - 0.5), columns=['xo', 'yo'])

    positions = None
    for k, v in parameters.items():
        if type(v) == tuple:
            if positions is None:
                positions = pd.DataFrame(v[0] + v[1] * (np.random.random(n_data) - 0.5), columns=[k])
            else:
                positions[k] = v[0] + v[1] * (np.random.random(n_data) - 0.5)

    assert positions is not None, 'at least one random variable required (one element of parameters should be a tupple)'

    # if 'xo' not in random_labels:
    #     del positions['xo']
    # if 'yo' not in random_labels:
    #     del positions['yo']
    # if 'dipole_height' in random_labels:
    #     positions['dipole_height'] = 20 * (np.random.random((n_data))) + 60
    # if 'theta_mag' in random_labels:
    #     positions['theta_mag'] = 90 * (np.random.random((n_data)))
    # if 'phi_mag' in random_labels:
    #     positions['phi_mag'] = 90 * (np.random.random((n_data)))
    # if 'particle_radius' in random_labels:
    #     positions['particle_radius'] = 2 * (np.random.random((n_data))) + 20
    # if 'nv_radius' in random_labels:
    #     positions['nv_radius'] = 3 * (np.random.random((n_data))) + 70

    X = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(worker_function)({**parameters, **positions.iloc[i].to_dict()})
        for i in tqdm(range(len(positions))))
    X = np.array(X, dtype=np.float16)

    Y = positions.values

    labels = positions.columns

    #     x_scaler = MinMaxScaler()
    #     # Xs = scaler.fit_transform(X.reshape(len(X), -1).astype(np.float32))
    #     x_scaler.fit(np.expand_dims(X.flatten().astype(np.float32), axis=1))  # flatten the array so that min_max scaling over full data and not for each feature (pixel)

    #     y_scaler = MinMaxScaler()
    #     y_scaler.fit(Y)

    #     {'X':X, 'Y' :Y, 'labels':labels, 'y_scaler':y_scaler, 'x_scaler':x_scaler}
    return {'X': X, 'Y': Y, 'labels': labels}


def esr_preprocessing(X):
    """

    process data as we get it from a measurement such that we can use it to fit our model

    esr has dips
    here we subtract each esr map from its mean to obtain maps that have peaks instead of dips

    X: matrix with dimensions (None, n_freq, n_angle) containing the esr data
    """

    x_shape = X.shape
    assert len(x_shape) == 3

    mean = np.mean(X.reshape(x_shape[0], -1), axis=1)

    return np.repeat(mean, np.product(x_shape[1:])).reshape(x_shape) - X


def get_x_scaler(X, option=0):
    """

    create a scaler for X
    option 1: normalize by min max of each feature, i.e. pixel
    option 2: normalize by min max of entire dataset

    """

    if option == 0:
        x_scaler = None
    elif option == 1:
        # option 1 (feature wise)
        x_shape = X.shape
        x_scaler = MinMaxScaler()
        x_scaler.fit(X.reshape(x_shape[0], -1).astype(np.float32))
    elif option == 2:
        # option 2 (global)
        x_scaler = MinMaxScaler()
        x_scaler.fit(np.expand_dims(X.flatten().astype(np.float32),
                                    axis=1))  # flatten the array so that min_max scaling over full data and not for each feature (pixel)

    return x_scaler


def split_and_scale(X, Y, x_scaler, y_scaler, test_size=0.1, option=2):
    """

    x_scaler: tip - use get_x_scaler() to create scaler object
    test_size: fraction for validation set
    option: option use to create scaler in get_x_scaler()
    """

    x_shape = X.shape

    if option == 0:
        Xs = X
    elif option == 1:
        # option 1 (feature wise)
        Xs = x_scaler.transform(X.reshape(x_shape[0], -1).astype(np.float32))
    elif option == 2:
        # option 2 (global)
        # flatten the array so that min_max scaling over full data and not for each feature (pixel)
        Xs = x_scaler.transform(np.expand_dims(X.flatten().astype(np.float32), axis=1))

    Xs = Xs.reshape(x_shape)
    Ys = y_scaler.transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(Xs, Ys, test_size=test_size, random_state=42)

    # add additional dimension since this is what the model expects (this is the "color channel")
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    return X_train, X_test, Y_train, Y_test


def analyze_fit(X, Y, model, labels,  magnet_parameters, n_plot=3, n_max=20, x_scaler=None, y_scaler=None):

    if x_scaler:
        x_shape = X.shape
        Xs = x_scaler.transform(X.reshape(x_shape[0], -1).astype(np.float32)).reshape(x_shape)
    else:
        Xs = X

    if y_scaler:
        Ys = y_scaler.transform(Y)
    else:
        Ys = Y

    if Xs.shape[-1] != 1:  # model expects a dimension for "color channels"
        Xs = np.expand_dims(Xs, -1)

    Y_predict = model.predict(Xs)

    #     for k, v in magnet_parameters.items():
    #         print(k, v)

    fig, ax = plt.subplots(1, 2, figsize=(8, 5))

    for i in range(len(Xs)):
        ax[0].plot([Ys[i, 0], Y_predict[i, 0]], [Ys[i, 1], Y_predict[i, 1]], 'go-', alpha=0.2)
        ax[0].set_xlabel('xo')
        ax[0].set_ylabel('yo')
    ax[0].scatter(Ys[:n_max, 0], Ys[:n_max, 1], marker='o')
    ax[0].scatter(Y_predict[:n_max, 0], Y_predict[:n_max, 1], marker='x')
    ax[0].set_title('scaled outputs')

    if y_scaler:
        Y_real = y_scaler.inverse_transform(Ys)
        Y_pred_real = y_scaler.inverse_transform(Y_predict)
    else:
        Y_real = Ys
        Y_pred_real = Y_predict

    ax[1].scatter(Y_real[0:n_max, 0], Y_real[0:n_max, 1], marker='o')
    ax[1].scatter(Y_pred_real[:, 0], Y_pred_real[:, 1], marker='x')
    ax[1].set_xlabel('xo')
    ax[1].set_ylabel('yo')
    ax[1].set_title('physical outputs')

    f_min = magnet_parameters['f_min']
    f_max = magnet_parameters['f_max']
    n_angle = magnet_parameters['n_angle']
    n_freq = magnet_parameters['n_freq']
    frequencies = np.linspace(f_min, f_max, n_freq)
    angle = np.linspace(0, 360, n_angle)

    if x_scaler:
        x_shape = Xs.shape[0:-1]
        X_real = x_scaler.inverse_transform(Xs.reshape(x_shape[0], -1)).reshape(x_shape)
    else:
        X_real = Xs

    for i in range(n_plot):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        img = create_image(*Y_pred_real[i, 0:2],
                           **{**magnet_parameters, **{k: v for k, v in zip(labels[2:], Y_pred_real[i, 2:])}})

        # pad generated image so that it has the same size as the one used for the model prediction
        if img.shape != (len(angles), len(frequencies)):
            img, angles, frequencies = pad_image(img, X_real.shape[-2:], angles=angles, frequencies=frequencies)
        else:
            img = pad_image(img, X_real.shape[-2:])


        ax[0].pcolor(frequencies, angle, np.squeeze(X_real[i]))
        ax[0].set_title('real\n' + ', '.join([label_map[k] + '={:0.2f}' for k in labels]).format(*Y_real[i]))
        # and create the image, construction in second argument constructs the updates parameter dictionary


        ax[1].pcolor(frequencies, angle, img)
        ax[1].set_title(
            'reconstructed\n' + ', '.join([label_map[k] + '={:0.2f}' for k in labels]).format(*Y_pred_real[i]))
        plt.tight_layout()


def pad_image(X, img_dims, angles=None, frequencies=None):
    """

    pads the image along the height dimension with periodic boundary conditions
    and crops the image symmetrically in the width dimensions



    :param X: array of shape (N, H, W) or (H, W)
    :param img_dims: target image dimensions (Hi, Wi), where Hi >=H and Wi<=W
    :return:
    """


    padding_dims = [d - s for s, d in zip(X.shape[-2:], img_dims)]

    print('typically we expect to cut off the image along the frequency dimension and pad it in the angle dimension')
    assert padding_dims[0] >= 0
    assert padding_dims[1] <= 0

    padding_dims = [[p // 2, p - p // 2] for p in
                    padding_dims]  # calculate the padding dims for the left/right, up/down
    padding_dims[1] = [max([-padding_dims[1][0] - 1, 0]),
                       X.shape[-1] + padding_dims[1][0]]  # for the freq we actually want the range
    X = np.concatenate([X[:, -padding_dims[0][0]:], X, X[:, 0:padding_dims[0][1]]], axis=1)
    X = X[:, :, padding_dims[1][0]:padding_dims[1][1]]

    if angles:
        angle = np.concatenate([angles[-padding_dims[0][0]:], angles, angles[0:padding_dims[0][1]]], axis=1)
        frequencies = frequencies[padding_dims[1][0]:padding_dims[1][1]]
        return X, angles, frequencies
    else:
        return X