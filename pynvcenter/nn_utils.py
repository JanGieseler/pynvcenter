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
                 avrg_count_rate=100,
                 MW_rabi = 10, Dgs=2.87, use_Pl=True, return_esr_freqs=False
                 ):
    """
    xo, yo center of the circle
    """

    signal = nv_analysis.esr_ring_scan_2D_map(nv_x=xo, nv_y=yo,
                                              particle_radius=particle_radius, nv_radius=nv_radius, theta_mag=theta_mag,
                                              phi_mag=phi_mag, dipole_height=dipole_height, shot_noise=shot_noise,
                                              linewidth=linewidth, n_angle=n_angle, n_freq=n_freq,
                                              f_min=f_min, f_max=f_max, avrg_count_rate=avrg_count_rate,
                                              MW_rabi=MW_rabi, Dgs=Dgs,
                                              return_data=True, show_plot=plot_img, use_Pl=use_Pl, return_esr_freqs=return_esr_freqs)

    if return_esr_freqs:
        signal, esr_freqs = signal
        return signal, esr_freqs
    else:
        return signal


def worker_function(parameters, pbar=None):
    img = create_image(**parameters)  # calculate the image
    if pbar:
        pbar.update()  # update the progress
    return img  # return the image


def generate_data(n_data, parameters=None, n_jobs=2, return_esr_freqs=False):

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
        if type(v) == tuple or type(v) == list:
            if positions is None:
                positions = pd.DataFrame(v[0] + v[1] * (np.random.random(n_data) - 0.5), columns=[k])
            else:
                positions[k] = v[0] + v[1] * (np.random.random(n_data) - 0.5)

    assert positions is not None, 'at least one random variable required (one element of parameters should be a tupple)'

    X = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(worker_function)({**parameters, **positions.iloc[i].to_dict(), 'return_esr_freqs':return_esr_freqs})
        for i in tqdm(range(len(positions))))

    if return_esr_freqs:
        Y_esr = np.array([elem[1] for elem in X])
        X = [elem[0] for elem in X]
    X = np.array(X, dtype=np.float16)

    Y = positions.values

    labels = positions.columns

    if return_esr_freqs:
        return {'X': X, 'Y': Y, 'labels': labels, 'Y_esr': Y_esr}
    else:
        return {'X': X, 'Y': Y, 'labels': labels}


def esr_preprocessing(X, reference_level=1):
    """

    process data as we get it from a measurement such that we can use it to fit our model

    esr has dips
    here we subtract each esr map from its mean to obtain maps that have peaks instead of dips

    X: matrix with dimensions (None, n_freq, n_angle) containing the esr data



    reference_level: level of background
        - int or float: data is normalized such that background counts are around one
        - mean: take the mean over the esr_map
        - mode: !!not implemented yet !!, find the mode of the esr_map (most frequent counts) and normalize to that
    """

    x_shape = X.shape
    assert len(x_shape) == 3

    if reference_level == 'mean':
        mean = np.mean(X.reshape(x_shape[0], -1), axis=1)
    elif type(reference_level) in (int, float):
        mean = reference_level * np.ones(x_shape[0])
    elif reference_level == 'min_max':
        mean = (np.max(X.reshape(x_shape[0], -1), axis=1) + np.min(X.reshape(x_shape[0], -1), axis=1))/2
    else:
        print('did not recognize datatype')
        raise TypeError

    val_range = np.max(X.reshape(x_shape[0], -1), axis=1) - np.min(X.reshape(x_shape[0], -1), axis=1)

    preprocessing_info = {'mean': mean}

    X = np.repeat(mean, np.product(x_shape[1:])).reshape(
        x_shape) - X  # flip and substract reference level (so that background is at zero)

    X = X / np.repeat(val_range, np.product(x_shape[1:])).reshape(x_shape)  # normalize image by the value range

    return X


class CustomScalerY():
    def __init__(self, magnet_parameters, labels):


        angle_labels = ['theta_mag', 'phi_mag']
        position_labels = ['particle_radius', 'nv_radius', 'dipole_height', 'xo', 'yo']
        frequency_labels = ['linewidth', 'MW_rabi', 'Dgs']

        norm_dict = {k: [p[0] - p[1] / 2, p[1]] for k, p in magnet_parameters.items() if
                     type(p) == list}  # build dictionary with min value and range of values

        max_range = max(np.array([v for k, v in norm_dict.items() if k in position_labels])[:,
                        1])  # this is the max range of the position values

        # for position values use max range instead of individual range
        for k, v in norm_dict.items():
            if k  in position_labels:
                norm_dict[k][1] = max_range

        self.norm_dict = norm_dict
        self.labels = labels

    def transform(self, Y, inplace=False):

        if inplace:
            # now normlize the Y values
            for k, v in self.norm_dict.items():
                column_id = [l == k for l in self.labels]
                Y[:, column_id] = (Y[:, column_id] - v[0]) / v[1]  # v[0]=min,  v[1]=range, i.e. normalize such that values are between 0 and 1

            return Y
        else:
            # now normalize the Y values
            return np.hstack([(Y[:, [l == k for l in self.labels]] - v[0]) / v[1] for k, v in self.norm_dict.items()])

    def inverse_transform(self, Y, inplace=True):

        if inplace:
            # now normlize the Y values
            for k, v in self.norm_dict.items():
                column_id = [l == k for l in self.labels]
                Y[:, column_id] = v[1] * Y[:, column_id] + v[
                    0]  # v[0]=min,  v[1]=range, i.e. normalize such that values are between 0 and 1

            return Y
        else:
            # now normlize the Y values
            return np.hstack([v[1] * Y[:, [l == k for l in self.labels]] + v[0] for k, v in self.norm_dict.items()])


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


def analyze_fit(X, Y, model, labels, magnet_parameters, labels_Y=None, n_plot=3, n_max=20, x_scaler=None, y_scaler=None,
                verbose=False):
    if labels_Y is None:
        labels_Y = labels
    if x_scaler:
        if verbose:
            print('rescaling X')
        x_shape = X.shape
        Xs = x_scaler.transform(X.reshape(x_shape[0], -1).astype(np.float32)).reshape(x_shape)
    else:
        Xs = X

    if y_scaler:
        if verbose:
            print('rescaling Y')
        Ys = y_scaler.transform(Y, inplace=False)
    else:
        Ys = Y

    if Xs.shape[-1] != 1:  # model expects a dimension for "color channels"
        if verbose:
            print('exapnd dims X')
        Xs = np.expand_dims(Xs, -1)

    Y_predict = model.predict(Xs)

    if verbose:
        print('Y_predict:')
        print(pd.DataFrame(Y_predict, columns=labels))

    fig, ax = plt.subplots(1, 2, figsize=(8, 5))

    for i in range(len(Xs)):
        ax[0].plot([Ys[i, labels_Y.index('xo')], Y_predict[i, labels_Y.index('xo')]],
                   [Ys[i, labels_Y.index('yo')], Y_predict[i, labels_Y.index('yo')]], 'go-', alpha=0.2)
        ax[0].set_xlabel('xo')
        ax[0].set_ylabel('yo')
    ax[0].scatter(Ys[:n_max, labels_Y.index('xo')], Ys[:n_max, labels_Y.index('yo')], marker='o')
    ax[0].scatter(Y_predict[:n_max, labels_Y.index('xo')], Y_predict[:n_max, labels_Y.index('yo')], marker='x')
    ax[0].set_title('scaled outputs')

    if y_scaler:
        Y_real = y_scaler.inverse_transform(Ys, inplace=False)
        Y_pred_real = y_scaler.inverse_transform(Y_predict, inplace=False)
    else:
        Y_real = Ys
        Y_pred_real = Y_predict

    ax[1].scatter(Y_real[0:n_max, labels_Y.index('xo')], Y_real[0:n_max, labels_Y.index('yo')], marker='o',
                  label='real')
    ax[1].scatter(Y_pred_real[:, labels_Y.index('xo')], Y_pred_real[:, labels_Y.index('yo')], marker='x', label='pred')
    ax[1].set_xlabel('xo')
    ax[1].set_ylabel('yo')
    ax[1].set_title('physical outputs')
    plt.legend()

    f_min = magnet_parameters['f_min']
    f_max = magnet_parameters['f_max']
    n_angle = magnet_parameters['n_angle']
    n_freq = magnet_parameters['n_freq']
    frequencies = np.linspace(f_min, f_max, n_freq)
    angles = np.linspace(0, 360, n_angle + 1)[0:-1]

    if x_scaler:
        x_shape = Xs.shape[0:-1]
        X_real = x_scaler.inverse_transform(Xs.reshape(x_shape[0], -1)).reshape(x_shape)
    else:
        X_real = Xs

    for i in range(n_plot):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        magnet_parameters_new = {**magnet_parameters, **{k: v for k, v in zip(labels_Y, Y_pred_real[i])}}
        # if magnet_parameters_new contains ranges we take the first value which corresponds to the center of the range
        # in the future we might provide an option to change this behaviour
        magnet_parameters_new = {**magnet_parameters_new,
                                 **{k: v[0] for k, v in magnet_parameters_new.items() if type(v) in (list, tuple)}}

        img_dim = X_real.shape[
                  1:3]  # get the dimensions of the image used for the model so that we can pad the generated image to the same dimensions

        if verbose:
            print('magnet_parameters_new', magnet_parameters_new)

        img = create_image(**magnet_parameters_new)

        if verbose:
            print('img_dim', img_dim, img.shape, (len(angles), len(frequencies)),
                  img_dim != (len(angles), len(frequencies)))

        # pad generated image so that it has the same size as the one used for the model prediction
        if img_dim != (len(angles), len(frequencies)):
            img, angles, frequencies = pad_image(img, img_dim, angles=angles, frequencies=frequencies)
        else:
            img = pad_image(img, img_dim)

        ax[0].pcolor(frequencies, angles, np.squeeze(X_real[i]))
        # ax[0].pcolor(np.squeeze(X_real[i]))
        ax[0].set_title('real\n' + ', '.join([label_map[k] + '={:0.2f}' for k in labels_Y]).format(*Y_real[i]))
        # and create the image, construction in second argument constructs the updates parameter dictionary

        ax[1].pcolor(frequencies, angles, img)
        ax[1].set_title(
            'reconstructed\n' + ', '.join([label_map[k] + '={:0.2f}' for k in labels_Y]).format(*Y_pred_real[i]))
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

    # print('typically we expect to cut off the image along the frequency dimension and pad it in the angle dimension')
    assert padding_dims[0] >= 0
    assert padding_dims[1] <= 0


    x_num_of_dims = len(X.shape)

    # if X of shape (H, W) expand to  (1, H, W)
    if x_num_of_dims ==2:
        X  = np.expand_dims(X, 0)

    padding_dims = [[p // 2, p - p // 2] for p in
                    padding_dims]  # calculate the padding dims for the left/right, up/down
    padding_dims[1] = [max([-padding_dims[1][0] - 1, 0]),
                       X.shape[-1] + padding_dims[1][0]]  # for the freq we actually want the range
    X = np.concatenate([X[:, -padding_dims[0][0]:], X, X[:, 0:padding_dims[0][1]]], axis=1)
    X = X[:, :, padding_dims[1][0]:padding_dims[1][1]]

    # if X of shape (H, W) undo  (1, H, W) back to (H, W)
    if x_num_of_dims == 2:
        X = np.squeeze(X, axis=0)

    if angles is not None:

        angles = np.hstack([angles[-padding_dims[0][0]:] - 360, angles, 360 + angles[0:padding_dims[0][1]]])
        frequencies = frequencies[padding_dims[1][0]:padding_dims[1][1]]
        return X, angles, frequencies
    else:
        return X