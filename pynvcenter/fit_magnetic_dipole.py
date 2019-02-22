
import numpy as np
from . import fields as f
from . import nv_optical_response as nv
import scipy.optimize as opt
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tqdm import tqdm

def esr_ring_scan_freqs(angle, particle_radius=30, nv_radius=70, nv_x=0, nv_y=0, theta_mag=0, phi_mag=45, phi_r=0,
                        dipole_height=80, Br = 0.1):
    """
    calcuate the esr frequencies form a magnetic dipole for NVs located on a ring at angles `angles`

    particle_radius: particle_radius in um
    dipole_height = height of the dipole in um
    Br: surface field of magnet in Tesla

    """

    # calc field in lab frame
    bfields = esr_ring_scan_fields(angle,
                                   particle_radius=particle_radius, nv_radius=nv_radius,
                                   nv_x=nv_x, nv_y=nv_y, theta_mag=theta_mag, phi_mag=phi_mag,
                                   phi_r=phi_r,dipole_height=dipole_height, Br=Br)

    esr_freqs = nv.esr_frequencies_ensemble(bfields)

    return esr_freqs

def esr_ring_scan_fields(angle, particle_radius=30, nv_radius=70, nv_x=0, nv_y=0, theta_mag=0, phi_mag=45, phi_r=0,
                        dipole_height=80, Br = 0.1):
    """
    calcuate the esr frequencies form a magnetic dipole for NVs located on a ring at angles `angles`

    particle_radius: particle_radius in um
    dipole_height = height of the dipole in um
    Br: surface field of magnet in Tesla

    """
    nv_po = np.array([nv_x, nv_y])  # center of ring where we measure the nvs



    mu0 = 4 * np.pi * 1e-7  # T m /A

    dipole_strength = Br * 4 * np.pi / 3 * (particle_radius) ** 3 / mu0

    _angle = angle+phi_r

    # positions of NV centers
    nv_pos = np.array([nv_radius * np.cos(_angle / 180 * np.pi) + nv_po[0],
                       nv_radius * np.sin(_angle / 180 * np.pi) + nv_po[1]]).T
    # get physical units
    r = np.hstack([nv_pos, np.zeros([len(nv_pos), 1])])  # nvs are assumed to be in the z=0 plane
    DipolePosition = np.array(
        [0, 0, -dipole_height])  # position of dipole is at -dipole_position in z-direction and 0,0 in xy


    tm = np.pi / 180 * theta_mag
    pm = np.pi / 180 * phi_mag

    m = dipole_strength * np.array([np.cos(pm) * np.sin(tm), np.sin(pm) * np.sin(tm), np.cos(tm)])

    # calc field in lab frame
    bfields = f.b_field_single_dipole(r, DipolePosition, m)


    return bfields


def fit_arc(angles, data, initial_guess, Do=2.87, verbose=False, transition=0, fit_parameter_keys=None, data_from_different_nvs=False):
    """

    fit  measured frequencies `data` to esr frequencies calculated form a magnetic dipole with parameters near `initial_guess`
    for NVs located on a ring at angles `angles`


    angles/data: a vector of length N or a list of vectors each of length Ni
    initial_guess: dictionary where values are either single values or a list / tupple of values
    in the former case the parameter is not constraint, in the latter, the first value is the mean and the second the range of the bound
    transition: 0 for lower NV transition (f_NV < D), 1 for upper NV transition (f_NV > D)

    """

    assert np.shape(angles) == np.shape(data)
    assert type(initial_guess) is dict

    if fit_parameter_keys is None:
        fit_parameter_keys = initial_guess.keys()

    fixed_parameter_keys = [k for k in initial_guess.keys() if k not in fit_parameter_keys]
    fixed_parameter_values = [v[0] if type(v) in (tuple, list) else v for k, v in initial_guess.items()
                  if k not in fit_parameter_keys]

    fixed_parameters = {k: v for k, v in zip(fixed_parameter_keys, fixed_parameter_values)}

    bounds = [(v[0] - v[1] / 2, v[0] + v[1] / 2) if type(v) in (tuple, list) else (None, None) for k, v in
              initial_guess.items() if k in fit_parameter_keys]

    param_init = [v[0] if type(v) in (tuple, list) else v for k, v in initial_guess.items()
                  if k in fit_parameter_keys]

    # figure out if we recieved just a single set or several datasets
    if len(np.shape(data)) == 1 and len(np.shape(data[0])) == 0:
        is_single = True
    else:
        is_single = False

    if verbose:
        print('is single:', is_single)

    # from now on we treat all the cases as a list of datasets
    if is_single:
        angles = [angles]
        data = [data]

    def loss(params):
        param_dict = {
            **fixed_parameters,
            **{k: v for k, v in zip(fit_parameter_keys, params)}
        }



        err, _, _ = loss_arcs(param_dict, angles, data, transition, data_from_different_nvs)
        # print('loss', np.sum(err))
        return np.sum(err)

    if verbose:
        print('param_init', param_init)
        print('bounds', bounds)
        print('fixed_parameters', fixed_parameters)

    result = minimize(loss, param_init, bounds=bounds)

    fit_result = {k: v for k, v in zip(fit_parameter_keys, result.x)}

    fit_result = {**fit_result, **fixed_parameters}
    return fit_result, result


def loss_arcs(params, angles, data, transition=0, data_from_different_nvs=False):
    """

    calculate the loss between esr frequencies calculated form a magnetic dipole with parameters `params`
    for NVs located on a ring at angles `angles` and measured frequencies `data`


    params: magnet parameters as dictionary
    angles: list of list / vector containing the anges of the arc
    data: list of list / vector containing the measured esr frequencies for each angle
    transition: 0 for lower NV transition (f_NV < D), 1 for upper NV transition (f_NV > D)
    data_from_different_nvs: if True each element in the data list should be from a different NV family


    returns:
        err: error each element in the angles/data lists
        freqs: the esr freq predicted by the model
        ids_of_detected_errs: the ids of the NV families corresponding to each element

    """

    for a, d in zip(angles, data):
        assert len(a) == len(d)

    # calculate the esr freq for current magnet parameters
    esr_freqs = [esr_ring_scan_freqs(_angles, **params)[:, :, transition] * 1e-9 for _angles in angles]

    # concat the data four times so that we can compare it with each ESR line
    esr_data = [np.tile(_data, 4).reshape(-1, len(_data)).T for _data in data]

    ids_of_detected_errs = []
    err = []
    freqs = []

    for _angles, _esr, _data in zip(angles, esr_freqs, esr_data):
        freq_diff = ((_data - _esr) ** 2)  # squared difference between the predicted esrs (_esr) and the data (y)

        # only use the lines  that we haven't used yet
        freq_diff = freq_diff
        _err = np.sum(freq_diff, axis=0)
        if data_from_different_nvs:
            assert len(ids_of_detected_errs)<4, 'There are only 4 different families, do not set data_from_different_nvs=True for more than 4 lines'
            # get the smallest error considering only the lines that have not been used yet
            idx_err = list(_err).index(_err[[i for i in range(4) if i not in ids_of_detected_errs]].min())
        else:
            idx_err = list(_err).index(_err.min())

        ids_of_detected_errs.append(idx_err)

        err.append(_err[idx_err])
        freqs.append(_esr[:, idx_err])

    # consistency check
    for a, f in zip(angles, freqs):
        assert len(a) == len(f)

    return err, freqs, ids_of_detected_errs




def fit_continuous_esr_lines(angles, data, initial_guess, verbose=False, fit_parameter_keys=None):
    """

    fit  measured frequencies `data` to esr frequencies calculated form a magnetic dipole with parameters near `initial_guess`
    for NVs located on a ring at angles `angles`


    angles: a vector of length N
    data: shape (N, 4, 2) in GHz
    initial_guess: dictionary where values are either single values or a list / tupple of values
    in the former case the parameter is not constraint, in the latter, the first value is the mean and the second the range of the bound
    transition: 0 for lower NV transition (f_NV < D), 1 for upper NV transition (f_NV > D)

    """

    assert type(initial_guess) is dict

    if fit_parameter_keys is None:
        fit_parameter_keys = initial_guess.keys()

    fixed_parameter_keys = [k for k in initial_guess.keys() if k not in fit_parameter_keys]
    fixed_parameter_values = [v[0] if type(v) in (tuple, list) else v for k, v in initial_guess.items()
                              if k not in fit_parameter_keys]

    fixed_parameters = {k: v for k, v in zip(fixed_parameter_keys, fixed_parameter_values)}

    bounds = [(v[0] - v[1] / 2, v[0] + v[1] / 2) if type(v) in (tuple, list) else (None, None) for k, v in
              initial_guess.items() if k in fit_parameter_keys]

    # param_init = [v[0] if type(v) in (tuple, list) else v for k, v in initial_guess.items()
    #               if k in fit_parameter_keys]

    param_init = {k: v for k, v in initial_guess.items() if k in fit_parameter_keys}
    fit_parameter_keys = param_init.keys()  # ensure that the order of the parameters matches the order of the keys
    param_init = [v[0] if type(v) in (tuple, list) else v for k, v in param_init.items()]


    #     X = sort_freq_pairs(data) # sort data such that freq pairs are ordered

    def loss(params):
        param_dict = {
            **fixed_parameters,
            **{k: v for k, v in zip(fit_parameter_keys, params)}
        }

        err = loss_continous_lines(param_dict, angles, data)

        return err

    if verbose:
        print('param_init', param_init)
        print('bounds', bounds)
        print('fixed_parameters', fixed_parameters)

    result = minimize(loss, param_init, bounds=bounds)

    fit_result = {k: v for k, v in zip(fit_parameter_keys, result.x)}

    fit_result = {**fit_result, **fixed_parameters}
    return fit_result, result


def loss_continous_lines(params, angles, data, plot_difference=False):
    """

    calculate the loss between esr frequencies calculated form a magnetic dipole with parameters `params`
    for NVs located on a ring at angles `angles` and measured frequencies `data`


    params: magnet parameters as dictionary
    angles: list of list / vector containing the anges of the arc
    data: list of list / vector containing the measured esr frequencies for each angle
    transition: 0 for lower NV transition (f_NV < D), 1 for upper NV transition (f_NV > D)
    data_from_different_nvs: if True each element in the data list should be from a different NV family


    returns:
        err: error each element in the angles/data lists
        freqs: the esr freq predicted by the model
        ids_of_detected_errs: the ids of the NV families corresponding to each element

    """

    assert list(np.shape(data)[1:]) == [4, 2]
    # calculate the esr freq for current magnet parameters
    esr_freqs = esr_ring_scan_freqs(angles, **params) * 1e-9  # shape (N, 4,2)

    freq_diff = ((data - esr_freqs) ** 2)
    err = np.sum(freq_diff)

    if plot_difference:
        for l1, l2 in zip(esr_freqs.reshape(-1, 8).T, data.reshape(-1, 8).T):
            plt.plot(angles, l1 - l2)
    #         plt.plot(angles, l2)

    return err


def loss_B_mag_ring(params, angles, data, plot_data=False):
    """

    loss function for fitting the total magnetig field on a ring to a dipole

    :param params:
    :param angles:
    :param data:
    :param plot_data:
    :return:
    """
    B_mag_full = np.linalg.norm(esr_ring_scan_fields(angles, **params), axis=1)
    err = np.sqrt(np.mean((data - B_mag_full) ** 2))

    if plot_data:
        plt.plot(angles, data)
        plt.plot(angles, B_mag_full)
    return err


def fit_B_mag_ring(angles, data, initial_guess, verbose=False, fit_parameter_keys=None):
    """

    fit  measured frequencies `data` to esr frequencies calculated form a magnetic dipole with parameters near `initial_guess`
    for NVs located on a ring at angles `angles`


    angles: a vector of length N
    data: shape (N, 4, 2) in GHz
    initial_guess: dictionary where values are either single values or a list / tupple of values
    in the former case the parameter is not constraint, in the latter, the first value is the mean and the second the range of the bound
    transition: 0 for lower NV transition (f_NV < D), 1 for upper NV transition (f_NV > D)

    """

    assert type(initial_guess) is dict

    if fit_parameter_keys is None:
        fit_parameter_keys = initial_guess.keys()

    fixed_parameter_keys = [k for k in initial_guess.keys() if k not in fit_parameter_keys]
    fixed_parameter_values = [v[0] if type(v) in (tuple, list) else v for k, v in initial_guess.items()
                              if k not in fit_parameter_keys]

    fixed_parameters = {k: v for k, v in zip(fixed_parameter_keys, fixed_parameter_values)}

    bounds = [(v[0] - v[1] / 2, v[0] + v[1] / 2) if type(v) in (tuple, list) else (None, None) for k, v in
              initial_guess.items() if k in fit_parameter_keys]

    param_init = {k: v for k, v in initial_guess.items() if k in fit_parameter_keys}
    fit_parameter_keys = param_init.keys()  # ensure that the order of the parameters matches the order of the keys
    param_init = [v[0] if type(v) in (tuple, list) else v for k, v in param_init.items()]

    def loss(params):
        param_dict = {
            **fixed_parameters,
            **{k: v for k, v in zip(fit_parameter_keys, params)}
        }

        err = loss_B_mag_ring(param_dict, angles, data)

        return err

    if verbose:
        print('param_init', param_init)
        print('bounds', bounds)
        print('fixed_parameters', fixed_parameters)

        param_dict = {
            **fixed_parameters,
            **{k: v for k, v in zip(fit_parameter_keys, param_init)}
            #             **{k: param_init[k] for k in fit_parameter_keys}
        }

    result = minimize(loss, param_init, bounds=bounds)

    fit_result = {k: v for k, v in zip(fit_parameter_keys, result.x)}

    fit_result = {**fit_result, **fixed_parameters}
    return fit_result, result


def plot_line_on_map(angles, frequencies, esr_map, lines):
    """


    lines shape (N, 4, 2) or (N, 8)
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.pcolor(frequencies, angles, esr_map)

    ax.set_ylabel('angle (deg)')
    ax.set_xlabel('freq (Hz)')
    # ax.set_title('theta = {:0.0f}, phi={:0.0f}'.format(theta_mag, phi_mag))
    plt.tight_layout()

    for line in lines.reshape(len(lines), -1).T:
        plt.plot(line, angles)
    plt.xlim([min(frequencies), max(frequencies)])
    plt.ylim([min(angles), max(angles)])


def fit_ring(B, phi, sB, magnet_diam, radius, fix_theta_mag=False):
    """


    fit function sqrt(e_b\cdot eb), where eb is the direction of the dipole

    this is used to fit magentic fields measured on a ring

    p  angles on the ring

    dp = argv[0]  # dipole strength
    tm = argv[1]  # azimuthal angle of magnet
    pm = argv[2]  # polar angle of magnet

    or

    dp = argv[0]  # dipole strength
    to = argv[1]  # azimuthal angle of ring
    tm = argv[2]  # azimuthal angle of magnet
    pm = argv[3]  # polar angle of magnet
    """

    if fix_theta_mag:
        to = np.arctan2(radius, magnet_diam / 2.)
        init_guess = [np.max(B) / 2, np.pi / 2, 0]  # initial guess
        #         par, pcov = opt.curve_fit(fit_err_fun_ring_3, [phi, to], B, init_guess, sigma = sB,  bounds=(0, [3., np.pi, np.pi]))

        par, pcov = opt.curve_fit(fit_err_fun_ring, [phi, to], B, init_guess, sigma=sB, bounds=(0, [3., np.pi, np.pi]))

        perr = np.sqrt(np.diag(pcov))  # fit error
        mag_moment, Br = nv.magnetic_moment_and_Br_from_fit(par[0], magnet_diam / 2., radius, mu0=4 * np.pi * 1e-7)

        # add the fixed value to to inital and final fit result, so that we always return 4 values
        init_guess = [init_guess[0], to, init_guess[1], init_guess[2]]
        par = [par[0], to, par[1], par[2]]

    else:
        to = np.arctan2(radius, magnet_diam / 2)
        to = np.arctan2(magnet_diam / 2, radius) # new
        #
        # print()
        init_guess = [np.max(B) / 2, to, np.pi / 2, 0]  # initial guess
        #         par, pcov = opt.curve_fit(fit_err_fun_ring_4, phi, B, init_guess, sigma = sB,  bounds=(0, [3., 2*np.pi, 2*np.pi, 2*np.pi]))
        par, pcov = opt.curve_fit(fit_err_fun_ring, phi, B, init_guess, sigma=sB,
                                  bounds=(0, [3., 2 * np.pi, 2 * np.pi, 2 * np.pi]))
        perr = np.sqrt(np.diag(pcov))  # fit error

        print('=====>', perr)
        mag_moment, Br = magnetic_moment_and_Br_from_fit(par[0], magnet_diam / 2., radius, mu0=4 * np.pi * 1e-7)

    return mag_moment, Br, par, perr, init_guess


def magnetic_moment_and_Br_from_fit(dp, a, r, mu0=4 * np.pi * 1e-7):
    """
    calculate the magentic moment and magnetic surface field from the fit parameter dp
    a: radius of magnet
    r: distance between NV circle and center of magnet
    """
    V = 4 * np.pi / 3 * a ** 3
    m = 4 * np.pi / mu0 * r ** 3 * dp
    Br = m / V * mu0

    Br = 4 * np.pi  * r ** 3 * dp


    return m, Br

def dipole_moment(Br, a, radius, zo):
    r = np.sqrt(zo**2+radius**2)
    dp = Br/3 * (a/(r))**3
    return dp

def fit_err_fun_ring(p, *argv):
    """


    fit function dp*sqrt(e_b\cdot eb), where eb is the direction of the dipole

    this is used to fit magentic fields measured on a ring

    p  angles on the ring

    dp = argv[0]  # dipole strength
    tm = argv[1]  # azimuthal angle of magnet
    pm = argv[2]  # polar angle of magnet

    """

    def f_ring(t, p, tm, pm=0):
        """
        angle dependency for magnetic field magnitude Squared!! on a ring
        the radial unit vector is defined as [cos(p)sin(t), sin(p)sin(t), cos(t)]
        t = azimuthal angle between 0 and pi
        p = polar angle between 0 and 2*pi
        tm = azimuthal angle between 0 and pi of magnet
        pm = polar angle between 0 and 2*pi of magnet
        """

        f = (34 + 6 * np.cos(2 * t) + 6 * np.cos(2 * tm)
             + 9 * np.cos(2 * (t - tm)) + 9 * np.cos(2 * (t + tm))
             + 24 * np.cos(2 * (p - pm)) * np.sin(t) ** 2 * np.sin(tm) ** 2
             + 24 * np.cos(p - pm) * np.sin(2 * t) * np.sin(2 * tm)) / 16

        return f

    if len(p) == 2:
        to = p[1]
        phi = p[0]

        dp = argv[0]  # dipole strength
        tm = argv[1]  # azimuthal angle of magnet
        pm = argv[2]  # polar angle of magnet
    else:
        phi = p
        dp = argv[0]  # dipole strength
        to = argv[1]  # azimuthal angle of ring
        tm = argv[2]  # azimuthal angle of magnet
        pm = argv[3]  # polar angle of magnet

    return dp * np.sqrt(f_ring(to, phi, tm, pm))


def fit_magnetic_field_ensemble(esr_data, initial_guess, Dgs=nv._Dgs):
    """
    esr_data: esr ensemble data in GHz shape (4, 2)
    initial_guess: initial guess for B field in cartesian coordinates (Lab frame) see also `propagate_initial_guess`

    future: leave _Dgs as free fit parameter
    """

    bounds = [(v[0] - v[1] / 2, v[0] + v[1] / 2) if type(v) in (tuple, list) else (None, None)
              for v in initial_guess]

    def loss(B):
        freq = 1e-9 * nv.esr_frequencies_ensemble(B, gs=27.969, muB=1, hbar=1, Dgs=Dgs).T
        err = (freq - esr_data) ** 2

        err = np.sum(err)
        return err

    #     result = minimize(loss, initial_guess, bounds=bounds)
    result = minimize(loss, initial_guess)

    return result


def get_magnetic_field_ensemble(data, initial_guess, vary_theta=range(0, 100, 15), propagate_initial_guess=True):
    """
    data: esr ensemble data in GHz shape (N, 4, 2)
    initial_guess: initial guess for B field in cartesian coordinates (Lab frame) see also `propagate_initial_guess`

    vary_theta: if not None is an array of theta values that is used
        in a grid search for the initial conditions
    propagate_initial_guess: if true we assume that the data is continious
        and we use the previous result as initial condition for the next data point


    returns:
        fit_B:  B field in cartesian coordinates (Lab frame)
        fit_esr: best match for esr freq, this should be close to data
        fit_err: mean square fit error for each data point
    """

    fit_esr = []
    fit_B = []
    fit_err = []
    for d in tqdm(data):
        Bmin = 1e9
        if vary_theta is not None:
            for t in vary_theta:
                B_mag, theta, phi = nv.B_spher(*initial_guess)
                theta += t
                Bc = nv.B_cart(B_mag, theta, phi)
                result = fit_magnetic_field_ensemble(d, Bc)

                if result.fun < Bmin:
                    Bmin = result.fun
                    final_result = result
        else:
            final_result = fit_magnetic_field_ensemble(d, initial_guess)
        if propagate_initial_guess:
            initial_guess = final_result.x
        fit_B.append(final_result.x)

        fit_esr.append(1e-9 * nv.esr_frequencies_ensemble(final_result.x, gs=27.969, muB=1, hbar=1, Dgs=2.87).T)
        fit_err.append(final_result.fun)
    fit_esr = np.array(fit_esr)
    fit_B = np.array(fit_B)

    return fit_B, fit_esr, fit_err



# def fit_ring2(B, phi, sB, magnet_diam, radius):
#     """
#
#     fit function sqrt(e_b\cdot eb), where eb is the direction of the dipole
#
#     this is used to fit magentic fields measured on a ring
#
#     phi  angles on the ring in deg
#
#     dp = argv[0]  # dipole strength
#     tm = argv[1]  # azimuthal angle of magnet
#     pm = argv[2]  # polar angle of magnet
#
#     """
#     init_guess = [0, 90, 0.5]
#
#     par, pcov = opt.curve_fit(fit_err_fun_ring2, [phi, magnet_diam, radius, 0], B, init_guess, sigma=sB,
#                               bounds=(0, [180, 180, 3]))
#     perr = np.sqrt(np.diag(pcov))  # fit error
#     mag_moment = f.magnetic_moment(radius, par[2])
#
#     return mag_moment, par[2], par, perr, init_guess
# #
# def fit_err_fun_ring2(p, *argv):
#
#     """
#
#     fit function to fit ring data using the field code
#
#     this is used to fit magentic fields measured on a ring
#
#     phi  angles on the ring in deg
#     magn_diam magnet diameter in um
#     radius_nvs radius at which data is taken
#     dz distance between magnet and diamond
#
#     Br = argv[2]  # surface field
#     theta_m = argv[1]  # azimuthal angle of magnet  in deg
#     phi_m = argv[0]  # polar angle of magnet in deg
#
#     """
#     phi = argv[0]
#     magn_diam = argv[1]
#     radius_nvs = argv[2]
#     dz = argv[3]
#
#     phi_m = argv[0]
#     theta_m = argv[1]
#     Br = argv[2]
#
#     #     dz = 0 # distane between diamond and magnet in um
#     #     radius_nvs = 3.2 # radius of NV measurements in um
#     DipolePosition = np.array([0, 0, 0])
#     #     phi_m = 15 # angle of magnetic dipole
#     #     theta_m = 89 # angle of magnetic dipole
#
#     muo = 4 * np.pi * 1e-7
#
#     m = f.magnetic_moment(magn_diam / 2, Br, muo) * np.array(
#         [np.cos(phi_m * np.pi / 180) * np.sin(theta_m * np.pi / 180),
#          np.sin(phi_m * np.pi / 180) * np.sin(theta_m * np.pi / 180),
#          np.cos(theta_m * np.pi / 180)])
#
#     zo = magn_diam / 2. + dz
#     # calculate the positions
#     x = radius_nvs * np.cos(phi* np.pi / 180)
#     y = radius_nvs * np.sin(phi* np.pi / 180)
#     r = np.array([x, y, zo * np.ones(len(x))]).T
#
#     B = f.b_field_single_dipole(r, DipolePosition, m)
#
#     return np.linalg.norm(B, axis=1)