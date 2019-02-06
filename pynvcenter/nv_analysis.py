# Here we collect code that is a bit higher level and helps to understand nv related measurements


from . import fields as f
from . import nv_optical_response as nv
from . import fields_plot as fp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt



def rotation_matrix_z(phi):
    """
    rotation matrix for a rotation about the z axis
    phi: rotation angle in degree
    """
    phi/=180/np.pi
    return np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0,0,1]
        ])

def rotation_matrix_x(phi):
    """
    rotation matrix for a rotation about the x axis
    phi: rotation angle in degree
    """
    phi/=180/np.pi
    return np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
        ])

def rotation_matrix_100_to_111(nv_id):
    """
    transforms the nv from the standard 100 type to the 111 type
    :param nv_id: id of NV that will be pointing up, i.e. along the z axis
    :return:
    """
    if nv_id ==0:
        theta_x, theta_z  = np.arctan(np.sqrt(2)), -np.pi/4
    elif nv_id ==1:
        theta_x, theta_z = -np.arctan(np.sqrt(2)), -np.pi / 4
    elif nv_id ==2:
        theta_x, theta_z = np.pi+np.arctan(np.sqrt(2)), -3*np.pi / 4
    elif nv_id ==3:
        theta_x, theta_z = np.pi-np.arctan(np.sqrt(2)), -3*np.pi / 4
    else:
        raise ValueError("wrong NV id, try values, 0,1,2,3")

    theta_x *= 180/np.pi
    theta_z *= 180 / np.pi

    return np.dot(rotation_matrix_x(theta_x), rotation_matrix_z(theta_z))

def get_full_nv_dataset(p, nv_id=1, n=[0, 0, 1], nv_rotation_matrix = None, wo=500e-9, gammaNV=28e9, Dgs=2.87, verbose=False):
    """
    returns a full dataset for Nv number nv_id, based on parameters p

    :param p:
    :param nv_id:
    :param n:
    :param nv_rotation_matrix: matrix that rotates the coordinate system of the nv center, e.g. to account for rotations of the diamond wrt the resonator
    :param wo:
    :param gammaNV:
    :param verbose:
    :return:
    """
    s = nv.nNV[nv_id - 1]  # NV orientation
    if nv_rotation_matrix is not None:
        assert np.shape(nv_rotation_matrix) == (3,3)
        s = np.dot(nv_rotation_matrix, s)
    # =============== calculate the gradients ==============

    df = f.calc_Gradient_single_dipole(p, s, n, verbose=verbose)
    # calculate gradent along x
    data = f.calc_Gradient_single_dipole(p, s, n, verbose=verbose)
    # calculate gradent along y
    data2 = f.calc_Gradient_single_dipole(p, s, n, verbose=verbose)
    # now calculate the avrg gradient in xy
    df['Gxy'] = np.sqrt(data['G'] ** 2 + data2['G'] ** 2)
    # calcualte the broadening
    df['Broadening'] = df['Gxy'] * gammaNV * wo  # the linewidth broadening due to a gradient in the xy-plane in MHz
    # calculate the magetic field
    data = f.calc_B_field_single_dipole(p, verbose=verbose)
    # df['Bx']
    # on-axis field
    df['Bpar'] = np.abs(np.dot(np.array(data[['Bx', 'By', 'Bz']]), np.array([s]).T))
    # off-axis field
    df['Bperp'] = np.linalg.norm(np.cross(np.array(data[['Bx', 'By', 'Bz']]), np.array([s])), axis=1)
    # total field
    df['Bmag'] = np.linalg.norm(np.array(data[['Bx', 'By', 'Bz']]), axis=1)

    # convert fields into NV frame
    BNV = nv.B_fields_in_NV_frame(np.array(data[['Bx', 'By', 'Bz']]), nv_id)
    esr_freq = nv.esr_frequencies(BNV, Dgs=Dgs)
    df['fm'] = esr_freq[:, 0]
    df['fp'] = esr_freq[:, 1]

    return df


def get_best_NV_position(df, max_broadening=100, max_off_axis_field=0.01, exclude_ring=0, verbose=False):
    """

    calculates the fields and gradients for parameters p
    and finds the position in space with the best conditions for the spin-mechanics experiment

    df: dataframe with all the fields and gradients calculated for Nv with nv_id

    max_broadening = 100 # max broadening in MHz
    max_off_axis_field = 0.01 # max off axis field in Teslas

    exclude_ring: requires that the position is outside a ring with radius exclude_ring

    verbose = True
    plot: if true plot the NV shift

    """

    if verbose:
        print(('Calculated fields and gradients at {:d} points'.format(len(df))))

    # =============== find the best position ==============
    # exclude the values within a ring of radius exclude_ring

    if exclude_ring>0:
        x = df.loc[(df['x']**2 + df['y']**2) >= exclude_ring**2]
    else:
        x = df
    if verbose:
        print(('Limited to xy within ring of radius {:0.2f}, {:d} points left'.format(exclude_ring, len(x))))

    # get the points where the xy gradient is less than the specified limit
    x = x.loc[(np.abs(x['Broadening']) < max_broadening)]
    if verbose:
        print(('Limited to xy inhomogeneous broadening < {:0.0f} MHz, {:d} points left'.format(max_broadening, len(x))))

    # get the points where the xy gradient is less than the specified limit
    x = x.loc[(np.abs(x['Bperp']) < max_off_axis_field)]
    if verbose:
        print(('Limited to off axis field < {:0.0f} mT, {:d} points left'.format(max_off_axis_field * 1e3, len(x))))



    # out of the subset get the point with the highest gradient
    x = x.loc[np.abs(x['G']) == np.max(np.abs(x['G']))]

    return x



def fit_ring(B, phi, sB, magnet_diam, radius, fix_theta_mag=False):
    """

    fit function sqrt(e_b\cdot eb), where eb is the direction of the dipole

    this is used to fit magentic fields measured on a ring

    p  angles on the ring

    dp = argv[0]  # dipole strength
    tm = argv[1]  # azimuthal angle of magnet
    pm = argv[2]  # polar angle of magnet

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
        init_guess = [np.max(B) / 2, to, np.pi / 2, 0]  # initial guess
        #         par, pcov = opt.curve_fit(fit_err_fun_ring_4, phi, B, init_guess, sigma = sB,  bounds=(0, [3., 2*np.pi, 2*np.pi, 2*np.pi]))
        par, pcov = opt.curve_fit(fit_err_fun_ring, phi, B, init_guess, sigma=sB,
                                  bounds=(0, [3., 2 * np.pi, 2 * np.pi, 2 * np.pi]))
        perr = np.sqrt(np.diag(pcov))  # fit error
        mag_moment, Br = nv.magnetic_moment_and_Br_from_fit(par[0], magnet_diam / 2., radius, mu0=4 * np.pi * 1e-7)

    return mag_moment, Br, par, perr, init_guess


def fit_err_fun_ring(p, *argv):
    """

    fit function sqrt(e_b\cdot eb), where eb is the direction of the dipole

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


def fit_ring2(B, phi, sB, magnet_diam, radius):
    """

    fit function sqrt(e_b\cdot eb), where eb is the direction of the dipole

    this is used to fit magentic fields measured on a ring

    phi  angles on the ring in deg

    dp = argv[0]  # dipole strength
    tm = argv[1]  # azimuthal angle of magnet
    pm = argv[2]  # polar angle of magnet

    """
    init_guess = [0, 90, 0.5]

    par, pcov = opt.curve_fit(fit_err_fun_ring2, [phi, magnet_diam, radius, 0], B, init_guess, sigma=sB,
                              bounds=(0, [180, 180, 3]))
    perr = np.sqrt(np.diag(pcov))  # fit error
    mag_moment = f.magnetic_moment(radius, par[2])

    return mag_moment, par[2], par, perr, init_guess

def fit_err_fun_ring2(p, *argv):

    """

    fit function to fit ring data using the field code

    this is used to fit magentic fields measured on a ring

    phi  angles on the ring in deg
    magn_diam magnet diameter in um
    radius_nvs radius at which data is taken
    dz distance between magnet and diamond

    Br = argv[2]  # surface field
    theta_m = argv[1]  # azimuthal angle of magnet  in deg
    phi_m = argv[0]  # polar angle of magnet in deg

    """
    phi = p[0]
    magn_diam = p[1]
    radius_nvs = p[2]
    dz = p[3]

    phi_m = argv[0]
    theta_m = argv[1]
    Br = argv[2]

    #     dz = 0 # distane between diamond and magnet in um
    #     radius_nvs = 3.2 # radius of NV measurements in um
    DipolePosition = np.array([0, 0, 0])
    #     phi_m = 15 # angle of magnetic dipole
    #     theta_m = 89 # angle of magnetic dipole

    muo = 4 * np.pi * 1e-7

    m = f.magnetic_moment(magn_diam / 2, Br, muo) * np.array(
        [np.cos(phi_m * np.pi / 180) * np.sin(theta_m * np.pi / 180),
         np.sin(phi_m * np.pi / 180) * np.sin(theta_m * np.pi / 180),
         np.cos(theta_m * np.pi / 180)])

    zo = magn_diam / 2. + dz
    # calculate the positions
    x = radius_nvs * np.cos(phi* np.pi / 180)
    y = radius_nvs * np.sin(phi* np.pi / 180)
    r = np.array([x, y, zo * np.ones(len(x))]).T

    B = f.b_field_single_dipole(r, DipolePosition, m)

    return np.linalg.norm(B, axis=1)


def calc_max_gradient(p, nv_id, n, max_broadening, max_off_axis_field, phi_diamond, theta_magnet, diamond111_nv_id = None, exclude_ring = 0, verbose = False):
    """
    calculates the maximum gradiend within the area defined by the parameter and angles

    p = {
    'tag':'bead_1',
    'a' : 1.4,
    'Br' : 0.31666357,
    'phi_m' : 0,
    'theta_m' : -np.arctan(np.sqrt(2))*180/np.pi,
    'mu_0' : 4 * np.pi * 1e-7,
    'd_bead_z': 0,
    'dx':0.05,
    'xmax':2
    }

    nv_id: number 0, 1, 2, or 3

    max_broadening: determines the maximum tolerated broadnening in MHz
    max_off_axis_field: determines the maximum tolerated off axis field in Teslas
    phi_diamond: polar (in plane) orientation of diamond wrt magnet
    theta_magnet: azimuthal (out of plane) orientation of magnet
    diamond111_nv_id: if not None, the id 0,1,2,3 specifies the NV that will be pointing along the z direction

    exclude_ring: requires that the position is outside a ring with radius exclude_ring

    """

    p['theta_m'] = theta_magnet

    nv_rot = rotation_matrix_z(phi_diamond)

    if diamond111_nv_id is not None:
        assert diamond111_nv_id in range(4)
        nv_rot = np.dot(nv_rot, rotation_matrix_100_to_111(diamond111_nv_id))

    df = get_full_nv_dataset(p, nv_id=nv_id, nv_rotation_matrix=nv_rot, n=n)

    x = get_best_NV_position(df, max_broadening=max_broadening, max_off_axis_field=max_off_axis_field, exclude_ring =exclude_ring)

    if len(x) == 0 and verbose:
        print('WARNING Gradient not found with current constraints. Run get_best_NV_position again...')
        x = get_best_NV_position(df, max_broadening=max_broadening, max_off_axis_field=max_off_axis_field,
                                 exclude_ring=exclude_ring, verbose=True)
    gradient = float(x['G'].iloc[0])

    return gradient


# def fill_in_missing_xy(data):
#     """
#     data is a pandas data set with colums 'x' and 'y'.
#     Assuming that the x and y values are on a grid will fill in the missing rows
#
#     :returns complete data set
#     """
#
#     Nx, Ny = len(np.unique(X)), len(np.unique(Y))
#
#     if len(data) < Nx * Ny:
#         # fill in missing elements
#
#         dx = int(np.min(
#             np.diff(np.unique(X))) * 1e5) * 1e-5  # we do this weird multiplication to chop of small rounding errors
#         dy = int(np.min(
#             np.diff(np.unique(Y))) * 1e5) * 1e-5  # we do this weird multiplication to chop of small rounding errors
#
#         xmin, xmax = np.min(X), np.max(Y)
#         ymin, ymax = np.min(Y), np.max(Y)
#
#         # these are all the x and y positons that we expect
#         Xo, Yo = np.meshgrid(np.arange(xmin, xmax, dx), np.arange(ymin, ymax, dy))
#
#         # fill in the rows where the data is missing (values are basically all NaN except for the position xy)
#         for xo, yo in zip(Xo.flatten(), Yo.flatten()):
#             if len(data[(data['x'] == xo) & (data['y'] == yo)]) == 0:
#                 data = data.append(pd.DataFrame.from_records({'x': [xo], 'y': [yo]}, index=[len(data)]))
#
#     return data



def esr_2D_map_ring_scan(particle_radius=30, nv_radius=70, nv_x=0, nv_y=0, theta_mag=0, phi_mag=45,
                    dipole_height=80, shot_noise=0, linewidth=1e7, n_angle=51, n_freq=501, f_min=2.65e9, f_max=3.15e9,
                    avrg_count_rate=1,MW_rabi=10,Dgs = 2.87,
                    return_data=False, show_plot=True, use_Pl=True):
    """
        simulates the data from a ring scan
        particle_radius: particle_radius in um
         dipole_height = height of the dipole in um

         use_Pl: if True we calculate the photoluminescence if false the contrast (Warning this is outdated!!)
    """
    nv_po = np.array([nv_x, nv_y])  # center of ring where we measure the nvs
    angle = np.linspace(0, 360, n_angle)

    frequencies = np.linspace(f_min, f_max, n_freq)

    Br = 0.1  # surface field of magnet in Tesla

    mu0 = 4 * np.pi * 1e-7  # T m /A

    dipole_strength = 4 * np.pi / 3 * (particle_radius) ** 3 / mu0

    # positions of NV centers
    nv_pos = np.array([nv_radius * np.cos(angle / 180 * np.pi) + nv_po[0],
                       nv_radius * np.sin(angle / 180 * np.pi) + nv_po[1]]).T

    # get physical units
    r = np.hstack([nv_pos, np.zeros([len(nv_pos), 1])])  # nvs are assumed to be in the z=0 plane
    DipolePosition = np.array(
        [0, 0, -dipole_height])  # position of dipole is at -dipole_position in z-direction and 0,0 in xy
    tm = np.pi / 180 * theta_mag
    pm = np.pi / 180 * phi_mag
    m = dipole_strength * np.array([np.cos(pm) * np.sin(tm), np.sin(pm) * np.sin(tm), np.cos(tm)])

    # calc field
    bfields = f.b_field_single_dipole(r, DipolePosition, m)


    rate_params = {
            'beta': 1,
            'kr' : 63.2,
            'k47' : 10.8,
            'k57' : 60.7,
            'k71' : 0.8,
            'k72' : 0.4,
            'Des' : 1.42
            }

    if use_Pl:
        signal = nv.signal_photoluminescence(frequencies, bfields, MW_rabi=MW_rabi, Dgs=Dgs,
                                   linewidth=linewidth, shot_noise=0, rate_params=rate_params)

    else:

        signal = nv.signal_contrast(frequencies, bfields, MW_rabi=MW_rabi, Dgs=Dgs, avrg_count_rate=avrg_count_rate,
                                   linewidth=linewidth, shot_noise=0)

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.pcolor(frequencies, angle, signal)

        ax.set_ylabel('angle (deg)')
        ax.set_xlabel('freq (Hz)')
        ax.set_title('theta = {:0.0f}, phi={:0.0f}'.format(theta_mag, phi_mag))
        plt.tight_layout()

    if return_data:
        return signal

if __name__ == '__main__':

    phi_diamond = 25
    for nv_id in range(4):
        nv_rot = rotation_matrix_z(phi_diamond)
        assert nv_id in range(4)
        nv_rot = np.dot(rotation_matrix_100_to_111(nv_id), nv_rot)

        print((np.dot(rotation_matrix_100_to_111(nv_id), nv.nNV[nv_id])))
        print(('---', np.dot(rotation_matrix_100_to_111(nv_id), nv.nNV[0])))


    print('---xxxx---------')
    phi_diamond = 0
    nv_id = 0
    nv_rot = rotation_matrix_z(phi_diamond)
    assert nv_id in range(4)
    nv_rot = np.dot(nv_rot, rotation_matrix_100_to_111(nv_id))

    for nv_id in range(4):

        print((np.dot(nv_rot, nv.nNV[nv_id])))


