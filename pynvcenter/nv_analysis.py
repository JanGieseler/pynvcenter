# Here we collect code that is a bit higher level and helps to understand nv related measurements


from . import fields as f

from scipy.optimize import minimize
from . import nv_optical_response as nv
from . import fields_plot as fp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




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


def esr_ring_scan_2D_map(particle_radius=30, nv_radius=70, nv_x=0, nv_y=0, theta_mag=0, phi_mag=45, phi_r=0,
                         dipole_height=80, shot_noise=0, linewidth=1e7, n_angle=51, n_freq=501, f_min=2.65e9, f_max=3.15e9,
                         avrg_count_rate=1, MW_rabi=10, Dgs = 2.87,Br = 0.1,
                         return_data=False, show_plot=True, use_Pl=True, return_esr_freqs=False):
    """
        simulates the data from a ring scan
        particle_radius: particle_radius in um
        dipole_height = height of the dipole in um
        Br: surface field of magnet in Tesla

         use_Pl: if True we calculate the photoluminescence if false the contrast (Warning this is outdated!!)
    """
    nv_po = np.array([nv_x, nv_y])  # center of ring where we measure the nvs
    angle = phi_r + np.linspace(0, 360, n_angle+1)[0:-1]

    frequencies = np.linspace(f_min, f_max, n_freq)



    mu0 = 4 * np.pi * 1e-7  # T m /A

    dipole_strength = 4 * np.pi / 3 * (particle_radius) ** 3 / mu0*Br

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

    # calc field in lab frame
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
        if return_esr_freqs:
            esr_freqs = nv.esr_frequencies_ensemble(bfields)
            return signal, esr_freqs
        else:
            return signal

#     err_of_combination = []
#     for _combination in combinations(range(4),len(angles)):

#         print('>>>>>> combination',_combination)

#         XXerr = 0


#         print('>>>>>> err', XXerr)
# #                 err.append(freq_diff[i for i in range(4) if i not in detected_ids].min())
# #                 freqs.append(y[freq_diff.argmin()])
# err, freqs, ids_of_detected_esrs = loss_arcs(magnet_params, angles, x)


def sort_freq_pairs(esr_lines):
    """
    esr_lines: shape (None, 8)

    returns:
        sorted freq pairs so that the first one is always the lower value
         shape (None, 4,2)

    """

    sorter = np.argsort(esr_lines[0])  # order such that the highest goes with the lowest freq, the second highest with the second lowest and so on
    X = esr_lines[:, sorter]

    sorter = list(range(4)) + list(range(7, 3, -1))  # not we pair them up
    X = X[:, sorter].reshape(-1, 2, 4)
    X = np.moveaxis(X, 1, 2)
    X = np.sort(X)  # now sort the pairs

    return X


def concat_freq_pairs(esr_lines):
    """

    esr_lines: shape (N, 4, 2)

    returns: concatenated values of freq pairs shape (2*N, 4)


    """

    return np.concatenate((esr_lines[:, :, 0], esr_lines[:, :, 1]))


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


