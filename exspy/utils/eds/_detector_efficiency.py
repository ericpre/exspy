# Copyright 2007-2026 The eXSpy developers
#
# This file is part of eXSpy.
#
# eXSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# eXSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with eXSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

"""Detection efficiency of EDS detectors.

The functions in this module either calculate the detection efficiency of a
silicon drift detector from the absorption in the layers in front of the
detector (:func:`detector_efficiency_from_layers`) or load a tabulated
efficiency curve from a two-column file (:func:`load_detector_efficiency`).
The efficiency can be passed to :func:`exspy.utils.eds.get_k_factors` to
correct the standardless k-factors for quantification from raw intensities.
"""

import numpy as np

from exspy import material


def detector_efficiency_from_layers(
    energies, layers, detector_thickness, cutoff_energy=0.05
):
    """Compute the detection efficiency of an EDS detector from its layers.

    The efficiency is calculated by estimating the absorption of the X-rays
    in the layers in front of the detector (window, dead layer, ...) and in
    the active detector volume, assumed to be silicon::

        eps(E) = prod_i exp(-mac_i(E) * rho_i * t_i)
                 * (1 - exp(-mac_Si(E) * rho_Si * t_detector))

    where ``mac_i`` are the mass absorption coefficients, ``rho_i`` the
    densities and ``t_i`` the thicknesses of the layers.

    Parameters
    ----------
    energies : float or array of float
        The energy of the X-rays reaching the detector in keV.
    layers : sequence of tuple
        The layers in front of the detector, given as
        ``(element, thickness)`` pairs, with the thickness in nm. One
        element per layer; use several layers to approximate the
        composition of compound layers.
    detector_thickness : float
        The thickness of the active detector volume in mm.
    cutoff_energy : float, default 0.05
        The energy in keV below which the efficiency is set to zero.

    Returns
    -------
    efficiency : numpy.ndarray
        The detection efficiency, with the same shape as ``energies``
        (a 0-d array for a scalar input). An efficiency of 1 corresponds
        to a totally efficient detector.

    Notes
    -----
    The equation follows Alvisi et al. 2006, as implemented in HyperSpy
    pull request #2535.

    See Also
    --------
    load_detector_efficiency, exspy.utils.eds.get_k_factors

    """
    energies_arr = np.atleast_1d(np.asarray(energies, dtype=float))
    efficiency = np.ones_like(energies_arr)
    for element, thickness in layers:
        macs = material.mass_absorption_coefficient(
            element=element, energies=energies_arr
        )
        density = material._elements_dict[element]["Physical_properties"][
            "density (g/cm^3)"
        ]
        efficiency *= np.nan_to_num(np.exp(-(macs * density * thickness * 1e-7)))
    macs = material.mass_absorption_coefficient(element="Si", energies=energies_arr)
    density = material._elements_dict["Si"]["Physical_properties"]["density (g/cm^3)"]
    efficiency *= 1 - np.nan_to_num(
        np.exp(-(macs * density * detector_thickness * 1e-1))
    )
    efficiency[energies_arr < cutoff_energy] = 0.0
    return efficiency.reshape(np.shape(energies))


def load_detector_efficiency(filename):
    """Load a tabulated detection efficiency curve.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to a two-column text file: the first column the X-ray
        energy in keV and the second the detection efficiency.

    Returns
    -------
    energies : numpy.ndarray
        The energies in keV, sorted in increasing order.
    efficiencies : numpy.ndarray
        The detection efficiency at the corresponding energies.

    See Also
    --------
    detector_efficiency_from_layers, exspy.utils.eds.get_k_factors

    """
    data = np.loadtxt(filename, ndmin=2)
    if data.shape[1] != 2:
        raise ValueError(
            f"The file {str(filename)!r} must contain exactly two columns: "
            "the energy in keV and the detection efficiency."
        )
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]
