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

"""Cliff-Lorimer k-factors from ``emtables`` X-ray cross-section tables.

The functions in this module compute X-ray quantification factors from
X-ray emission cross-section tables produced with the `emtables
<https://github.com/adriente/emtables>`_ package (see Teurtrie et al.,
Ultramicroscopy 2023). The tables store, for every element and X-ray line, the
emission cross-section (in cm^2) computed from first principles (ionisation
cross-section, fluorescence yield, radiative rates) at a given beam energy.

eXSpy ships the tables computed at 100, 200 and 300 keV (see the ``data``
directory). To use a different beam energy, generate a table with ``emtables``
and load it with :func:`load_cross_section_table`.

The k-factors returned by :func:`get_k_factors` are standardless theoretical
factors that do not account for detector efficiency; pass
``detector_efficiency`` to correct them for quantification from raw
intensities (see its documentation for details).
"""

from pathlib import Path

import numpy as np

from exspy import material
from exspy.utils.eds._detector_efficiency import load_detector_efficiency
from exspy.utils.eds._xray_lines import _get_element_and_line

_DATA_DIR = Path(__file__).parent / "data"

# Beam energies (keV) for which cross-section tables are bundled with eXSpy.
_BEAM_ENERGIES = (100, 200, 300)

# Mapping between the X-ray line names used by eXSpy (Siegbahn notation) and
# the IUPAC transitions stored in the emtables tables. This follows the same
# convention as ``material._elements_dict`` and :func:`get_xray_lines`; the
# ``Mz`` entry is not part of that database and follows the classic M-zeta
# doublet (M4-N2 and M5-N3).
SIEGBAHN_TO_IUPAC = {
    "Ka": ["KL3", "KL2"],
    "Kb": ["KM3", "KM2"],
    "La": ["L3M5", "L3M4"],
    "Lb1": ["L2M4"],
    "Lb2": ["L3N5"],
    "Lb3": ["L1M3"],
    "Lb4": ["L1M2"],
    "Lg1": ["L2N4"],
    "Lg3": ["L1N3"],
    "Ll": ["L3M1"],
    "Ln": ["L2M1"],
    "Ma": ["M5N7", "M5N6"],
    "Mb": ["M4N6"],
    "Mg": ["M3N5"],
    "Mz": ["M5N3", "M4N2"],
}


def load_cross_section_table(filename):
    """Load an ``emtables``-format X-ray emission cross-section table.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the JSON file produced by ``emtables`` (or shipped with
        ``espm``, e.g. ``300keV_xrays.json``).

    Returns
    -------
    table : dict
        Dictionary of cross-sections keyed by atomic number (as string) and
        then by IUPAC line name. Each entry is a dictionary with ``"energy"``
        (keV) and ``"cs"`` (cm^2) keys.
    metadata : dict
        The table metadata (beam energy, energy range, cross-section
        threshold, ...).

    See Also
    --------
    get_k_factors
    """
    import json

    with open(filename) as f:
        data = json.load(f)
    return data["table"], data["metadata"]


def _bundled_table(beam_energy):
    """Load the bundled table for the given beam energy (in keV)."""
    if beam_energy not in _BEAM_ENERGIES:
        requested = beam_energy
        try:
            requested = int(beam_energy)
        except (TypeError, ValueError):
            pass
        raise ValueError(
            f"No cross-section table is available for a beam energy of "
            f"{requested!r} keV. Available energies are "
            f"{_BEAM_ENERGIES}. Use ``emtables`` to generate a table for other "
            "energies and load it with ``load_cross_section_table``."
        )
    return load_cross_section_table(_DATA_DIR / f"{int(beam_energy)}keV_xrays.json")[0]


def _element_symbol(element):
    """Return the chemical symbol for an element given as symbol or Z."""
    if isinstance(element, str):
        if element in material._elements_dict:
            return element
        if element.isdigit() and int(element) in material.atomic_number_to_name:
            return material.atomic_number_to_name[int(element)]
        raise ValueError(f"Unknown element: {element!r}.")
    try:
        return material.atomic_number_to_name[int(element)]
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"Unknown atomic number: {element!r}.") from err


def _atomic_number(element):
    """Return the atomic number of an element given as symbol or Z."""
    return material._elements_dict[_element_symbol(element)]["General_properties"]["Z"]


def _atomic_weight(element):
    """Return the atomic weight of an element given as symbol or Z."""
    return material._elements_dict[_element_symbol(element)]["General_properties"][
        "atomic_weight"
    ]


def _resolve_lines(lines, line):
    """Resolve an eXSpy line name to the IUPAC lines present in ``lines``.

    ``line`` may be a Siegbahn name (``"Ka"``, ``"La"``, ``"Mz"``) or a raw
    IUPAC line name used directly by eXSpy for some M-shell lines (e.g.
    ``"M2N4"``).
    """
    if line in SIEGBAHN_TO_IUPAC:
        selected = [l for l in SIEGBAHN_TO_IUPAC[line] if l in lines]
    elif line in lines:
        selected = [line]
    else:
        raise ValueError(
            f"{line!r} is not a recognised eXSpy X-ray line. Use one of the "
            "eXSpy line names: " + ", ".join(SIEGBAHN_TO_IUPAC) + "."
        )

    if not selected:
        raise ValueError(
            f"The lines of the {line!r} family are not present in the "
            "cross-section table for this element."
        )
    return selected


def _line_cross_section(table, element, line):
    """Return the atomic number and summed cross-section for ``element``/``line``."""
    z = str(_atomic_number(element))
    if z not in table:
        raise ValueError(
            f"Element {_element_symbol(element)!r} (Z={z}) is not present in "
            "the cross-section table."
        )
    lines = table[z]
    names = _resolve_lines(lines, line)
    return z, sum(lines[name]["cs"] for name in names)


def _k_factor_from_sigma(
    table, element, sigma, reference_element, reference_line, form
):
    """Compute a k-factor from an element cross-section ``sigma``."""
    _, sigma_ref = _line_cross_section(table, reference_element, reference_line)

    if form == "cross_section":
        return sigma / sigma_ref
    if form == "atomic":
        return sigma_ref / sigma
    # weight form
    a_element = _atomic_weight(element)
    a_reference = _atomic_weight(reference_element)
    return (a_element / a_reference) * (sigma_ref / sigma)


def _get_k_factor(table, element, line, reference_element, reference_line, form):
    """Compute the k-factor of a single line from a loaded table."""
    _, sigma = _line_cross_section(table, element, line)
    return _k_factor_from_sigma(
        table, element, sigma, reference_element, reference_line, form
    )


def _combined_cross_section(table, group):
    """Return the common element and summed cross-section of a group of lines."""
    element = _get_element_and_line(group[0])[0]
    sigma = 0.0
    for xray_line in group:
        elt, line = _get_element_and_line(xray_line)
        if elt != element:
            raise ValueError(
                "All lines of a combined group must belong to the same element."
            )
        _, cross_section = _line_cross_section(table, elt, line)
        sigma += cross_section
    return element, sigma


def _as_efficiency_callable(detector_efficiency):
    """Normalise a detector-efficiency input to a callable ``eps(E)``.

    Accepts a callable, a two-column (energy, efficiency) array-like or
    the path to a two-column file, as accepted by :func:`get_k_factors`.
    """
    if callable(detector_efficiency):
        return lambda E: float(detector_efficiency(E))
    if isinstance(detector_efficiency, (str, Path)):
        energies, efficiencies = load_detector_efficiency(detector_efficiency)
    else:
        data = np.asarray(detector_efficiency, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(
                "The detector efficiency must be given as a callable, a "
                "two-column (energy in keV, efficiency) array or the path "
                "to a two-column file."
            )
        energies, efficiencies = data[:, 0], data[:, 1]
    order = np.argsort(energies)
    energies, efficiencies = energies[order], efficiencies[order]

    def eps(E):
        if E < energies[0] or E > energies[-1]:
            raise ValueError(
                "The detector efficiency curve does not cover an energy "
                f"of {E:.3f} keV; it spans energies from {energies[0]} to "
                f"{energies[-1]} keV."
            )
        return float(np.interp(E, energies, efficiencies))

    return eps


def _line_efficiency(table, element, line, eps):
    """Return the cross-section-weighted detection efficiency of a line."""
    lines = table[str(_atomic_number(element))]
    names = _resolve_lines(lines, line)
    total = sum(lines[name]["cs"] for name in names)
    weighted = sum(lines[name]["cs"] * eps(lines[name]["energy"]) for name in names)
    return weighted / total


def _group_efficiency(table, group, eps):
    """Return the cross-section-weighted detection efficiency of a group."""
    sigma_total = 0.0
    sigma_weighted = 0.0
    for xray_line in group:
        element, line = _get_element_and_line(xray_line)
        _, sigma = _line_cross_section(table, element, line)
        sigma_total += sigma
        sigma_weighted += sigma * _line_efficiency(table, element, line, eps)
    return sigma_weighted / sigma_total


def _checked_efficiency(eps_value, label):
    """Raise a clear error for lines the detector cannot detect."""
    if eps_value <= 0:
        raise ValueError(
            f"The detection efficiency of {label} is zero; this line cannot "
            "be quantified with this detector."
        )
    return eps_value


def get_k_factors(
    beam_energy,
    xray_lines,
    reference_element="Si",
    reference_line="Ka",
    form="weight",
    detector_efficiency=None,
):
    """Return k-factors for a list of X-ray lines, in the same order.

    Parameters
    ----------
    beam_energy : int or float
        The beam energy in keV. eXSpy ships cross-section tables for 100, 200
        and 300 keV; use ``emtables`` to generate a table for another energy
        and load it with :func:`load_cross_section_table`.
    xray_lines : str or list of (str or list of str)
        The X-ray lines given as ``"Element_Line"`` strings, e.g. ``["Fe_Ka",
        "Pt_La"]``, using the same line names as elsewhere in eXSpy (``Ka``,
        ``Kb``, ``La``, ``Lb1``, ``Lb2``, ``Lb3``, ``Lb4``, ``Lg1``, ``Lg3``,
        ``Ll``, ``Ln``, ``Ma``, ``Mb``, ``Mg`` and ``Mz``). A single string is
        also accepted. To obtain a single *combined* k-factor for a group of
        lines of the same element (e.g. the whole K shell ``["Cr_Ka",
        "Cr_Kb"]``), pass the group as a nested list or tuple: the
        cross-sections of the lines of a group are summed before computing its
        k-factor. The order of the returned factors matches the order of
        ``xray_lines`` (which should match the order of the intensities passed
        to ``EDSTEMSpectrum.quantification``; HyperSpy sorts these
        alphabetically).
    reference_element : str or int
        The element used as reference for the k-factors. Default is silicon.
    reference_line : str
        The reference X-ray line, using the same names as ``xray_lines``.
        Default is ``"Ka"``.
    form : {"weight", "atomic", "cross_section"}
        The convention of the returned factors:

        - ``"weight"`` (default): the Cliff-Lorimer k-factor for weight
          fractions, ``k = (A_element / A_reference) * (sigma_reference /
          sigma_element)``, where ``A`` are atomic weights and ``sigma`` the
          (summed) emission cross-sections. This is the factor expected by
          ``EDSTEMSpectrum.quantification`` with ``method="CL"``.
        - ``"atomic"``: the Cliff-Lorimer k-factor for atomic fractions,
          ``sigma_reference / sigma_element``.
        - ``"cross_section"``: the raw cross-section ratio,
          ``sigma_element / sigma_reference``.
    detector_efficiency : None, callable, array-like, str or pathlib.Path, optional
        The detection efficiency of the detector, used to correct the
        k-factors for quantification from raw (uncorrected) intensities:
        each factor is multiplied by the ratio of the (cross-section
        weighted) efficiency of the reference line to that of the
        corresponding line or group of lines. It can be given as a
        callable taking the energy in keV, a two-column (energy in keV,
        efficiency) array-like or the path to a two-column file (see
        :func:`load_detector_efficiency`); see
        :func:`detector_efficiency_from_layers` to calculate such an
        efficiency from the detector geometry. Not available with
        ``form="cross_section"``.

    Returns
    -------
    k_factors : list of float
        The k-factor of each line, in the same order as ``xray_lines``.

    Examples
    --------
    >>> exspy.utils.eds.get_k_factors(300, ["Fe_Ka", "Pt_La"])
    [1.75..., 6.17...]

    Whole shells and sub-shells are described by the group of lines they are
    made of. The K shell has no sub-shells, so the whole K shell of chromium is
    its K-alpha and K-beta lines, obtained either as separate factors or as a
    single combined factor (useful to compare with the k-factors reported by
    vendor software):

    >>> exspy.utils.eds.get_k_factors(300, ["Cr_Ka", "Cr_Kb"])   # separate
    [1.5841..., 12.312...]
    >>> exspy.utils.eds.get_k_factors(300, [["Cr_Ka", "Cr_Kb"]])  # combined
    [1.4036...]

    The L shell is split into the L1, L2 and L3 sub-shells (and the M shell
    into M1 to M5). The L3 sub-shell of tungsten groups its L3 lines, and the
    whole L shell is obtained by passing all its lines, either separately or
    combined:

    >>> exspy.utils.eds.get_k_factors(300, ["W_La", "W_Lb2", "W_Ll"])  # L3 sub-shell
    [5.977..., 37.516..., 127.697...]
    >>> exspy.utils.eds.get_k_factors(300, [["W_La", "W_Lb2", "W_Ll"]])  # combined L3
    [4.9558...]

    When quantifying from raw (uncorrected) intensities, the detection
    efficiency of the detector can be folded into the k-factors:

    >>> layers = [("Si", 100)]  # 100 nm silicon dead layer
    >>> efficiency = lambda E: exspy.utils.eds.detector_efficiency_from_layers(
    ...     E, layers, 0.45)
    >>> exspy.utils.eds.get_k_factors(300, ["Y_Ka"], detector_efficiency=efficiency)
    [8.3490...]

    Notes
    -----
    The X-ray line names identify the shell (and, for the L and M shells, the
    sub-shell) they originate from:

    - K shell: ``Ka``, ``Kb`` (no sub-shells)
    - L1 sub-shell: ``Lb3``, ``Lb4``, ``Lg3``
    - L2 sub-shell: ``Lb1``, ``Lg1``, ``Ln``
    - L3 sub-shell: ``La``, ``Lb2``, ``Ll``

    The k-factors are returned one per requested line. A group of lines of the
    same element (a nested list or tuple) is combined into a single k-factor by
    summing the cross-sections of its lines; this yields the k-factor of a
    whole shell or sub-shell as a single number (note that a combined factor is
    not the sum of the per-line factors).

    The k-factors are standardless theoretical factors computed from the
    emission cross-sections, which do not account for the detector
    efficiency (entrance window, dead layer, crystal response and
    geometry). Pass ``detector_efficiency`` to correct the k-factors for
    quantification from raw intensities, or correct the intensities
    themselves; for quantitative analysis, calibrate the k-factors
    against standards measured on the same instrument, as recommended
    for vendor-provided k-factors.

    See Also
    --------
    load_cross_section_table, load_detector_efficiency,
    detector_efficiency_from_layers
    """
    if not isinstance(form, str) or form not in ("weight", "atomic", "cross_section"):
        raise ValueError('`form` must be one of "weight", "atomic" or "cross_section".')

    if isinstance(xray_lines, str):
        xray_lines = [xray_lines]

    table = _bundled_table(beam_energy)
    eps = None
    eps_ref = None
    if detector_efficiency is not None:
        if form == "cross_section":
            raise ValueError(
                "The detector efficiency correction is not available with "
                'form="cross_section"; use the "weight" or "atomic" forms.'
            )
        eps = _as_efficiency_callable(detector_efficiency)
        eps_ref = _line_efficiency(table, reference_element, reference_line, eps)
        if eps_ref <= 0:
            raise ValueError(
                f"The detection efficiency of the reference line "
                f"({reference_element}_{reference_line}) is zero; this "
                "reference cannot be used to quantify with this detector."
            )
    k_factors = []
    for entry in xray_lines:
        if isinstance(entry, str):
            element, line = _get_element_and_line(entry)
            k = _get_k_factor(
                table,
                element,
                line,
                reference_element=reference_element,
                reference_line=reference_line,
                form=form,
            )
            if eps is not None:
                eps_el = _line_efficiency(table, element, line, eps)
                k *= eps_ref / _checked_efficiency(eps_el, entry)
            k_factors.append(k)
        else:
            element, sigma = _combined_cross_section(table, entry)
            k = _k_factor_from_sigma(
                table,
                element,
                sigma,
                reference_element,
                reference_line,
                form,
            )
            if eps is not None:
                eps_el = _group_efficiency(table, entry, eps)
                k *= eps_ref / _checked_efficiency(eps_el, "+".join(entry))
            k_factors.append(k)
    return k_factors
