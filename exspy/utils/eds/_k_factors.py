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
factors that do not account for detector efficiency; see its documentation for
details.
"""

from pathlib import Path

from exspy import material
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


def get_k_factors(
    beam_energy,
    xray_lines,
    reference_element="Si",
    reference_line="Ka",
    form="weight",
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
    emission cross-sections, which do not account for the detector efficiency
    (entrance window, dead layer, crystal response and geometry). For
    quantitative analysis, calibrate the k-factors against standards measured
    on the same instrument, as recommended for vendor-provided k-factors.

    See Also
    --------
    load_cross_section_table
    """
    if not isinstance(form, str) or form not in ("weight", "atomic", "cross_section"):
        raise ValueError('`form` must be one of "weight", "atomic" or "cross_section".')

    if isinstance(xray_lines, str):
        xray_lines = [xray_lines]

    table = _bundled_table(beam_energy)
    k_factors = []
    for entry in xray_lines:
        if isinstance(entry, str):
            element, line = _get_element_and_line(entry)
            k_factors.append(
                _get_k_factor(
                    table,
                    element,
                    line,
                    reference_element=reference_element,
                    reference_line=reference_line,
                    form=form,
                )
            )
        else:
            element, sigma = _combined_cross_section(table, entry)
            k_factors.append(
                _k_factor_from_sigma(
                    table,
                    element,
                    sigma,
                    reference_element,
                    reference_line,
                    form,
                )
            )
    return k_factors
