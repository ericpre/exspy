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

import numpy as np
import pytest

from exspy import material
from exspy.utils.eds import (
    _get_element_and_line,
    detector_efficiency_from_layers,
    get_k_factors,
    load_cross_section_table,
    load_detector_efficiency,
)
from exspy.utils.eds._k_factors import _DATA_DIR


def test_get_element_and_line():
    assert _get_element_and_line("Mn_Ka") == ("Mn", "Ka")

    with pytest.raises(ValueError):
        _get_element_and_line("MnKa")


class TestLoadCrossSectionTable:
    def test_load(self):
        table, metadata = load_cross_section_table(_DATA_DIR / "300keV_xrays.json")
        assert metadata["beam_energy"] == 300
        assert metadata["lines"] is True
        assert table["24"]["KL3"]["energy"] == 5.4147
        assert table["24"]["KL3"]["cs"] == pytest.approx(5.56890e-23, rel=1e-4)


class TestGetKFactors:
    def test_weight_cr_ka(self):
        assert get_k_factors(300, ["Cr_Ka"]) == [pytest.approx(1.5841454)]

    def test_weight_o_ka(self):
        assert get_k_factors(300, ["O_Ka"]) == [pytest.approx(0.9388157)]

    def test_weight_y_ka(self):
        assert get_k_factors(300, ["Y_Ka"]) == [pytest.approx(5.3171011)]

    def test_cross_section_form(self):
        assert get_k_factors(300, ["Cr_Ka"], form="cross_section") == [
            pytest.approx(1.1686745)
        ]

    def test_atomic_form(self):
        assert get_k_factors(300, ["Cr_Ka"], form="atomic") == [
            pytest.approx(0.8556703)
        ]

    def test_ka_vs_kb(self):
        assert get_k_factors(300, ["Cr_Kb"], form="cross_section") == [
            pytest.approx(0.1503695)
        ]

    def test_w_l_families(self):
        assert get_k_factors(300, ["W_La", "W_Lb1", "W_Lb2"]) == [
            pytest.approx(5.9773607),
            pytest.approx(15.7097153),
            pytest.approx(37.5158117),
        ]

    def test_w_m_families(self):
        assert get_k_factors(300, ["W_Ma", "W_Mz"]) == [
            pytest.approx(2.8650881),
            pytest.approx(21.7629399),
        ]

    def test_reference_is_self(self):
        assert get_k_factors(300, ["Si_Ka"], form="cross_section") == [1.0]

    def test_single_string(self):
        assert get_k_factors(300, "Cr_Ka") == [pytest.approx(1.5841454)]

    def test_combined_ka_kb(self):
        # A nested list groups lines into a single combined factor (whole K shell).
        assert get_k_factors(300, [["Cr_Ka", "Cr_Kb"]]) == [pytest.approx(1.4035547)]

    def test_combined_mixed(self):
        assert get_k_factors(300, [["Cr_Ka", "Cr_Kb"], "Pt_La"]) == [
            pytest.approx(1.4035547),
            pytest.approx(6.1705867),
        ]

    def test_combined_group_same_element_required(self):
        with pytest.raises(ValueError, match="same element"):
            get_k_factors(300, [["Cr_Ka", "Fe_Ka"]])

    def test_beam_energy_variants(self):
        cr_ka_100 = get_k_factors(100, ["Cr_Ka"])[0]
        cr_ka_300 = get_k_factors(300, ["Cr_Ka"])[0]
        assert isinstance(cr_ka_100, float) and cr_ka_100 > 0
        assert cr_ka_100 != pytest.approx(cr_ka_300)

    def test_beam_energy_as_float(self):
        # Microscope metadata stores the beam energy as a float (e.g. 200.0);
        # it must resolve to the same bundled table as the int.
        assert get_k_factors(200.0, ["Cr_Ka"]) == pytest.approx(
            get_k_factors(200, ["Cr_Ka"])
        )

    def test_order(self):
        assert get_k_factors(300, ["Cr_Ka", "Y_Ka"]) == [
            pytest.approx(1.5841454),
            pytest.approx(5.3171011),
        ]

    def test_invalid_beam_energy(self):
        with pytest.raises(ValueError, match="beam energy"):
            get_k_factors(150, ["Cr_Ka"])

    def test_invalid_form(self):
        with pytest.raises(ValueError, match="form"):
            get_k_factors(300, ["Cr_Ka"], form="bogus")

    def test_unknown_element(self):
        with pytest.raises(ValueError):
            get_k_factors(300, ["Xx_Ka"])

    def test_element_not_in_table(self):
        # Beryllium has no X-ray lines in the bundled tables (min Z is 5).
        with pytest.raises(ValueError):
            get_k_factors(300, ["Be_Ka"])

    def test_unknown_line(self):
        with pytest.raises(ValueError):
            get_k_factors(300, ["Cr_Bogus"])

    def test_line_name_missing_from_table(self):
        # Expected: oxygen has no K-beta line, so "Kb" cannot be resolved.
        with pytest.raises(ValueError):
            get_k_factors(300, ["O_Kb"])


class TestDetectorEfficiencyFromLayers:
    def test_full_absorption(self):
        energies = np.array([1.0, 5.0, 10.0])
        efficiency = detector_efficiency_from_layers(energies, [], 10.0)
        macs = material.mass_absorption_coefficient(element="Si", energies=energies)
        density = material._elements_dict["Si"]["Physical_properties"][
            "density (g/cm^3)"
        ]
        expected = 1 - np.exp(-(macs * density * 10.0 * 1e-1))
        np.testing.assert_allclose(efficiency, expected)

    def test_window_absorption(self):
        energies = np.array([1.74, 14.96])
        layers = [("Be", 25000)]  # 25 um Be window
        efficiency = detector_efficiency_from_layers(energies, layers, 0.45)
        macs_be = material.mass_absorption_coefficient(element="Be", energies=energies)
        rho_be = material._elements_dict["Be"]["Physical_properties"][
            "density (g/cm^3)"
        ]
        window = np.exp(-(macs_be * rho_be * 25000 * 1e-7))
        macs_si = material.mass_absorption_coefficient(element="Si", energies=energies)
        rho_si = material._elements_dict["Si"]["Physical_properties"][
            "density (g/cm^3)"
        ]
        active = 1 - np.exp(-(macs_si * rho_si * 0.45 * 1e-1))
        np.testing.assert_allclose(efficiency, window * active)

    def test_cutoff_energy(self):
        energies = np.array([0.01, 0.06, 5.0])
        efficiency = detector_efficiency_from_layers(energies, [], 10.0)
        assert efficiency[0] == 0
        assert efficiency[1] > 0
        assert efficiency[2] > 0

    def test_scalar_input(self):
        # at 20 keV, a 0.45 mm thick silicon detector absorbs ~80% of the
        # X-rays, so the efficiency is strictly between 0 and 1
        efficiency = detector_efficiency_from_layers(20.0, [], 0.45)
        assert np.ndim(efficiency) == 0
        assert 0 < efficiency < 1

    def test_thicker_window_absorbs_more(self):
        energies = np.array([1.74])
        thin = detector_efficiency_from_layers(energies, [("Al", 10)], 10.0)
        thick = detector_efficiency_from_layers(energies, [("Al", 10000)], 10.0)
        assert thick[0] < thin[0]


class TestLoadDetectorEfficiency:
    def test_load_and_sort(self, tmp_path):
        filename = tmp_path / "efficiency.txt"
        filename.write_text("6.4 0.5\n1.7 0.9\n10.0 0.4\n")
        energies, efficiencies = load_detector_efficiency(filename)
        np.testing.assert_allclose(energies, [1.7, 6.4, 10.0])
        np.testing.assert_allclose(efficiencies, [0.9, 0.5, 0.4])

    def test_single_row(self, tmp_path):
        filename = tmp_path / "efficiency.txt"
        filename.write_text("5.0 0.5\n")
        energies, efficiencies = load_detector_efficiency(filename)
        assert energies[0] == 5.0
        assert efficiencies[0] == 0.5

    def test_wrong_number_of_columns(self, tmp_path):
        filename = tmp_path / "efficiency.txt"
        filename.write_text("1.0 0.5 3\n")
        with pytest.raises(ValueError, match="two columns"):
            load_detector_efficiency(filename)


class TestGetKFactorsDetectorEfficiency:
    def test_constant_efficiency_unchanged(self):
        def eps(E):
            return 0.7

        for lines in (["Cr_Ka"], [["Cr_Ka", "Cr_Kb"]], ["W_La", "Pt_La"]):
            assert get_k_factors(300, lines, detector_efficiency=eps) == pytest.approx(
                get_k_factors(300, lines)
            )

    def test_efficiency_ratio(self):
        def eps(E):
            return E / 40.0

        table, _ = load_cross_section_table(_DATA_DIR / "300keV_xrays.json")

        def eps_eff(z, lines):
            total = sum(table[z][l]["cs"] for l in lines)
            weighted = sum(
                table[z][l]["cs"] * table[z][l]["energy"] / 40 for l in lines
            )
            return weighted / total

        corrected = get_k_factors(300, ["Cr_Ka"], detector_efficiency=eps)[0]
        uncorrected = get_k_factors(300, ["Cr_Ka"])[0]
        expected = (
            uncorrected * eps_eff("14", ["KL3", "KL2"]) / eps_eff("24", ["KL3", "KL2"])
        )
        assert corrected == pytest.approx(expected)

    def test_group_efficiency(self):
        def eps(E):
            return E / 40.0

        table, _ = load_cross_section_table(_DATA_DIR / "300keV_xrays.json")

        def eps_eff(z, lines):
            total = sum(table[z][l]["cs"] for l in lines)
            weighted = sum(
                table[z][l]["cs"] * table[z][l]["energy"] / 40 for l in lines
            )
            return weighted / total

        corrected = get_k_factors(300, [["Cr_Ka", "Cr_Kb"]], detector_efficiency=eps)[0]
        uncorrected = get_k_factors(300, [["Cr_Ka", "Cr_Kb"]])[0]
        expected = (
            uncorrected
            * eps_eff("14", ["KL3", "KL2"])
            / eps_eff("24", ["KL3", "KL2", "KM3", "KM2"])
        )
        assert corrected == pytest.approx(expected)

    def test_file_input(self, tmp_path):
        # linear curve sampled densely, so that the interpolation is exact
        energies = np.linspace(0.2, 40, 198)
        curve = np.column_stack([energies, energies / 40])
        filename = tmp_path / "efficiency.txt"
        np.savetxt(filename, curve)
        from_file = get_k_factors(300, ["Cr_Ka"], detector_efficiency=filename)
        from_array = get_k_factors(300, ["Cr_Ka"], detector_efficiency=curve)
        assert from_file == pytest.approx(from_array)

    def test_out_of_range(self, tmp_path):
        filename = tmp_path / "efficiency.txt"
        filename.write_text("1.0 0.5\n10.0 0.2\n")
        with pytest.raises(ValueError, match="does not cover"):
            get_k_factors(300, ["Y_Ka"], detector_efficiency=filename)

    def test_cross_section_form_unsupported(self):
        def eps(E):
            return 0.5

        with pytest.raises(ValueError, match="cross_section"):
            get_k_factors(300, ["Cr_Ka"], form="cross_section", detector_efficiency=eps)

    def test_zero_efficiency(self):
        # zero above 5 keV, i.e. at the Cr_Ka line (5.41 keV), but not at
        # the Si_Ka reference line (1.74 keV)
        def eps(E):
            return 0.0 if E > 5 else 1.0

        with pytest.raises(ValueError, match="zero"):
            get_k_factors(300, ["Cr_Ka"], detector_efficiency=eps)

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="two-column"):
            get_k_factors(300, ["Cr_Ka"], detector_efficiency=42)
