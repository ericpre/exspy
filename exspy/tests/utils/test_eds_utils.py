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

import pytest

from exspy.utils.eds import (
    _get_element_and_line,
    get_k_factors,
    load_cross_section_table,
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
