"""
Detector efficiency curves: bundled SuperX versus calculated ones
==================================================================

This example compares the detection efficiency of windowless silicon
drift detectors: the interpolated curve of a SuperX detector bundled with
eXSpy (redistributed from the `ASTORUM
<https://github.com/sebastian-cozma/ASTORUM>`_ package) and curves
calculated from a detector geometry with a 100 nm silicon dead layer and
active depths of 0.5, 1 and 2 mm.

"""

import matplotlib.pyplot as plt
import numpy as np

import exspy
from exspy.utils.eds import (
    SUPERX_EFFICIENCY_FILE,
    detector_efficiency_from_layers,
    load_detector_efficiency,
)

# %%
# Load the bundled SuperX curve and define the calculated windowless
# detectors:
energies_superX, efficiency_superX = load_detector_efficiency(SUPERX_EFFICIENCY_FILE)

layers = [("Si", 100)]  # 100 nm silicon dead layer
detector_thicknesses = (0.3, 0.4, 0.5, 1.0, 2.0)  # mm


def efficiency(E, detector_thickness):
    return detector_efficiency_from_layers(E, layers, detector_thickness)


# %%
# Plot the curves between 0.2 and 40 keV:
energies = np.linspace(0.2, 40, 200)
plt.figure()
plt.plot(energies_superX, efficiency_superX, label="SuperX (bundled)")
for detector_thickness in detector_thicknesses:
    label = f"calculated, {detector_thickness:g} mm"
    plt.plot(energies, efficiency(energies, detector_thickness), label=label)
plt.xlabel("Energy (keV)")
plt.ylabel("Detection efficiency")
plt.legend()

# %%
# At the energies of a few common X-ray lines:
header = f"{'line':>8} {'E (keV)':>8} {'SuperX':>8}"
for detector_thickness in detector_thicknesses:
    header += f" {f'{detector_thickness:g} mm':>8}"
print(header)
for line, energy in [("Si_Ka", 1.74), ("Fe_Ka", 6.40), ("Y_Ka", 14.96)]:
    row = (
        f"{line:>8} {energy:>8.2f} "
        f"{float(np.interp(energy, energies_superX, efficiency_superX)):>8.4f}"
    )
    for detector_thickness in detector_thicknesses:
        row += f" {efficiency(energy, detector_thickness):>8.4f}"
    print(row)

# %%
# The efficiency correction is folded into the standardless k-factors by
# multiplying each factor by the ratio of the efficiency of the reference
# line to that of the line of interest. For the k-factor of the yttrium
# K-alpha line at 300 keV (silicon K-alpha reference):
print(f"{'uncorrected:':>16}", exspy.utils.eds.get_k_factors(300, ["Y_Ka"])[0])
print(
    f"{'SuperX:':>16}",
    exspy.utils.eds.get_k_factors(
        300, ["Y_Ka"], detector_efficiency=SUPERX_EFFICIENCY_FILE
    )[0],
)
for detector_thickness in detector_thicknesses:
    k = exspy.utils.eds.get_k_factors(
        300,
        ["Y_Ka"],
        detector_efficiency=lambda E, t=detector_thickness: efficiency(E, t),
    )[0]
    print(f"{f'{detector_thickness:g} mm:':>16} {k}")

plt.show()
