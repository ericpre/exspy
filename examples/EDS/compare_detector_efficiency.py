"""
Detector efficiency curves: bundled SuperX versus calculated ones
==================================================================

This example compares the detection efficiency of silicon drift
detectors: the interpolated curve of a windowless SuperX detector bundled with
eXSpy (redistributed from the `ASTORUM
<https://github.com/sebastian-cozma/ASTORUM>`_ package) and curves
calculated from a detector geometry with a 25 nm aluminium window and a
50 nm silicon dead layer in front of active depths of 0.5, 1 and 2 mm.

"""

from functools import partial

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
# detectors. The calculated detectors have a 25 nm aluminium window and a
# 50 nm silicon dead layer in front of the active volume:
energies_superX, efficiency_superX = load_detector_efficiency(SUPERX_EFFICIENCY_FILE)

al_window = ("Al", 25)  # 25 nm aluminium window
dead_layer = ("Si", 25)  # 25 nm silicon dead layer
layers = [al_window, dead_layer]
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

# %%
# The dead layer in front of the active volume absorbs low-energy X-rays;
# compare the calculated curves for dead-layer thicknesses of 0, 25, 50,
# 100, 250 and 500 nm, with an active depth of 0.5 mm. The 25 nm aluminium
# window is included in front of the dead layer:
detector_thickness = 0.5  # mm
dead_layer_thicknesses = (0, 25, 50, 100, 250, 500)  # nm

plt.figure()
energies = np.linspace(0.2, 40, 200)
for dead_layer_thickness in dead_layer_thicknesses:
    layers = [al_window, ("Si", dead_layer_thickness)]
    label = f"calculated, {dead_layer_thickness:g} nm dead layer"
    efficiency = detector_efficiency_from_layers(energies, layers, detector_thickness)
    plt.plot(energies, efficiency, label=label)
plt.xlabel("Energy (keV)")
plt.ylabel("Detection efficiency")
plt.legend()

# %%
# The dead layer mainly affects the low energies, where the efficiency of
# the reference line is measured. At the energies of a few common X-ray
# lines, with an active depth of 0.5 mm:
header = f"{'line':>8} {'E (keV)':>8}"
for dead_layer_thickness in dead_layer_thicknesses:
    header += f" {f'{dead_layer_thickness:g} nm':>8}"
print(header)
for line, energy in [("Si_Ka", 1.74), ("Fe_Ka", 6.40), ("Y_Ka", 14.96)]:
    row = f"{line:>8} {energy:>8.2f}"
    for dead_layer_thickness in dead_layer_thicknesses:
        layers = [al_window, ("Si", dead_layer_thickness)]
        row += f" {detector_efficiency_from_layers(energy, layers, detector_thickness):>8.4f}"
    print(row)

# %%
# For the k-factor of the yttrium K-alpha line at 300 keV (silicon K-alpha
# reference), the dead layer lowers the efficiency of the reference line
# more than that of the yttrium line, so the corrected k-factor decreases:
for dead_layer_thickness in dead_layer_thicknesses:
    layers = [al_window, ("Si", dead_layer_thickness)]
    efficiency = partial(
        detector_efficiency_from_layers,
        layers=layers,
        detector_thickness=detector_thickness,
    )
    k = exspy.utils.eds.get_k_factors(300, ["Y_Ka"], detector_efficiency=efficiency)[0]
    print(f"{f'{dead_layer_thickness:g} nm:':>10} {k}")

# %%
# Finally, isolate the effect of the aluminium window itself by comparing
# the calculated curves with and without it, for a 50 nm dead layer and
# an active depth of 0.5 mm:
detector_thickness = 0.5  # mm
layers_with_window = [al_window, dead_layer]
layers_windowless = [dead_layer]

plt.figure()
energies = np.linspace(0.2, 40, 200)
plt.plot(
    energies,
    detector_efficiency_from_layers(energies, layers_with_window, detector_thickness),
    label=f"calculated, {al_window[1]:g} nm Al window",
)
plt.plot(
    energies,
    detector_efficiency_from_layers(energies, layers_windowless, detector_thickness),
    label="calculated, windowless",
)
plt.xlabel("Energy (keV)")
plt.ylabel("Detection efficiency")
plt.legend()

# %%
# The aluminium window absorbs the softest X-rays: at the energies of a few
# common X-ray lines, its transmission is:
header = f"{'line':>8} {'E (keV)':>8} {'window':>8} {'no window':>10}"
print(header)
for line, energy in [("Si_Ka", 1.74), ("Fe_Ka", 6.40), ("Y_Ka", 14.96)]:
    with_window = detector_efficiency_from_layers(
        energy, layers_with_window, detector_thickness
    )
    without_window = detector_efficiency_from_layers(
        energy, layers_windowless, detector_thickness
    )
    print(
        f"{line:>8} {energy:>8.2f} {float(with_window):>8.4f} {float(without_window):>10.4f}"
    )

# %%
# Correcting the k-factor of the yttrium K-alpha line at 300 keV
# (silicon K-alpha reference) for the aluminium window:
efficiency_with_window = partial(
    detector_efficiency_from_layers,
    layers=layers_with_window,
    detector_thickness=detector_thickness,
)
efficiency_windowless = partial(
    detector_efficiency_from_layers,
    layers=layers_windowless,
    detector_thickness=detector_thickness,
)
k_with_window = exspy.utils.eds.get_k_factors(
    300, ["Y_Ka"], detector_efficiency=efficiency_with_window
)[0]
k_windowless = exspy.utils.eds.get_k_factors(
    300, ["Y_Ka"], detector_efficiency=efficiency_windowless
)[0]
print(f"{'windowless:':>14} {k_windowless}")
print(f"{'Al window:':>14} {k_with_window}")

# %%
# Finally, X-rays entering the detector at an angle to its normal — the
# *X-ray entrance angle*, called the beam entrance angle in the Bruker
# QUANTAX software — cross every layer and the active volume along a path
# longer by ``1 / cos(xray_entrance_angle)``, i.e. a larger effective
# thickness. Compare the calculated curves for entrance angles of 0, 20,
# 35 and 45 degrees, with the 25 nm aluminium window and the 50 nm dead
# layer in front of a 0.5 mm active depth:
xray_entrance_angles = (0, 20, 35, 45)  # degrees

plt.figure()
energies = np.linspace(0.2, 40, 200)
for xray_entrance_angle in xray_entrance_angles:
    label = f"calculated, {xray_entrance_angle:g} degrees entrance angle"
    efficiency = detector_efficiency_from_layers(
        energies,
        layers_with_window,
        detector_thickness,
        xray_entrance_angle=xray_entrance_angle,
    )
    plt.plot(energies, efficiency, label=label)
plt.xlabel("Energy (keV)")
plt.ylabel("Detection efficiency")
plt.legend()

# %%
# The entrance angle mostly improves the absorption in the active volume,
# which dominates at high energies. At the energies of a few common X-ray
# lines:
header = f"{'line':>8} {'E (keV)':>8}"
for xray_entrance_angle in xray_entrance_angles:
    header += f" {f'{xray_entrance_angle:g} deg':>8}"
print(header)
for line, energy in [("Si_Ka", 1.74), ("Fe_Ka", 6.40), ("Y_Ka", 14.96)]:
    row = f"{line:>8} {energy:>8.2f}"
    for xray_entrance_angle in xray_entrance_angles:
        efficiency = detector_efficiency_from_layers(
            energy,
            layers_with_window,
            detector_thickness,
            xray_entrance_angle=xray_entrance_angle,
        )
        row += f" {float(efficiency):>8.4f}"
    print(row)

# %%
# For the k-factor of the yttrium K-alpha line at 300 keV (silicon
# K-alpha reference), the entrance angle increases the efficiency of the
# yttrium line more than that of the reference line, so the corrected
# k-factor decreases:
for xray_entrance_angle in xray_entrance_angles:
    efficiency = partial(
        detector_efficiency_from_layers,
        layers=layers_with_window,
        detector_thickness=detector_thickness,
        xray_entrance_angle=xray_entrance_angle,
    )
    k = exspy.utils.eds.get_k_factors(300, ["Y_Ka"], detector_efficiency=efficiency)[0]
    print(f"{f'{xray_entrance_angle:g} deg:':>10} {k}")

# %%
# Note that the entrance angle also reduces the solid angle subtended by
# the detector by ``cos(xray_entrance_angle)``, lowering the measured
# count rates; that effect does not change the efficiency and cancels
# out in the k-factor correction.

plt.show()
