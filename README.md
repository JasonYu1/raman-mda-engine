# raman-mda-engine

[![License](https://img.shields.io/pypi/l/raman-mda-engine.svg?color=green)](https://github.com/JasonYu1/raman-mda-engine/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/raman-mda-engine.svg?color=green)](https://pypi.org/project/raman-mda-engine)
[![Python Version](https://img.shields.io/pypi/pyversions/raman-mda-engine.svg?color=green)](https://www.python.org)
[![CI](https://github.com/JasonYu1/raman-mda-engine/actions/workflows/ci/badge.svg)](https://github.com/JasonYu1/raman-mda-engine/actions)

`raman-mda-engine` extends the
[`pymmcore-plus`](https://github.com/pymmcore-plus/pymmcore-plus) MDA engine to
coordinate microscope imaging with point-based Raman acquisition. It adds Raman
aiming sources, spectral-acquisition events, optional autofocus and cell
tracking, and a writer that stores Raman spectra alongside the image data from
an MDA sequence.

This package is the experiment-orchestration layer in a three-package Raman
microscope stack:

- [`raman-control`](https://github.com/JasonYu1/raman-control) provides the
  hardware-control layer for the detector, spectrograph, DAQ, galvos, shutter,
  filter actuator, and coordinate calibration.
- `raman-mda-engine` coordinates imaging, Raman aiming, acquisition, tracking,
  autofocus, and data writing.
- [`napari-raman-widget`](https://github.com/JasonYu1/napari-raman-widget)
  provides the napari-based user interface.

Hardware SDK calls belong in `raman-control`, while experiment timing and MDA
behavior belong here. This separation allows a detector or DAQ integration to
be replaced without rewriting the acquisition workflow or user interface.

## Acquisition workflow

`RamanEngine` accepts a `useq.MDASequence` through `pymmcore-plus`. Raman
collection is enabled by adding a `raman` mapping to the sequence metadata and
attaching one or more aiming sources to the engine.

```python
from useq import MDASequence

sequence = MDASequence(
    metadata={
        "raman": {
            "channel": "BF",
            "z": "center",
        }
    },
    stage_positions=[(100, 100, 30), (200, 150, 35)],
    channels=["BF", "DAPI"],
    time_plan={"interval": 60, "loops": 10},
    z_plan={"range": 4, "step": 0.5},
    axis_order="tpcz",
)
```

The `channel` value identifies the imaging channel associated with Raman
collection. The `z` value controls which Z planes collect spectra; use
`"center"` for the center plane or `"all"` for every plane.

An engine also needs a hardware-compatible spectra collector, coordinate
transformer, and aiming source. The exact construction is rig-specific:

```python
from pymmcore_plus import CMMCorePlus

from raman_mda_engine import RamanEngine
from raman_mda_engine.aiming import SimpleGridSource

core = CMMCorePlus.instance()
core.loadSystemConfiguration("path/to/micromanager.cfg")

collector = ...    # collector supplied by raman-control
transformer = ...  # maps bright-field coordinates to galvo voltages

engine = RamanEngine(
    mmc=core,
    spectra_collector=collector,
    transformer=transformer,
    sources=[SimpleGridSource(N_x=5, N_y=5)],
    default_rm_exp=20,  # milliseconds
)
core.register_mda_engine(engine)
core.run_mda(sequence)
```

The collector and transformer must match the installed microscope. See
[`raman-control`](https://github.com/JasonYu1/raman-control) for the included
hardware integrations and the interface expected from a custom collector.

## Raman aiming sources

Aiming sources return normalized `(N, 2)` coordinates in the range `[0, 1]`
for each MDA event. The package includes:

- `SimpleGridSource` for a fixed rectangular grid across the Raman field of
  view.
- `PointsLayerSource` for points stored in a napari broadcastable-points layer.
- `ShapesLayerSource` for sampling points within shapes displayed in napari.
- `LabelsLayerSource` for sampling points from a napari labels layer.

Custom sources can be used by implementing the `RamanAimingSource` protocol:

```python
import numpy as np


class CenterPointSource:
    name = "center"

    def get_mda_points(self, event):
        return np.array([[0.5, 0.5]])
```

Sources used for interactive snapshots should also implement
`get_current_points()` and satisfy the `SnappableRamanAimingSource` protocol.

## Saving images and spectra

`RamanTiffAndNumpyWriter` extends the `pymmcore-mda-writers` TIFF writer. It
stores image data normally and creates two additional directories:

- `raman/` contains spectral arrays, acquisition locations, source
  designations, and JSON metadata for each Raman event.
- `aiming/` contains compressed segmentation and tracking results.

```python
from raman_mda_engine import RamanTiffAndNumpyWriter

writer = RamanTiffAndNumpyWriter("path/to/output", core=core)
```

Keep the writer alive for the duration of the acquisition so it remains
connected to the engine's Raman signals.

## Installation

This project is research software for a custom microscope and is normally
installed from source in the same environment as its companion packages:

```bash
git clone https://github.com/JasonYu1/raman-mda-engine.git
cd raman-mda-engine
pip install -e .
```

Install `raman-control` and the vendor SDKs required by the selected detector,
spectrograph, and DAQ separately. Development dependencies can be installed
with:

```bash
pip install -e ".[dev,testing]"
```

## Hardware configuration and safety

This repository does not provide a universal microscope configuration. Before
running an acquisition, verify all rig-specific settings, including:

- the Micro-Manager configuration and device names;
- camera and spectrograph SDK initialization;
- DAQ channels, triggers, and timing;
- galvo coordinate calibration, polarity, and voltage limits;
- shutter and filter-actuator behavior;
- autofocus ranges, image dimensions, and segmentation parameters.

Treat all example positions, exposure times, voltage limits, device names, and
timing values as instrument-specific placeholders. Confirm the wiring and safe
operating range of every connected device before enabling laser output or
moving the galvo mirrors.

## Development status

`raman-mda-engine` is alpha-stage research software developed for a custom
Raman microscope. Its hardware configuration, calibration, autofocus, and
tracking settings may need to be adapted and validated for each instrument.

## License

`raman-mda-engine` is distributed under the BSD 3-Clause License.
