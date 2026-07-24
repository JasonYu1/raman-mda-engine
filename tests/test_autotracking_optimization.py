import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from raman_mda_engine import RamanEngine
from raman_mda_engine.aiming.autotracking import CellposeSegmenter, track_one_T


class _FakeCellposeModel:
    def __init__(self, eval_shapes):
        self.cp = self
        self.eval_shapes = eval_shapes

    def eval(self, image, **kwargs):
        self.eval_shapes.append(image.shape)
        mask = np.zeros(image.shape, dtype=np.int32)
        mask[image.shape[0] // 2, image.shape[1] // 2] = 1
        return mask, None, None


def test_segmenter_crops_roi_reuses_model_and_preserves_global_scale():
    model_creations = []
    eval_shapes = []

    def model_factory(**kwargs):
        model_creations.append(kwargs)
        return _FakeCellposeModel(eval_shapes)

    segmenter = CellposeSegmenter(
        cellpose_model="cyto2",
        model_factory=model_factory,
    )
    image = np.zeros((100, 120), dtype=np.uint16)

    first = segmenter.segment(
        image,
        scale=4,
        crop=True,
        circle_center=(50, 60),
        circle_radius=10,
    )
    second = segmenter.segment(
        image,
        scale=4,
        crop=True,
        circle_center=(50, 60),
        circle_radius=10,
    )

    assert len(model_creations) == 1
    assert eval_shapes == [(8, 8), (8, 8)]
    assert first.shape == (25, 30)
    assert second.shape == first.shape
    # The ROI label remains addressable using full-image point / scale.
    assert first[13, 15] == 1
    assert np.count_nonzero(first) == 1


def test_track_one_t_retains_point_when_current_mask_has_no_cells():
    labels = np.zeros((2, 20, 20), dtype=np.uint16)
    original_point = np.array([[12.0, 16.0]])

    returned_labels, new_points = track_one_T(
        labels,
        scale=2,
        pts=original_point,
        tracked=labels,
        use_same_img=True,
    )

    assert returned_labels is labels
    np.testing.assert_allclose(new_points, original_point)


class _FakeSource:
    def __init__(self, name, point):
        self.name = name
        self.point = np.asarray(point, dtype=float)
        self._points = object()

    def get_mda_points(self, event, transform=False):
        return np.atleast_2d(self.point)


def _bare_engine(sources):
    engine = RamanEngine.__new__(RamanEngine)
    engine.aiming_sources = sources
    engine._image_x = 100
    engine._image_y = 100
    engine._scale = 4
    engine._segment_crop = True
    engine._cellpose_model = "cyto2"
    engine._tracking_config = "particle_config.json"
    engine._circle_center = (50, 50)
    engine._circle_radius = 20
    engine._cellpose_segmenter = object()
    engine._last_segments = {}
    engine._tracks = {}
    engine.raman_events = SimpleNamespace(
        aimUpdated=SimpleNamespace(emit=MagicMock())
    )
    return engine


def test_update_aim_skips_cellpose_when_neighbour_mask_is_reused():
    source = _FakeSource("cells", (0.1, 0.2))
    engine = _bare_engine([source])
    previous_mask = np.ones((25, 25), dtype=np.int32)
    engine._last_segments[0] = previous_mask
    event = SimpleNamespace(index={"p": 1, "t": 0})

    with patch("raman_mda_engine._engine.segment_single_img") as segment:
        engine.update_aim(1, event, np.zeros((100, 100)), use_same_img=True)

    segment.assert_not_called()
    assert engine._last_segments[1] is previous_mask


def test_update_aim_accepts_one_point_and_updates_every_cell_source():
    sources = [
        _FakeSource("cells-a", (0.1, 0.2)),
        _FakeSource("cells-b", (0.3, 0.4)),
    ]
    engine = _bare_engine(sources)
    previous_mask = np.ones((25, 25), dtype=np.int32)
    new_mask = np.full((25, 25), 2, dtype=np.int32)
    tracked = np.stack([previous_mask, new_mask])
    new_points = np.array([[20.0, 30.0], [40.0, 50.0]])
    engine._last_segments[0] = previous_mask
    event = SimpleNamespace(index={"p": 0, "t": 1})

    with patch(
        "raman_mda_engine._engine.segment_single_img",
        return_value=new_mask,
    ):
        with patch(
            "raman_mda_engine._engine.track_one_T",
            return_value=(tracked, new_points),
        ):
            with patch(
                "raman_mda_engine._engine.update_pos_points"
            ) as update_points:
                engine.update_aim(0, event, np.zeros((100, 100)))

    assert update_points.call_count == 2
    np.testing.assert_allclose(update_points.call_args_list[0].args[1], [[20, 30]])
    np.testing.assert_allclose(update_points.call_args_list[1].args[1], [[40, 50]])


def test_update_aim_keeps_a_valid_single_tracked_point():
    source = _FakeSource("cells", (0.1, 0.2))
    engine = _bare_engine([source])
    previous_mask = np.ones((25, 25), dtype=np.int32)
    new_mask = np.full((25, 25), 2, dtype=np.int32)
    tracked = np.stack([previous_mask, new_mask])
    engine._last_segments[0] = previous_mask
    event = SimpleNamespace(index={"p": 0, "t": 1})

    with patch(
        "raman_mda_engine._engine.segment_single_img",
        return_value=new_mask,
    ):
        with patch(
            "raman_mda_engine._engine.track_one_T",
            return_value=(tracked, np.array([[20.0, 30.0]])),
        ):
            with patch(
                "raman_mda_engine._engine.update_pos_points"
            ) as update_points:
                engine.update_aim(0, event, np.zeros((100, 100)))

    np.testing.assert_allclose(update_points.call_args.args[1], [[20, 30]])


class _RecoveringImageCore:
    def __init__(self, *, always_fail=False):
        self.always_fail = always_fail
        self.snap_calls = 0
        self.get_calls = 0
        self.wait_calls = 0

    def snapImage(self):
        self.snap_calls += 1

    def getImage(self):
        self.get_calls += 1
        if self.always_fail or self.get_calls == 1:
            raise RuntimeError("camera buffer read failed")
        return np.full((3, 4), 7, dtype=np.uint16)

    def waitForSystem(self):
        self.wait_calls += 1


class _ReloadCore:
    def __init__(self, *, fail_loads=0):
        self.fail_loads = fail_loads
        self.load_calls = 0
        self.unload_calls = 0
        self.config_history = []
        signal = lambda: SimpleNamespace(disconnect=MagicMock())
        self.events = SimpleNamespace(
            channelGroupChanged=signal(),
            configGroupChanged=signal(),
            propertyChanged=signal(),
            systemConfigurationLoaded=signal(),
            configSet=signal(),
        )

    def unloadAllDevices(self):
        self.unload_calls += 1

    def loadSystemConfiguration(self, config_file):
        self.load_calls += 1
        if self.load_calls <= self.fail_loads:
            raise RuntimeError("COM17 initialization failed")

    def setConfig(self, group, config):
        self.config_history.append((group, config))

    def waitForSystem(self):
        pass


def test_snap_and_get_repeats_the_whole_transaction_after_reload():
    engine = RamanEngine.__new__(RamanEngine)
    engine._mmc = _RecoveringImageCore()
    engine.reload = MagicMock()

    with patch("raman_mda_engine._engine.time.sleep"):
        image = engine.try_snap_and_get_image(N=2)

    np.testing.assert_array_equal(image, np.full((3, 4), 7, dtype=np.uint16))
    assert engine._mmc.snap_calls == 2
    assert engine._mmc.get_calls == 2
    engine.reload.assert_called_once_with()
    assert engine._last_operation_reloaded is True


def test_snap_and_get_raises_instead_of_returning_none():
    engine = RamanEngine.__new__(RamanEngine)
    engine._mmc = _RecoveringImageCore(always_fail=True)
    engine.reload = MagicMock()

    with patch("raman_mda_engine._engine.time.sleep"):
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "snap/get image failed after 2 attempts"
        ):
            engine.try_snap_and_get_image(N=2)

    assert engine._mmc.snap_calls == 2
    engine.reload.assert_called_once_with()


def test_reload_reports_failures_then_restores_channels():
    engine = RamanEngine.__new__(RamanEngine)
    engine._mmc = _ReloadCore(fail_loads=1)
    engine._config_file = "microscope.cfg"

    with patch("raman_mda_engine._engine.time.sleep"):
        engine.reload(N=2)

    assert engine._mmc.load_calls == 2
    assert engine._mmc.unload_calls == 2
    assert engine._mmc.config_history == [
        ("Channel", "GFP"),
        ("Channel", "BF"),
    ]


def test_reload_raises_the_last_hardware_error():
    engine = RamanEngine.__new__(RamanEngine)
    engine._mmc = _ReloadCore(fail_loads=2)
    engine._config_file = "microscope.cfg"

    with patch("raman_mda_engine._engine.time.sleep"):
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "COM17 initialization failed"
        ):
            engine.reload(N=2)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_function in (
        test_segmenter_crops_roi_reuses_model_and_preserves_global_scale,
        test_track_one_t_retains_point_when_current_mask_has_no_cells,
        test_update_aim_skips_cellpose_when_neighbour_mask_is_reused,
        test_update_aim_accepts_one_point_and_updates_every_cell_source,
        test_update_aim_keeps_a_valid_single_tracked_point,
        test_snap_and_get_repeats_the_whole_transaction_after_reload,
        test_snap_and_get_raises_instead_of_returning_none,
        test_reload_reports_failures_then_restores_channels,
        test_reload_raises_the_last_hardware_error,
    ):
        suite.addTest(unittest.FunctionTestCase(test_function))
    return suite
