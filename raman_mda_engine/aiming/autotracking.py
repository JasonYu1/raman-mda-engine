from __future__ import annotations
from skimage.transform import rescale
import numpy as np
from cellpose.models import Cellpose
from time import perf_counter
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import btrack
from pathlib import Path
from scipy.ndimage import center_of_mass

__all__= [
    "CellposeSegmenter",
    "track_one_T",
    "segment_single_img",
    "find_com",
    "update_pos_points",
]


import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import center_of_mass
import multiprocessing as mp
import btrack

def run_tracking(labels, radius, tracking_config='particle_config.json'):
    objects = btrack.utils.segmentation_to_objects(labels)
    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure(tracking_config)
        tracker.max_search_radius = radius
        tracker.volume = ((0, labels.shape[-2]), (0, labels.shape[-1]), (-1e5, 1e5))
        tracker.append(objects)
        tracker.track(step_size=100)
        tracker.optimize()
        tracks = tracker.tracks

        if len(tracks) == 0:
            return labels
        
    return btrack.utils.update_segmentation(labels, tracks)

def track_with_timeout(labels, radius, timeout_sec=30, tracking_config='particle_config.json'):
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        async_result = pool.apply_async(run_tracking, (labels, radius, tracking_config))
        try:
            return async_result.get(timeout=timeout_sec)
        except mp.context.TimeoutError:
            print(f"BTrack timed out after {timeout_sec}s. Using fallback.")
            return None
        except Exception as e:
            print(f"BTrack failed in worker: {e!r}. Using fallback.")
            return None

def track_one_T(labels: np.ndarray, scale: int, pts, radius: float = 5, threshold=60,
                tracked=None, use_same_img=False, tracking_config='particle_config.json'):
    radius = radius / scale
    if not use_same_img:
        tracked = track_with_timeout(labels, radius, timeout_sec=300,
                                     tracking_config=tracking_config)
        if tracked is None:
            # fallback: use raw segmentation with no tracking
            tracked = labels.copy()
            # tracked = np.stack([tracked]*2)  # fake 2 timepoints if needed

    pts = np.atleast_2d(pts)
    new_aim = []

    for pt in pts:
        pt = (np.array(pt) / scale).astype(int)
        label = tracked[0, pt[0], pt[1]]
        if label == 0:
            new_label = tracked[1, pt[0], pt[1]]
            if new_label == 0:
                ids = np.unique(tracked[1])
                ids = ids[ids != 0]
                if len(ids) == 0:
                    print(
                        f"lost tracking of a cell {pt * scale}, "
                        "current mask contains no cells; retaining original point"
                    )
                    new_aim.append((pt * scale).astype(int))
                    continue
                coms = np.asarray(
                    [center_of_mass(tracked[1] == label_id) for label_id in ids],
                    dtype=float,
                ).reshape(-1, 2)
                finite = np.isfinite(coms).all(axis=1)
                ids = ids[finite]
                coms = coms[finite]
                if len(coms) == 0:
                    print(
                        f"lost tracking of a cell {pt * scale}, "
                        "cell centers are invalid; retaining original point"
                    )
                    new_aim.append((pt * scale).astype(int))
                    continue
                dists = np.linalg.norm(coms - pt, axis=1)
                if len(dists) == 0 or np.min(dists) > 3 * threshold / scale:
                    print(f"lost tracking of a cell {pt * scale}, no potential cells within threshold")
                    new_aim.append((pt * scale).astype(int))
                else:
                    new_id = ids[np.argmin(dists)]
                    print(f"lost tracking of a cell {pt * scale}, moved to closest point within threshold")
                    new_aim.append((np.array(center_of_mass(tracked[1] == new_id)) * scale))
            else:
                print(f"maybe lost tracking of a cell {pt * scale}")
                new_aim.append((np.array(center_of_mass(tracked[1] == new_label)) * scale))
        elif np.sum(tracked[1] == label) != 0:
            new_aim.append((np.array(center_of_mass(tracked[1] == label)) * scale))
        else:
            new_aim.append((pt * scale).astype(int))

    return tracked, new_aim

def mask_outside_circle(img, circle_center=(540, 740), circle_radius=400):
    h, w = img.shape
    Y, X = np.ogrid[:h, :w]
    cy, cx = circle_center
    dist_sq = (Y - cy) ** 2 + (X - cx) ** 2
    mask = dist_sq <= circle_radius ** 2
    masked_img = img.copy()
    masked_img[~mask] = img.min()
    return masked_img

class CellposeSegmenter:
    """Reusable Cellpose segmentation with optional true ROI cropping.

    Returned masks keep the historical ``segment_single_img`` contract: their
    coordinates are the full camera coordinates divided by ``scale``.  This is
    required by ``track_one_T``, which applies the same scale to aiming points.
    """

    def __init__(self, cellpose_model="cyto2", gpu=False, model_factory=None):
        self.cellpose_model = str(cellpose_model)
        self.gpu = bool(gpu)
        self._model_factory = model_factory or Cellpose
        self._model = None

    @property
    def model(self):
        """Load the requested Cellpose model only on first use."""
        if self._model is None:
            self._model = self._model_factory(
                model_type=self.cellpose_model,
                gpu=self.gpu,
            )
        return self._model

    def segment(
        self,
        img: np.ndarray,
        scale: int = 4,
        crop=True,
        circle_center=(540, 740),
        circle_radius=100,
        crop_padding=4,
    ) -> np.ndarray:
        """Segment an image while preserving downscaled global coordinates."""
        img = np.asarray(img)
        if img.ndim != 2:
            raise ValueError("img must be a two-dimensional image")
        scale = int(scale)
        if scale < 1:
            raise ValueError("scale must be at least one")
        if circle_radius < 0:
            raise ValueError("circle_radius cannot be negative")
        if crop_padding < 0:
            raise ValueError("crop_padding cannot be negative")

        full_scaled_shape = tuple(
            np.maximum(1, np.round(np.asarray(img.shape) / scale).astype(int))
        )
        y0 = x0 = 0
        segment_img = img
        local_center = circle_center

        if crop:
            center = np.asarray(circle_center, dtype=float)
            if center.shape != (2,):
                raise ValueError("circle_center must contain one (y, x) pair")
            extent = float(circle_radius) + float(crop_padding)
            y0 = max(0, int(np.floor((center[0] - extent) / scale)) * scale)
            x0 = max(0, int(np.floor((center[1] - extent) / scale)) * scale)
            y1 = min(
                img.shape[0],
                int(np.ceil((center[0] + extent + 1) / scale)) * scale,
            )
            x1 = min(
                img.shape[1],
                int(np.ceil((center[1] + extent + 1) / scale)) * scale,
            )
            if y1 <= y0 or x1 <= x0:
                raise ValueError("The segmentation circle does not overlap the image")
            segment_img = img[y0:y1, x0:x1]
            local_center = (center[0] - y0, center[1] - x0)
            segment_img = mask_outside_circle(
                segment_img,
                circle_center=local_center,
                circle_radius=circle_radius,
            )

        scaled_img = rescale(segment_img, 1 / scale, anti_aliasing=True)
        value_range = float(scaled_img.max() - scaled_img.min())
        if value_range > 0:
            scaled_img = (scaled_img - scaled_img.min()) / value_range
        else:
            scaled_img = np.zeros_like(scaled_img, dtype=float)

        masks, _, _ = self.model.cp.eval(
            scaled_img,
            batch_size=1024,
            channels=[[0, 0]],
            diameter=50 / scale,
            flow_threshold=0.6,
            cellprob_threshold=-2,
            normalize=False,
        )
        masks = np.asarray(masks)
        if not crop:
            return masks

        full_mask = np.zeros(full_scaled_shape, dtype=masks.dtype)
        target_y0 = y0 // scale
        target_x0 = x0 // scale
        target_y1 = min(full_mask.shape[0], target_y0 + masks.shape[0])
        target_x1 = min(full_mask.shape[1], target_x0 + masks.shape[1])
        source_y1 = target_y1 - target_y0
        source_x1 = target_x1 - target_x0
        full_mask[target_y0:target_y1, target_x0:target_x1] = masks[
            :source_y1,
            :source_x1,
        ]
        return full_mask


_SEGMENTER_CACHE = {}


def segment_single_img(
    img: np.ndarray,
    scale: int = 4,
    crop=True,
    cellpose_model="cyto2",
    circle_center=(540, 740),
    circle_radius=100,
    segmenter=None,
    crop_padding=4,
):
    """Segment one image using a cached model and an actual circular ROI."""
    if segmenter is None:
        key = (str(cellpose_model), False)
        segmenter = _SEGMENTER_CACHE.get(key)
        if segmenter is None:
            segmenter = CellposeSegmenter(cellpose_model=cellpose_model, gpu=False)
            _SEGMENTER_CACHE[key] = segmenter
    return segmenter.segment(
        img,
        scale=scale,
        crop=crop,
        circle_center=circle_center,
        circle_radius=circle_radius,
        crop_padding=crop_padding,
    )


def find_com(img: np.ndarray, pt_xy: np.ndarray, scale: int=4, dist_thres: float=80, plot=False, cellpose_model='cyto2')->np.ndarray:
    """
    Find the COM of a moving object of interest by creating masks with cellpose.
    Return the closest match if no overlap is found, and return the input if the closest match is too far away.
    
    Parameters
    ----------
    pt_xy : (int, int)
        the coordinates of the scan point
    scale: int
        the downscaling scalar to improve segmentation speed
    dist_thres: float
        a distance threshold above which a point is not going to be considered to be matched with a label
    plot: bool
        Show a snapshot of the image and its cellposed mask alone with pt_xy
        
    Returns
    -------
    new_aim: (int, int)
        The new coordinates of the scan points
    """
    t_start = perf_counter()
    t0 = perf_counter()

    model = Cellpose(model_type = cellpose_model, gpu=False)
    channels = [[0, 0]]

    # seg_imgs = img[::scale, ::scale]
    seg_imgs = rescale(img, 1/scale, anti_aliasing=True)
    seg_imgs = (seg_imgs - seg_imgs.min())/(seg_imgs.max() - seg_imgs.min())

    masks, flow, styles = model.cp.eval(
        seg_imgs,
        batch_size=1024,
        channels=channels,
        diameter=40/scale,
        flow_threshold=0.6,
        cellprob_threshold=-2,
        normalize=False,
        tile=False,
        tile_overlap=0
    )
    
    pt_xy = np.atleast_2d(pt_xy)
    
    new_aim = []
    for pt in pt_xy:
        pt = (np.array(pt)/scale).astype(int)

        label = masks[pt[0], pt[1]]
        if label != 0:
            print('label found (exact match)')
            new_aim.append((np.array(ndi.center_of_mass(masks==label))*scale).astype(int))
        else:
            distances = []
            for i in range(1, masks.max()+1):
                distances.append(np.linalg.norm(pt - ndi.center_of_mass(masks==i)))
            if np.min(distances) <= dist_thres/scale:
                print('label found (closest match)')
                new_aim.append((np.array(ndi.center_of_mass(masks==np.argmin(distances)+1))*scale).astype(int))
            else:
                print('label not found')
                new_aim.append((pt*scale).astype(int))
                
    new_aim = np.asarray(new_aim)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
        ax[0].imshow(seg_imgs, cmap='gray')
        ax[0].scatter(pt_xy[:, 1]/scale, pt_xy[:, 0]/scale, c='r', marker='x')
        ax[1].imshow(masks)
        ax[1].scatter(pt_xy[:, 1]/scale, pt_xy[:, 0]/scale, c='r', marker='x')
        ax[1].scatter(new_aim[:, 1]/scale, new_aim[:, 0]/scale, c='k', marker='x')
            
    return new_aim

# from qtpy.QtCore import QTimer, QEventLoop

# def run_on_main_thread(fn):
#     loop = QEventLoop()
#     def wrapped():
#         try:
#             fn()
#         finally:
#             loop.quit()
#     QTimer.singleShot(0, wrapped)
#     loop.exec_()  # blocks until loop.quit() is called
   
def update_pos_points(P, new_pts, points_layer, p_idx=1):
    """
    Parameters
    ----------
    P : int
        The current position step
    new_pts : (N, 2) array
        the new points to set
    points_layer : poitns layer
        the layer to update
    p_idx : int
        Which dimension is position
    """
    new = np.copy(points_layer.data)
    # new[:, -1] = 600
    new[new[:, p_idx] == P,-2:] = new_pts
    points_layer.data = new
    # run_on_main_thread(lambda: setattr(points_layer, 'data', new))
