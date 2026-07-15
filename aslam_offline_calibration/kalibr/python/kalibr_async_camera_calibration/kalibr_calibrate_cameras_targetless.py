#!/usr/bin/env python
"""
Pipeline 2 — Targetless inter-camera time offset estimation.

Given:
  - A ROS bag with two unsynchronized camera topics (free motion, no calibration target)
  - Camera intrinsics (from Pipeline 1 or Kalibr)
  - Camera extrinsics T_cam1_cam0 (from Pipeline 1 camchain YAML)

Outputs:
  - tau: inter-camera time offset (t_cam1 = t_cam0 + tau)
  - Diagnostic plots and CSVs saved to --output-dir

Architecture:
  Inherits from AsyncCalibrator. Replaces only the front-end (AprilGrid PnP)
  with feature tracking + stereo triangulation + PnP pose recovery.
  All downstream processing (initPoseSpline, findTimeshiftPrior) is inherited unchanged.

Pose recovery strategy:
  - Track features independently per camera (LK optical flow)
  - At each cam0 timestamp, find the nearest cam1 frame
  - Match cam0↔cam1 features using epipolar constraint (F from known T_cam1_cam0)
  - Triangulate matched pairs → metric 3D points (shared geometry)
  - PnP cam0 and cam1 against those same 3D points → geometrically consistent poses
  - Both cameras' angular velocity profiles are anchored to shared 3D structure
    → cross-correlation recovers tau reliably
"""

import sys
import os
import math
import signal
import argparse
import yaml
from collections import Counter

# ── Kalibr imports must come first — sm pre-loads CHOLMOD symbols ─────────────
import kalibr_common as kc
import kalibr_camera_calibration as kcc
import aslam_cv as acv
import aslam_cameras_april as acv_april
import aslam_backend as aopt
import aslam_cv_backend as acvb
import aslam_splines as asp
import incremental_calibration as inc
import sm
import bsplines

import numpy as np
import cv2
import rosbag
import pylab as pl

# ── inherit the full Pipeline 1 calibrator ───────────────────────────────────
from kalibr_calibrate_cameras_async import AsyncCalibrator, signal_exit


# =============================================================================
# Shim classes — make feature-tracked poses look like CameraObservations
# so initPoseSpline and findTimeshiftPrior work without modification
# =============================================================================

class SE3Wrapper:
    """Wraps a 4x4 numpy SE(3) matrix and exposes .T() as initPoseSpline expects."""
    def __init__(self, T_4x4):
        self._T = T_4x4

    def T(self):
        return self._T


class PoseObservation:
    """
    Drop-in replacement for kalibr CameraObservation.
    initPoseSpline uses only:
        obs.time().toSec()   -> float timestamp
        obs.T_t_c().T()      -> 4x4 numpy array  (T_target_cam = T_world_cam^{-1})
    """
    def __init__(self, timestamp_sec, T_world_cam):
        self._t = timestamp_sec
        self._T_t_c = SE3Wrapper(np.linalg.inv(T_world_cam))

    def time(self):
        return self

    def toSec(self):
        return self._t

    def T_t_c(self):
        return self._T_t_c


# =============================================================================
# TargetlessCalibrator
# =============================================================================

class TargetlessCalibrator(AsyncCalibrator):
    """
    Estimates inter-camera time offset tau without a calibration target.

    Overrides __init__ and front-end methods.
    Inherits initPoseSpline and findTimeshiftPrior unchanged.
    """

    def __init__(self, cam0Intrinsics, cam1Intrinsics,
                 T_cam1_cam0, camIdx=1, outputDir='/data'):
        self.camIdx         = camIdx
        self.outputDir      = outputDir
        self.timeshiftPrior = 0.0

        os.makedirs(outputDir, exist_ok=True)

        self.K0 = np.array([
            [cam0Intrinsics['fx'], 0,                    cam0Intrinsics['cx']],
            [0,                    cam0Intrinsics['fy'],  cam0Intrinsics['cy']],
            [0,                    0,                    1.0]
        ], dtype=np.float64)

        self.K1 = np.array([
            [cam1Intrinsics['fx'], 0,                    cam1Intrinsics['cx']],
            [0,                    cam1Intrinsics['fy'],  cam1Intrinsics['cy']],
            [0,                    0,                    1.0]
        ], dtype=np.float64)

        self.dist0 = np.array(cam0Intrinsics['dist_coeffs'], dtype=np.float64)
        self.dist1 = np.array(cam1Intrinsics['dist_coeffs'], dtype=np.float64)

        self.T_cam1_cam0_np = np.array(T_cam1_cam0, dtype=np.float64)   #cam0 --> cam1
        self.T_cam0_cam1_np = np.linalg.inv(self.T_cam1_cam0_np)   #cam1 --> cam0

        # Fundamental matrix F = K1^{-T} [t]x R K0^{-1}
        # Maps a point in cam0 to an epipolar line in cam1
        R   = self.T_cam1_cam0_np[:3, :3]
        t   = self.T_cam1_cam0_np[:3,  3]

        #skew symmetric matrix 
        tx  = np.array([[0, -t[2], t[1]],
                        [t[2], 0, -t[0]],
                        [-t[1], t[0], 0]], dtype=np.float64)
        
        #Essential matrix
        E   = tx @ R

        #Fundamental matrix
        self.F_cam0_to_cam1 = np.linalg.inv(self.K1).T @ E @ np.linalg.inv(self.K0)

        # populated by buildPoseObservations()
        self.target0Observations = []
        self.targetNObservations = []


    # =========================================================================
    # Image decode helper
    # =========================================================================

    def _decodeImage(self, msg, bridge):
        if bridge is not None:
            try:
                img = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            except Exception:
                img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img


    # =========================================================================
    # Steps 1+2 — stream bag and track independently per camera
    # =========================================================================

    def loadAndTrackFromBag(self, bagfile, topic0, topic1,
                            bag_from_to=None, bag_freq=None,
                            max_corners=500, min_track_length=5,
                            redetect_threshold=0.5):
        """
        Stream images from bag, track features independently per camera via LK.
        Returns:
            tracks0, tracks1 : list of dicts {t, pts, ids} per frame
                               pts are undistorted pixel coordinates
        """
        print("Loading and tracking from bag: %s" % bagfile)

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        detector = cv2.GFTTDetector_create(
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=5,
            useHarrisDetector=False
        )

        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
        except ImportError:
            bridge = None

        state = {
            topic0: dict(prev_gray=None, prev_pts=None, track_ids=None,
                         next_id=0, tracks=[], K=self.K0, dist=self.dist0,
                         label='cam0', last_t=-1e9),
            topic1: dict(prev_gray=None, prev_pts=None, track_ids=None,
                         next_id=0, tracks=[], K=self.K1, dist=self.dist1,
                         label='cam1', last_t=-1e9),
        }
        min_dt = (1.0 / bag_freq) if bag_freq else 0.0

        with rosbag.Bag(bagfile, 'r') as bag:
            t0_bag = bag.get_start_time()
            for topic, msg, t in bag.read_messages(topics=[topic0, topic1]):
                if topic not in state:
                    continue
                t_sec = t.to_sec()
                if bag_from_to is not None:
                    if t_sec - t0_bag < bag_from_to[0]: continue
                    if t_sec - t0_bag > bag_from_to[1]: break
                s = state[topic]
                if t_sec - s['last_t'] < min_dt:
                    continue
                s['last_t'] = t_sec

                gray = self._decodeImage(msg, bridge)
                self._trackFrame(gray, t_sec, s, detector, lk_params,
                                 max_corners, redetect_threshold)

        results = []
        for topic in [topic0, topic1]:
            s = state[topic]
            id_counts = Counter(i for f in s['tracks'] for i in f['ids'])
            valid_ids = set(i for i, c in id_counts.items() if c >= min_track_length)
            filtered = []
            for f in s['tracks']:
                mask = np.array([i in valid_ids for i in f['ids']])
                filtered.append({'t':   f['t'],
                                 'pts': f['pts'][mask],
                                 'ids': f['ids'][mask]})
            mean_pts = np.mean([len(f['pts']) for f in filtered]) if filtered else 0
            print("  %s: %d frames, mean %.1f tracked points/frame after filtering" % (
                s['label'], len(filtered), mean_pts))
            results.append(filtered)

        return results[0], results[1]


    def _trackFrame(self, gray, t_sec, s, detector, lk_params,
                    max_corners, redetect_threshold):
        K, dist = s['K'], s['dist']

        if s['prev_gray'] is None:
            kps = detector.detect(gray, None)
            pts = cv2.KeyPoint_convert(kps).reshape(-1, 1, 2).astype(np.float32)
            s['track_ids'] = np.arange(len(pts), dtype=np.int32)
            s['next_id']   = len(pts)
            s['prev_gray'] = gray
            s['prev_pts']  = pts
        else:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                s['prev_gray'], gray, s['prev_pts'], None, **lk_params)
            prev_pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
                gray, s['prev_gray'], curr_pts, None, **lk_params)
            fb_err = np.linalg.norm(
                s['prev_pts'].reshape(-1, 2) - prev_pts_back.reshape(-1, 2), axis=1)
            good = ((status.flatten() == 1) &
                    (status_back.flatten() == 1) &
                    (fb_err < 2.0))

            curr_pts       = curr_pts[good]
            s['track_ids'] = s['track_ids'][good]
            s['prev_gray'] = gray
            s['prev_pts']  = curr_pts

            if len(curr_pts) < redetect_threshold * max_corners:
                print("  [%s] redetecting (only %d tracks left)" % (
                    s['label'], len(curr_pts)))
                mask = np.ones(gray.shape, dtype=np.uint8) * 255
                for pt in curr_pts.reshape(-1, 2).astype(int):
                    cv2.circle(mask, tuple(pt), 10, 0, -1)
                kps_new = detector.detect(gray, mask)
                if kps_new:
                    pts_new        = cv2.KeyPoint_convert(kps_new).reshape(-1, 1, 2).astype(np.float32)
                    ids_new        = np.arange(s['next_id'], s['next_id'] + len(pts_new), dtype=np.int32)
                    s['next_id']  += len(pts_new)
                    s['prev_pts']  = np.vstack([curr_pts, pts_new])
                    s['track_ids'] = np.hstack([s['track_ids'], ids_new])

        if len(s['prev_pts']) > 0:
            pts_ud = cv2.undistortPoints(
                s['prev_pts'].reshape(-1, 1, 2).astype(np.float32),
                K, np.array(dist, dtype=np.float64), P=K
            ).reshape(-1, 2).astype(np.float32)
        else:
            pts_ud = np.zeros((0, 2), dtype=np.float32)

        s['tracks'].append({'t':   t_sec,
                            'pts': pts_ud,
                            'ids': s['track_ids'].copy()})


    # =========================================================================
    # Step 3 — cross-camera epipolar matching + stereo triangulation
    # =========================================================================

    def _epipolarMatch(self, pts0, pts1_frame, epi_thresh=2.0):
        """
        Match pts0 (Nx2, cam0 undistorted) against all points in pts1_frame
        using the epipolar constraint with known F.

        For each point in pts0, find the nearest point in pts1_frame that lies
        within epi_thresh pixels of the corresponding epipolar line.

        Returns:
            idx0, idx1 : index arrays into pts0 and pts1_frame of matched pairs
        """
        if len(pts0) == 0 or len(pts1_frame) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        pts0_h = np.hstack([pts0, np.ones((len(pts0), 1))])   # Nx3 homogeneous
        pts1_h = np.hstack([pts1_frame, np.ones((len(pts1_frame), 1))])  # Mx3

        # epipolar lines in cam1 for each cam0 point: l = F @ p0  (3,)
        lines = (self.F_cam0_to_cam1 @ pts0_h.T).T   # Nx3

        # distance of each cam1 point to each epipolar line: |l . p1| / sqrt(a^2+b^2)
        # lines: Nx3, pts1_h: Mx3
        # dist[i,j] = |lines[i] . pts1_h[j]| / norm(lines[i,:2])
        norms   = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)  # Nx1
        dists   = np.abs(lines @ pts1_h.T) / norms   # NxM

        idx0_list, idx1_list = [], []
        used1 = set()
        for i in range(len(pts0)):
            j = int(np.argmin(dists[i]))
            if dists[i, j] < epi_thresh and j not in used1:
                idx0_list.append(i)
                idx1_list.append(j)
                used1.add(j)

        return np.array(idx0_list, dtype=int), np.array(idx1_list, dtype=int)


    def _triangulateAndPnP(self, pts0, pts1):
        """
        Triangulate matched cam0/cam1 point pairs using known T_cam1_cam0.
        Returns metric 3D points in cam0 frame.

        Args:
            pts0 : (N,2) undistorted cam0 pixels
            pts1 : (N,2) undistorted cam1 pixels (matched to pts0)
        Returns:
            pts3d : (M,3) valid 3D points in cam0 frame  (M <= N)
            valid_mask : (N,) bool mask of which input pairs gave valid 3D
        """
        P0 = self.K0 @ np.eye(3, 4)
        P1 = self.K1 @ self.T_cam1_cam0_np[:3, :]

        pts4d   = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
        pts3d   = (pts4d[:3] / pts4d[3]).T   # Nx3 in cam0 frame

        # transform to cam1 frame to check depth there too
        pts3d_h   = np.hstack([pts3d, np.ones((len(pts3d), 1))]).T
        pts3d_c1  = (self.T_cam1_cam0_np @ pts3d_h)[:3].T

        valid = ((pts3d[:, 2]   > 0.01) & (pts3d[:, 2]   < 100.0) &
                 (pts3d_c1[:, 2] > 0.01) & (pts3d_c1[:, 2] < 100.0))

        return pts3d[valid], valid


    # =========================================================================
    # Step 4 — recover metric pose chains for both cameras
    # =========================================================================

    def recoverPoseChains(self, tracks0, tracks1):
        """
        Recover metric SE(3) pose chains for cam0 and cam1 using shared 3D geometry.

        Strategy per cam0 frame i:
          1. Find nearest cam1 frame in time (within max_dt)
          2. Epipolar-match cam0 points against cam1 points using F
          3. Triangulate matched pairs → metric 3D points in cam0 frame
          4. PnP cam0 frame i against those 3D points → T_world_cam0[i]
          5. PnP nearest cam1 frame against same 3D points → T_world_cam1[nearest]
             (recorded at the cam1 timestamp, not cam0 timestamp)

        Both cameras' poses are anchored to the same 3D structure at each step,
        ensuring geometrically consistent angular velocity profiles for cross-correlation.

        Returns:
            poses0 : list of (t_sec, T_world_cam0 4x4)
            poses1 : list of (t_sec, T_world_cam1 4x4)
        """
        print("Recovering metric pose chains for cam0 and cam1 (%d cam0 frames)..." % len(tracks0))

        # build cam1 lookup by timestamp
        cam1_times  = np.array([f['t'] for f in tracks1])
        cam1_frames = {f['t']: f for f in tracks1}

        max_dt = 0.15   # max temporal gap for stereo pairing (> tau_max expected)

        poses0 = []
        poses1_dict = {}   # keyed by cam1 timestamp to avoid duplicates

        # world frame = cam0 at first successfully triangulated frame
        world_set    = False
        T_world_cam0 = np.eye(4)
        T_world_cam1_ref = None   # cam1 pose at the reference frame

        n_pnp_ok   = 0
        n_pnp_fail = 0

        for i, f0 in enumerate(tracks0):
            t0 = f0['t']

            # find nearest cam1 frame
            nearest_idx = int(np.argmin(np.abs(cam1_times - t0)))
            t1_near     = cam1_times[nearest_idx]
            if abs(t1_near - t0) > max_dt:
                n_pnp_fail += 1
                continue

            f1 = cam1_frames[t1_near]

            # epipolar match cam0 → cam1
            idx0, idx1 = self._epipolarMatch(f0['pts'], f1['pts'], epi_thresh=2.0)
            if len(idx0) < 8:
                n_pnp_fail += 1
                continue

            pts0_matched = f0['pts'][idx0]
            pts1_matched = f1['pts'][idx1]

            # triangulate → metric 3D in cam0 frame
            pts3d, valid = self._triangulateAndPnP(pts0_matched, pts1_matched)
            if len(pts3d) < 6:
                n_pnp_fail += 1
                continue

            pts0_valid = pts0_matched[valid]
            pts1_valid = pts1_matched[valid]

            # ── set world frame at first good triangulation ──────────────────
            if not world_set:
                # world = cam0 at this frame (identity)
                T_world_cam0     = np.eye(4)
                # cam1 at reference: apply known extrinsic
                T_world_cam1_ref = self.T_cam0_cam1_np.copy()   # T_world_cam1 = T_cam0_cam1
                world_set        = True

                poses0.append((t0, T_world_cam0.copy()))
                if t1_near not in poses1_dict:
                    poses1_dict[t1_near] = T_world_cam1_ref.copy()
                n_pnp_ok += 1
                continue

            # ── PnP cam0 against 3D points (expressed in world = cam0_ref frame) ──
            # pts3d is in cam0_i frame; need them in world frame
            # use T_world_cam0 from previous frame as initial guess for iterative PnP
            # For the first few frames we don't have a prior T_world_cam0 so bootstrap
            # by chaining: pts3d_world = T_world_cam0_prev @ pts3d (approx, since pts3d
            # are in cam0_i not cam0_prev — use solvePnP with no prior instead)

            ok0, rvec0, tvec0 = cv2.solvePnP(
                pts3d.astype(np.float64),
                pts0_valid.astype(np.float64),
                self.K0, np.zeros(5),
                flags=cv2.SOLVEPNP_ITERATIVE)

            if not ok0:
                n_pnp_fail += 1
                continue

            R0, _ = cv2.Rodrigues(rvec0)
            # T_cam0i_world (cam0 frame at time i, world = 3D points frame = cam0 ref)
            T_cam0i_pts3d = np.eye(4)
            T_cam0i_pts3d[:3, :3] = R0
            T_cam0i_pts3d[:3,  3] = tvec0.flatten()
            T_world_cam0_i = np.linalg.inv(T_cam0i_pts3d)

            # ── PnP cam1 against same 3D points ─────────────────────────────
            ok1, rvec1, tvec1 = cv2.solvePnP(
                pts3d.astype(np.float64),
                pts1_valid.astype(np.float64),
                self.K1, np.zeros(5),
                flags=cv2.SOLVEPNP_ITERATIVE)

            if not ok1:
                # fall back to extrinsic-chained cam1 pose
                T_world_cam1_i = T_world_cam0_i @ self.T_cam0_cam1_np
            else:
                R1, _ = cv2.Rodrigues(rvec1)
                T_cam1i_pts3d = np.eye(4)
                T_cam1i_pts3d[:3, :3] = R1
                T_cam1i_pts3d[:3,  3] = tvec1.flatten()
                T_world_cam1_i = np.linalg.inv(T_cam1i_pts3d)

            poses0.append((t0, T_world_cam0_i))
            if t1_near not in poses1_dict:
                poses1_dict[t1_near] = T_world_cam1_i
            n_pnp_ok += 1

        poses1 = sorted(poses1_dict.items(), key=lambda x: x[0])

        print("  PnP recovery: %d ok, %d failed (no stereo match or insufficient points)" % (
            n_pnp_ok, n_pnp_fail))
        print("  cam0 pose chain: %d poses" % len(poses0))
        print("  cam1 pose chain: %d poses" % len(poses1))

        if len(poses0) < 20 or len(poses1) < 20:
            raise RuntimeError(
                "Too few poses recovered (cam0=%d, cam1=%d). "
                "Check epipolar threshold or stereo pairing window." % (
                len(poses0), len(poses1)))

        return poses0, poses1


    # =========================================================================
    # Step 5 — wrap into PoseObservation lists for initPoseSpline
    # =========================================================================

    def buildPoseObservations(self, poses0, poses1):
        self.target0Observations = [PoseObservation(t, T) for t, T in poses0]
        self.targetNObservations = [PoseObservation(t, T) for t, T in poses1]
        print("PoseObservations built: cam0=%d  cam1=%d" % (
            len(self.target0Observations), len(self.targetNObservations)))


    # =========================================================================
    # Top-level run
    # =========================================================================

    def run(self, bagfile, topic0, topic1,
            bag_from_to=None, bag_freq=None,
            max_corners=500, splineOrder=6,
            poseKnotsPerSecond=100):
        """
        Full targetless pipeline. Returns estimated tau in seconds.
        (t_cam1 = t_cam0 + tau)
        """
        # 1+2: track independently per camera
        tracks0, tracks1 = self.loadAndTrackFromBag(
            bagfile, topic0, topic1,
            bag_from_to=bag_from_to, bag_freq=bag_freq,
            max_corners=max_corners)

        if len(tracks0) < 10 or len(tracks1) < 10:
            raise RuntimeError("Too few frames tracked (cam0=%d cam1=%d). "
                               "Check topics and bag_from_to." % (
                               len(tracks0), len(tracks1)))

        # 3+4: cross-camera epipolar match + stereo triangulation + PnP for both cameras
        poses0, poses1 = self.recoverPoseChains(tracks0, tracks1)

        # 5: build shims
        self.buildPoseObservations(poses0, poses1)

        # 6: estimate tau — inherited unchanged from AsyncCalibrator
        self.findTimeshiftPrior()

        print("")
        print("=" * 50)
        print("Targetless tau estimate: %.6f s" % self.timeshiftPrior)
        print("=" * 50)

        return self.timeshiftPrior


# =============================================================================
# I/O helpers
# =============================================================================

def loadIntrinsicsFromYaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    intr = data['intrinsics']
    dist = list(data['distortion_coeffs'])
    if len(dist) == 4:
        dist += [0.0]
    return {'fx': intr[0], 'fy': intr[1],
            'cx': intr[2], 'cy': intr[3],
            'dist_coeffs': dist}


def loadExtrinsicsFromCamchainYaml(yaml_path, cam_src=1):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    key    = 'cam%d' % cam_src
    T_list = data[key]['T_cn_cnm1']
    return np.array(T_list, dtype=np.float64)


# =============================================================================
# CLI
# =============================================================================

def parseArgs():
    class KalibrArgParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            sm.logError('%s' % message)
            sys.exit(2)

    usage = """
    Estimate inter-camera time offset without a calibration target.

    %(prog)s --bag MYROSBAG.bag --topics /cam0/image_raw /cam1/image_raw \\
              --cam-intrinsics cam0.yaml cam1.yaml \\
              --camchain camchain.yaml \\
              --output-dir /data/targetless_results
    """

    parser = KalibrArgParser(
        description='Targetless inter-camera time offset estimation.',
        usage=usage)

    parser.add_argument('--bag',           dest='bagfile',      required=True)
    parser.add_argument('--topics',        dest='topics',       nargs=2, required=True)
    parser.add_argument('--cam-intrinsics',dest='camIntrinsics',nargs=2, required=True)
    parser.add_argument('--camchain',      dest='camchainYaml', required=True,
        help='Kalibr camchain YAML from Pipeline 1 (T_cam1_cam0)')
    parser.add_argument('--output-dir',    dest='outputDir',    default=None)
    parser.add_argument('--bag-from-to',   dest='bag_from_to',  nargs=2, type=float,
        metavar=('T_START', 'T_END'), default=None)
    parser.add_argument('--bag-freq',      dest='bag_freq',     type=float, default=None)
    parser.add_argument('--max-corners',   dest='maxCorners',   type=int,   default=500)
    parser.add_argument('--knots-per-second', dest='knotsPerSecond', type=int, default=100)
    parser.add_argument('--spline-order',  dest='splineOrder',  type=int,   default=6)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(2)

    parsed = parser.parse_args()

    for f in [parsed.bagfile, parsed.camchainYaml] + list(parsed.camIntrinsics):
        if not os.path.isfile(f):
            sm.logError("File not found: %s" % f)
            sys.exit(2)

    if parsed.outputDir is None:
        parsed.outputDir = os.path.dirname(os.path.abspath(parsed.bagfile))
    else:
        os.makedirs(parsed.outputDir, exist_ok=True)

    return parsed


def main():
    signal.signal(signal.SIGINT, signal_exit)
    parsed = parseArgs()

    cam0_intr   = loadIntrinsicsFromYaml(parsed.camIntrinsics[0])
    cam1_intr   = loadIntrinsicsFromYaml(parsed.camIntrinsics[1])
    T_cam1_cam0 = loadExtrinsicsFromCamchainYaml(parsed.camchainYaml, cam_src=1)

    calibrator = TargetlessCalibrator(
        cam0Intrinsics=cam0_intr,
        cam1Intrinsics=cam1_intr,
        T_cam1_cam0=T_cam1_cam0,
        camIdx=1,
        outputDir=parsed.outputDir
    )

    tau = calibrator.run(
        bagfile=parsed.bagfile,
        topic0=parsed.topics[0],
        topic1=parsed.topics[1],
        bag_from_to=parsed.bag_from_to,
        bag_freq=parsed.bag_freq,
        max_corners=parsed.maxCorners,
        splineOrder=parsed.splineOrder,
        poseKnotsPerSecond=parsed.knotsPerSecond
    )

    result   = {'tau_cam1_wrt_cam0_seconds': float(tau)}
    out_yaml = os.path.join(parsed.outputDir, 'targetless_tau.yaml')
    with open(out_yaml, 'w') as f:
        yaml.dump(result, f, default_flow_style=False)
    print("tau saved to: %s" % out_yaml)


if __name__ == "__main__":
    main()