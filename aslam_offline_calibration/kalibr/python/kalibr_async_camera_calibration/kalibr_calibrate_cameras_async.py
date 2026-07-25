import kalibr_common as kc
import kalibr_camera_calibration as kcc
import aslam_cv as acv
import aslam_cameras_april as acv_april
import aslam_backend as aopt
import aslam_cv_backend as acvb
import aslam_splines as asp
import incremental_calibration as inc
import numpy as np
import sm
import sys
import bsplines
import signal
import math
import pylab as pl
import multiprocessing
import os
import yaml

from helpers.read_dataset import initBagDataset
from helpers.cli import parseArgs
from helpers.geometry import rotation_matrix_to_rotvec

#constants for grouping variables in optimization and helper variables 
CALIBRATION_GROUP_ID = 0
HELPER_GROUP_ID = 1

#for interupting pipeline 
def signal_exit(signal, frame):
    sm.logWarn("Shutdown requested! (CTRL+C)")
    sys.exit(2)

class AsyncCalibrator():
    # camera config, target config, dataset
    def __init__(self, cam0Config, camNConfig, camIdx, targetConfig, dataset0, datasetN, reprojectionSigma=1.0, showCorners=True, showReproj=True, showOneStep=False, outputDir='/data'):
        self.targetConfig = targetConfig

        self.cornerUncertainty = reprojectionSigma

        #set the extrinsic prior to default
        self.T_camN_cam0 = sm.Transformation()   #T_dest_src notation

        #initialize timeshift prior to zero
        self.timeshiftPrior = 0.0

        self.camIdx = camIdx

        ## camera 0
        self.cam0Config = cam0Config
        self.dataset0 = dataset0
        self.camera0 = kc.AslamCamera.fromParameters( self.cam0Config )   #load camera model from yaml
        self.setupCalibrationTarget( targetConfig, self.camera0, showExtraction=showCorners, showReproj=showReproj, imageStepping=showOneStep )
        multithreading = not (showCorners or showReproj or showOneStep)  #parallel processing for corner extraction if not visualizing
        self.cam0_detector = self.detector
        self.target0Observations = kc.extractCornersFromDataset(self.dataset0, self.cam0_detector, multithreading=multithreading) #T_t_c0 pose (target pose relative to cam0 frame thru PnP)

        ## camera 1
        self.camNConfig = camNConfig
        self.datasetN = datasetN
        self.cameraN = kc.AslamCamera.fromParameters( self.camNConfig )
        self.setupCalibrationTarget( targetConfig, self.cameraN, showExtraction=showCorners, showReproj=showReproj, imageStepping=showOneStep )
        multithreading = not (showCorners or showReproj or showOneStep)
        self.camN_detector = self.detector
        self.targetNObservations = kc.extractCornersFromDataset(self.datasetN, self.camN_detector, multithreading=multithreading)  #T_t_c1 pose (target pose relative to cam1 frame thru PnP)

        print("Cam0: %d/%d images had valid detections (detector + PnP)" % (len(self.target0Observations), self.dataset0.numImages()))
        print("Cam%d: %d/%d images had valid detections (detector + PnP)" % (self.camIdx, len(self.targetNObservations), self.datasetN.numImages()))

        # problem
        # self.problem = aopt.OptimizationProblem()
        self.problem = inc.CalibrationOptimizationProblem()

        # for camchain AsyncCalibrator.__init__
        self.cameras = [self.camera0, self.cameraN]

        self.outputDir = outputDir

    # from IccSensors.py --> class IccCamera 
    def setupCalibrationTarget(self, targetConfig, camera, showExtraction=False, showReproj=False, imageStepping=False):
        
        #load the calibration target configuration
        targetParams = targetConfig.getTargetParams()
        targetType = targetConfig.getTargetType()
    
        if targetType == 'checkerboard':
            options = acv.CheckerboardOptions() 
            options.filterQuads = True
            options.normalizeImage = True
            options.useAdaptiveThreshold = True        
            options.performFastCheck = False
            options.windowWidth = 5
            options.showExtractionVideo = showExtraction
            grid = acv.GridCalibrationTargetCheckerboard(targetParams['targetRows'], 
                                                            targetParams['targetCols'], 
                                                            targetParams['rowSpacingMeters'], 
                                                            targetParams['colSpacingMeters'],
                                                            options)
        elif targetType == 'circlegrid':
            options = acv.CirclegridOptions()
            options.showExtractionVideo = showExtraction
            options.useAsymmetricCirclegrid = targetParams['asymmetricGrid']
            grid = acv.GridCalibrationTargetCirclegrid(targetParams['targetRows'],
                                                          targetParams['targetCols'], 
                                                          targetParams['spacingMeters'], 
                                                          options)
        elif targetType == 'aprilgrid':
            options = acv_april.AprilgridOptions() 
            options.showExtractionVideo = showExtraction
            options.minTagsForValidObs = int( np.max( [targetParams['tagRows'], targetParams['tagCols']] ) + 1 )
            
            grid = acv_april.GridCalibrationTargetAprilgrid(targetParams['tagRows'],
                                                            targetParams['tagCols'], 
                                                            targetParams['tagSize'], 
                                                            targetParams['tagSpacing'], 
                                                            options)
        else:
            raise RuntimeError( "Unknown calibration target." )
                          
        options = acv.GridDetectorOptions() 
        options.imageStepping = imageStepping
        options.plotCornerReprojection = showReproj
        options.filterCornerOutliers = True
        options.filterCornerSigmaThreshold = 2.0
        options.filterCornerMinReprojError = 0.2
        self.detector = acv.GridDetector(camera.geometry, grid, options)     

    #initialize a pose spline using camera poses (pose spline = T_wb)
    def initPoseSpline(self, targetObservations, splineOrder=6, poseKnotsPerSecond=200, timeOffsetPadding=0.02, label='cam'):
        '''
        poseKnotsPerSecond: number of knots per second for the pose spline (200 => 20ms)
        Eg:
            T_t_c: target pose relative to cam frame (from PnP)
            (T_t_c).T(): cam0 poses w.r.t target frame, for building spline (discrete poses)
        '''
        pose = bsplines.BSplinePose(splineOrder, sm.RotationVector() )
                
        # Get the grid detected times
        cam_times = np.array([obs.time().toSec() for obs in targetObservations])       #cam detection discrete timestamps in seconds      
        cam_curves = np.matrix([ pose.transformationToCurveValue(obs.T_t_c().T()) for obs in targetObservations]).T   #4x4 transformation matrix --> 6xN axis angle representation [tx, ty, tz, rx, ry, rz], each column = discrete pose 
        
        if np.isnan(cam_curves).any():
            raise RuntimeError("Nans in cam_pose values for initPoseSpline")
            sys.exit(0)
        
        ''' 
        Add 2 seconds on either end to allow the spline to slide during optimization

        time:
            before: t[0], t[1], t[2], ..., t[N]
            after:  t[0]-0.04, t[0], t[1], t[2], ..., t[N], t[N]+0.04

        curve values:
            before: pose[0], pose[1], pose[2], ..., pose[N]
            after:  pose[0], pose[0], pose[1], pose[2], ..., pose[N], pose[N]
       
        '''
        cam_times = np.hstack((cam_times[0] - (timeOffsetPadding * 2.0), cam_times, cam_times[-1] + (timeOffsetPadding * 2.0)))   #timeoffsetpadding(0.02) * 2 seconds = 0.04 = 40 ms on either end
        cam_curves = np.hstack((cam_curves[:,0], cam_curves, cam_curves[:,-1]))
        
        # Make sure the rotation vector doesn't flip
        for i in range(1,cam_curves.shape[1]):
            previousRotationVector = cam_curves[3:6,i-1]
            r = cam_curves[3:6,i]
            angle = np.linalg.norm(r)
            axis = r/angle
            best_r = r
            best_dist = np.linalg.norm( best_r - previousRotationVector)
            
            for s in range(-3,4):
                aa = axis * (angle + math.pi * 2.0 * s)   #angle wrapping --> axis*(angle + 2pi*s)
                dist = np.linalg.norm( aa - previousRotationVector ) 
                if dist < best_dist:
                    best_r = aa
                    best_dist = dist
            cam_curves[3:6,i] = best_r;      
        
        # discrete cam0 poses from PnP 
        np.savetxt(os.path.join(self.outputDir, 'poses_discrete_%s.csv' % label), 
            np.hstack((cam_times.reshape(-1,1), cam_curves.T)), 
            delimiter=',', 
            header='t,tx,ty,tz,rx,ry,rz')

        # Fitting the spline    
        seconds = cam_times[-1] - cam_times[0]
        knots = int(round(seconds * poseKnotsPerSecond))
        
        print("")
        print("Initializing a pose spline with %d knots (%f knots per second over %f seconds)" % ( knots, poseKnotsPerSecond, seconds))
        pose.initPoseSplineSparse(cam_times, cam_curves, knots, 1e-4)
        
        # Dense sampling to visualize smoothness
        dense_times = np.linspace(cam_times[0], cam_times[-1], 5000)
        dense_curves = np.array([pose.eval(t) for t in dense_times]).T
        np.savetxt(os.path.join(self.outputDir, 'poses_spline_dense_%s.csv' % label),
                np.hstack((dense_times.reshape(-1,1), dense_curves.T)),
                delimiter=',',
                header='t,tx,ty,tz,rx,ry,rz')
        
        return pose   
    
    def findTimeshiftPrior(self, verbose=False):
        print("Estimating time shift camera %d to camera 0:" % self.camIdx)
        
        poseSplineCam0 = self.initPoseSpline(self.target0Observations, timeOffsetPadding=0.0, label='cam0_prior')
        poseSplineCamN = self.initPoseSpline(self.targetNObservations, timeOffsetPadding=0.0, label='cam%d_prior' % self.camIdx)

        t_start_cam0 = poseSplineCam0.t_min()
        t_end_cam0   = poseSplineCam0.t_max()
        t_start_camN = poseSplineCamN.t_min()
        t_end_camN   = poseSplineCamN.t_max()

        print("  [diag] cam0 spline range : %.6f  to  %.6f  (%.3fs)" % (
            t_start_cam0, t_end_cam0, t_end_cam0 - t_start_cam0))
        print("  [diag] cam%d spline range : %.6f  to  %.6f  (%.3fs)" % (
            self.camIdx, t_start_camN, t_end_camN, t_end_camN - t_start_camN))

        t_start = max(t_start_cam0, t_start_camN)
        t_end   = min(t_end_cam0,   t_end_camN)
        overlap = t_end - t_start
        print("  [diag] overlap window    : %.6f  to  %.6f  (%.3fs)" % (t_start, t_end, overlap))

        dT = 0.01
        uniform_times = np.arange(t_start + dT, t_end - dT, dT)
        effective_dT  = uniform_times[1] - uniform_times[0] if len(uniform_times) > 1 else float('nan')
        print("  [diag] uniform grid      : %d samples,  dT_nominal=%.4fs,  dT_effective=%.6fs" % (
            len(uniform_times), dT, effective_dT))

        # ── CHANGE 1: collect full 3D angular velocity vectors, not just norms ──
        omega_cam0_xyz = np.zeros((len(uniform_times), 3))
        omega_camN_xyz = np.zeros((len(uniform_times), 3))

        for i, tk in enumerate(uniform_times):
            omega_cam0_xyz[i] = np.array(poseSplineCam0.angularVelocityBodyFrame(tk)).flatten()
            omega_camN_xyz[i] = np.array(poseSplineCamN.angularVelocityBodyFrame(tk)).flatten()

        # scalar norms kept for diagnostics and plots (unchanged)
        omega_cam0_norm = np.linalg.norm(omega_cam0_xyz, axis=1)
        omega_camN_norm = np.linalg.norm(omega_camN_xyz, axis=1)

        print("  [diag] omega_cam0  : min=%.4f  max=%.4f  mean=%.4f  std=%.4f" % (
            omega_cam0_norm.min(), omega_cam0_norm.max(),
            omega_cam0_norm.mean(), omega_cam0_norm.std()))
        print("  [diag] omega_cam%d : min=%.4f  max=%.4f  mean=%.4f  std=%.4f" % (
            self.camIdx,
            omega_camN_norm.min(), omega_camN_norm.max(),
            omega_camN_norm.mean(), omega_camN_norm.std()))

        pearson_r = np.corrcoef(omega_cam0_norm, omega_camN_norm)[0, 1]
        print("  [diag] Pearson r (zero-lag) : %.4f  (>0.99 = signals nearly identical)" % pearson_r)

        np.savetxt(os.path.join(self.outputDir, 'omega_cam0_prior.csv'),
            np.column_stack((uniform_times, omega_cam0_norm)),
            delimiter=',', header='t,omega_norm')
        np.savetxt(os.path.join(self.outputDir, 'omega_cam%d_prior.csv' % self.camIdx),
            np.column_stack((uniform_times, omega_camN_norm)),
            delimiter=',', header='t,omega_norm')

        if len(omega_camN_norm) == 0 or len(omega_cam0_norm) == 0:
            sm.logFatal("The time ranges of camera 0 and camera {0} do not overlap.".format(self.camIdx))
            sys.exit(-1)

        # ── CHANGE 2: 3D vector cross-correlation (dot-product correlation) ─────
        # correlate each axis independently then sum — equivalent to C(τ) = Σ_t ω0(t)·ωN(t+τ)
        # zero-mean each axis first to remove DC bias that can distort the peak
        def zero_mean(x):
            return x - x.mean()

        cc_x = np.correlate(zero_mean(omega_camN_xyz[:, 0]), zero_mean(omega_cam0_xyz[:, 0]), "full")
        cc_y = np.correlate(zero_mean(omega_camN_xyz[:, 1]), zero_mean(omega_cam0_xyz[:, 1]), "full")
        cc_z = np.correlate(zero_mean(omega_camN_xyz[:, 2]), zero_mean(omega_cam0_xyz[:, 2]), "full")
        corr = cc_x + cc_y + cc_z

        # scalar norm correlation kept separately for diagnostic plot comparison
        corr_norm = np.correlate(
            zero_mean(omega_camN_norm), zero_mean(omega_cam0_norm), "full")

        N = len(uniform_times)
        lag_times = np.arange(-(N - 1), N) * dT
        peak_idx  = corr.argmax()
        peak_val  = corr[peak_idx]
        peak_lag  = lag_times[peak_idx]

        # ── CHANGE 3: parabolic interpolation for sub-grid precision ─────────────
        if 0 < peak_idx < len(corr) - 1:
            y0 = corr[peak_idx - 1]
            y1 = corr[peak_idx]
            y2 = corr[peak_idx + 1]
            denom = y0 - 2*y1 + y2
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom          # fractional bin offset
                refined_lag_samples = (peak_idx - (N - 1)) + delta
                refined_lag = refined_lag_samples * dT
                print("  [diag] parabolic refinement  : raw lag=%.4fs  refined lag=%.6fs  (delta=%.4f bins)" % (
                    peak_lag, refined_lag, delta))
            else:
                refined_lag = peak_lag
                print("  [diag] parabolic refinement  : flat peak, no refinement applied")
        else:
            refined_lag = peak_lag
            print("  [diag] parabolic refinement  : peak at boundary, skipped")

        shift = -refined_lag

        mask = np.ones(len(corr), dtype=bool)
        lo = max(0, peak_idx - 5)
        hi = min(len(corr), peak_idx + 6)
        mask[lo:hi] = False

        secondary_val = corr[mask].max()
        secondary_lag = lag_times[mask][corr[mask].argmax()]
        snr = peak_val / secondary_val if secondary_val > 0 else float('inf')

        target_lag_pos =  0.050
        target_lag_neg = -0.050

        idx_pos = int(round(target_lag_pos / dT)) + (N - 1)
        idx_neg = int(round(target_lag_neg / dT)) + (N - 1)
        idx_pos = np.clip(idx_pos, 0, len(corr) - 1)
        idx_neg = np.clip(idx_neg, 0, len(corr) - 1)

        corr_at_pos50 = corr[idx_pos]
        corr_at_neg50 = corr[idx_neg]

        print("  [diag] corr peak (3D vec)    : lag=%.4fs,  value=%.4f" % (peak_lag, peak_val))
        print("  [diag] corr peak (refined)   : lag=%.6fs" % refined_lag)
        print("  [diag] secondary peak        : lag=%.4fs,  value=%.4f" % (secondary_lag, secondary_val))
        print("  [diag] corr at +50ms offset  : %.4f  (peak=%.4f,  ratio=%.3f)" % (corr_at_pos50, peak_val, corr_at_pos50 / peak_val if peak_val > 0 else float('nan')))
        print("  [diag] corr at -50ms offset  : %.4f  (peak=%.4f,  ratio=%.3f)" % (corr_at_neg50, peak_val, corr_at_neg50 / peak_val if peak_val > 0 else float('nan')))
        print("  [diag] peak SNR              : %.3f  (<1.05 = flat/unreliable peak)" % snr)

        t0_rel = uniform_times - t_start

        cam0_obs_times = np.array([obs.time().toSec() for obs in self.target0Observations])
        camN_obs_times = np.array([obs.time().toSec() for obs in self.targetNObservations])

        cam0_obs_rel = cam0_obs_times[(cam0_obs_times >= t_start) & (cam0_obs_times <= t_end)] - t_start
        camN_obs_rel = camN_obs_times[(camN_obs_times >= t_start) & (camN_obs_times <= t_end)] - t_start

        cam0_obs_omega = np.interp(cam0_obs_rel, t0_rel, omega_cam0_norm)
        camN_obs_omega = np.interp(camN_obs_rel, t0_rel, omega_camN_norm)

        # plot 1: angular velocity + observation markers 
        fig, axes = pl.subplots(3, 1, figsize=(10, 9), sharex=True)

        axes[0].plot(t0_rel, omega_cam0_norm, color='tab:blue')
        axes[0].scatter(cam0_obs_rel, cam0_obs_omega, marker='o', s=20, color='tab:blue',
                        zorder=5, label='cam0 obs (%d)' % len(cam0_obs_rel))
        axes[0].set_ylabel("Angular vel (rad/s)")
        axes[0].set_title("cam0")
        axes[0].legend(fontsize=7)
        axes[0].grid(True)

        axes[1].plot(t0_rel, omega_camN_norm, color='tab:orange')
        axes[1].scatter(camN_obs_rel, camN_obs_omega, marker='x', s=30, color='tab:orange',
                        zorder=5, label='cam%d obs (%d)' % (self.camIdx, len(camN_obs_rel)))
        axes[1].set_ylabel("Angular vel (rad/s)")
        axes[1].set_title("cam%d" % self.camIdx)
        axes[1].legend(fontsize=7)
        axes[1].grid(True)

        axes[2].plot(t0_rel, omega_cam0_norm, color='tab:blue', label='cam0 original')
        axes[2].plot(t0_rel - shift, omega_cam0_norm, color='tab:green',
                    linestyle='--', label='cam0 shifted (τ=%.4fs)' % shift)
        axes[2].plot(t0_rel, omega_camN_norm, color='tab:orange', label='cam%d' % self.camIdx)
        axes[2].scatter(cam0_obs_rel, cam0_obs_omega, marker='o', s=15,
                        color='tab:blue', zorder=5)
        axes[2].scatter(camN_obs_rel, camN_obs_omega, marker='x', s=25,
                        color='tab:orange', zorder=5)
        axes[2].set_ylabel("Angular vel (rad/s)")
        axes[2].set_title("Alignment check")
        axes[2].set_xlabel("Time from overlap start (s)")
        axes[2].legend()
        axes[2].grid(True)

        fig.suptitle("Time shift prior: cam0 to cam%d" % self.camIdx)
        pl.tight_layout()
        fig.savefig(os.path.join(self.outputDir, 'omega_angular_velocity_cam%d.png' % self.camIdx), dpi=150, bbox_inches='tight')

        # plot 2: full cross-correlation (3D vec vs norm, overlaid)
        # normalise both to their own peak so they're on the same scale
        corr_norm_plot = corr_norm / corr_norm.max()
        corr_3d_plot   = corr      / corr.max()

        pl.figure()
        pl.plot(lag_times, corr_norm_plot, color='tab:gray',  alpha=0.6, label='scalar norm (old)')
        pl.plot(lag_times, corr_3d_plot,   color='tab:blue',             label='3D vector (new)')
        pl.axvline(x=peak_lag,    color='r',  linestyle='--', label='raw peak %.4fs'     % peak_lag)
        pl.axvline(x=refined_lag, color='g',  linestyle=':',  label='refined peak %.6fs' % refined_lag)
        pl.plot(peak_lag,    corr_3d_plot[peak_idx], 'kx', markersize=10, markeredgewidth=2)
        pl.xlabel("Lag (s)")
        pl.ylabel("Normalised correlation")
        pl.title("Cross-correlation: cam%d vs cam0" % self.camIdx)
        pl.legend()
        pl.grid(True)
        pl.savefig(os.path.join(self.outputDir, 'omega_crosscorr_full_cam%d.png' % self.camIdx), dpi=150, bbox_inches='tight')
        pl.close('all')

        # plot 3: zoomed cross-correlation
        zoom_margin = 0.100
        zoom_mask   = (lag_times >= peak_lag - zoom_margin) & (lag_times <= peak_lag + zoom_margin)
        lag_zoom    = lag_times[zoom_mask]
        corr_zoom   = corr_3d_plot[zoom_mask]

        pl.figure()
        pl.plot(lag_zoom, corr_zoom, color='tab:blue')
        pl.axvline(x=peak_lag,    color='r', linestyle='--', label='raw peak %.4fs'     % peak_lag)
        pl.axvline(x=refined_lag, color='g', linestyle=':',  label='refined %.6fs'      % refined_lag)
        pl.plot(peak_lag, corr_3d_plot[peak_idx], 'kx', markersize=10, markeredgewidth=2)
        pl.xlabel("Lag (s)")
        pl.ylabel("Normalised correlation")
        pl.title("Cross-correlation zoomed around peak\n3D vector: cam%d vs cam0" % self.camIdx)
        pl.legend()
        pl.grid(True)
        pl.savefig(os.path.join(self.outputDir, 'omega_crosscorr_near_peak_cam%d.png' % self.camIdx), dpi=150, bbox_inches='tight')
        pl.close('all')

        self.timeshiftPrior = shift

        print("  Time shift camera 0 to cam%d (t_cam%d = t_cam0 + shift):" % (self.camIdx, self.camIdx))
        print(self.timeshiftPrior)

    def addDesignVariables(self, dvc, setActive=True,
                       baselinedv_group_id=HELPER_GROUP_ID, 
                       calibration_group_id=CALIBRATION_GROUP_ID):
        '''
        Register global design variables - called ONCE for the whole optimization problem
        
        Input parameters:
            dvc: BSplinePoseDesignVariable --> wraps cam0 spline coefficients as design variables
            dvc = asp.BSplinePoseDesignVariable(poseSpline)
        
        Flags:
            setActive: whether to activate spline design variables for optimization
            baselinedv_group_id: Group ID for baseline-related variables (extrinsic, spline)
            calibration_group_id: Group ID for calibration-specific variables (time offset) 

        Design variables:
            T_cam0_cam0 identity transformation for cam0 (for building expressions) --> not needed in general but problem demands all design variables to be registered
            cam0 spline control points (pose trajectory)
        '''
        # identity transformation for cam0 — only registered once
        self.T_cam0_cam0_Dv = aopt.TransformationDv(
            sm.Transformation(),
            rotationActive=False,
            translationActive=False
        )
        for i in range(self.T_cam0_cam0_Dv.numDesignVariables()):
            self.problem.addDesignVariable(self.T_cam0_cam0_Dv.getDesignVariable(i), baselinedv_group_id)

        # spline — only for cam0
        if dvc is not None:
            for i in range(dvc.numDesignVariables()):
                dv = dvc.designVariable(i)
                dv.setActive(setActive)
                self.problem.addDesignVariable(dv, baselinedv_group_id)

    def addCameraDesignVariables(self, camIdx, T_camN_cam0, timeshiftPrior,
                                noTimeCalibration=False,
                                baselinedv_group_id=HELPER_GROUP_ID,
                                calibration_group_id=CALIBRATION_GROUP_ID):
        '''
        Registers per-camera design variables - called ONCE per non-reference camera

        Input parameters:
            camIdx: index of the non-reference camera (e.g. 1 for cam1)
            T_camN_cam0: initial extrinsic estimate (sm.Transformation)
            timeshiftPrior: initial time offset estimate (float, seconds)

        Flags:
            noTimeCalibration: True = temporal offset fixed, False = estimate temporal offset
            baselinedv_group_id: Group ID for baseline-related variables (extrinsic, spline)
            calibration_group_id: Group ID for calibration-specific variables (time offset)

        Design variables:
            T_camN_cam0 extrinsic
            timeshiftCam0ToCamN scalar
        '''
        # extrinsic
        T_dv = aopt.TransformationDv(T_camN_cam0, rotationActive=True, translationActive=True)
        for i in range(T_dv.numDesignVariables()):
            self.problem.addDesignVariable(T_dv.getDesignVariable(i), baselinedv_group_id)

        # timeshift
        tau_dv = aopt.Scalar(timeshiftPrior)
        tau_dv.setActive(not noTimeCalibration)
        self.problem.addDesignVariable(tau_dv, calibration_group_id)

        # store DVs indexed by camera index for later lookup in addCameraErrorTerms
        if not hasattr(self, 'T_camN_cam0_Dvs'):
            self.T_camN_cam0_Dvs = {}
            self.timeshiftDvs = {}

        self.T_camN_cam0_Dvs[camIdx] = T_dv
        self.timeshiftDvs[camIdx] = tau_dv

    def exportReprojErrors(self, camIdx, allReprojectionErrors, tag):
        """
        tag: 'pre' or 'post' — used in filename and print label
        """
        rows = []
        for frameIdx, frameErrors in enumerate(allReprojectionErrors):
            for cornerIdx, rerr in enumerate(frameErrors):
                e = np.array(rerr.error()).flatten()
                norm = np.linalg.norm(e)
                rows.append([frameIdx, cornerIdx, e[0], e[1], norm])

        if rows:
            rows = np.array(rows)
            path = os.path.join(self.outputDir, 'reproj_errors_cam%d_%s.csv' % (camIdx, tag))
            np.savetxt(path, rows, delimiter=',',
                    header='frame_idx,corner_idx,eu,ev,norm', comments='')
            print("  [%s] cam%d reproj — mean: %.4f  std: %.4f  max: %.4f px" % (
                tag, camIdx, np.mean(rows[:, 4]), np.std(rows[:, 4]), np.max(rows[:, 4])))

    def addCameraErrorTerms(self, camIdx, dataset, camera, targetobservations,
                            T_camN_cam0, poseSplineDv=None, blakeZissermanDf=0.0,
                            timeOffsetPadding=0.0, applyFrameTimeShift=False, tag='pre'):
        '''
        Add reprojection error terms for one camera to the optimization problem.

        Parameters:
            camIdx: camera index (0 for reference cam, N for non-reference)
            dataset: BagImageDatasetReader object (carries topic name for logging)
            camera: AslamCamera object — carries geometry, frameType, keypointType, reprojectionErrorType
            targetobservations: list of AprilGrid observations (one per frame)
            T_camN_cam0: transformation expression (identity DV for cam0, T_camN_cam0_Dv for camN)
            poseSplineDv: BSplinePoseDesignVariable wrapping cam0 spline
            blakeZissermanDf: degrees of freedom for Blake-Zisserman M-estimator (0 = disabled)
            timeOffsetPadding: padding for spline boundary check (seconds)
            applyFrameTimeShift: False for cam0 (reference), True for camN (apply tau DV)
            tag: 'pre' or 'post' — used in filename and print label
        '''
        print("")
        print("Adding camera reprojection error terms ({0})".format(dataset.topic))

        # progress bar
        iProgress = sm.Progress2(len(targetobservations))
        iProgress.sample()

        allReprojectionErrors = list()  # list of lists: [frame][corner]
        error_t = camera.reprojectionErrorType

        # corner uncertainty
        R = np.eye(2) * self.cornerUncertainty * self.cornerUncertainty
        invR = np.linalg.inv(R)

        # projection export state
        proj_rows = []
        frameIdx_export = 0  # counts only frames that pass the spline boundary check

        for obs in targetobservations:
            # build frame time expression
            if applyFrameTimeShift:
                frameTime = self.timeshiftDvs[camIdx].toExpression() + obs.time().toSec()
                frameTimeScalar = frameTime.toScalar()
            else:
                frameTime = aopt.ScalarExpression(obs.time().toSec())
                frameTimeScalar = frameTime.toScalar()

            # skip observations outside spline range
            if frameTimeScalar <= poseSplineDv.spline().t_min() or frameTimeScalar >= poseSplineDv.spline().t_max():
                continue

            T_w_cam0 = poseSplineDv.transformationAtTime(frameTime, timeOffsetPadding, timeOffsetPadding)
            T_cam0_w = T_w_cam0.inverse()
            T_camN_w = T_camN_cam0 * T_cam0_w

            imageCornerPoints  = np.array(obs.getCornersImageFrame()).T
            targetCornerPoints = np.array(obs.getCornersTargetFrame()).T

            frame = camera.frameType()
            frame.setGeometry(camera.geometry)

            for pidx in range(0, imageCornerPoints.shape[1]):
                k = camera.keypointType()
                k.setMeasurement(imageCornerPoints[:, pidx])
                k.setInverseMeasurementCovariance(invR)
                frame.addKeypoint(k)

            reprojectionErrors = list()
            for pidx in range(0, imageCornerPoints.shape[1]):
                targetPoint = np.insert(targetCornerPoints.transpose()[pidx], 3, 1)
                p = T_camN_w * aopt.HomogeneousExpression(targetPoint)

                rerr = error_t(frame, pidx, p)

                if blakeZissermanDf > 0.0:
                    mest = aopt.BlakeZissermanMEstimator(blakeZissermanDf)
                    rerr.setMEstimatorPolicy(mest)

                # projection export for all cameras
                p_hom = p.toHomogeneous()
                w = float(p_hom[3])
                p_euc = np.array([float(p_hom[0])/w, float(p_hom[1])/w, float(p_hom[2])/w])
                z = p_euc[2]
                try:
                    kp_proj = camera.geometry.projection().euclideanToKeypoint(p_euc)
                    u_proj, v_proj = float(kp_proj[0]), float(kp_proj[1])
                except Exception:
                    u_proj, v_proj = float('nan'), float('nan')

                u_meas = float(imageCornerPoints[0, pidx])
                v_meas = float(imageCornerPoints[1, pidx])
                proj_rows.append([frameIdx_export, pidx,
                                u_meas, v_meas, u_proj, v_proj,
                                u_meas - u_proj, v_meas - v_proj, z])

                self.problem.addErrorTerm(rerr)
                reprojectionErrors.append(rerr)

            allReprojectionErrors.append(reprojectionErrors)
            frameIdx_export += 1
            iProgress.sample()

        print("\r  Added {0} camera error terms                      ".format(len(targetobservations)))

        # save projection export for all cameras
        if proj_rows:
            proj_arr = np.array(proj_rows)
            path = os.path.join(self.outputDir, 'projections_cam%d_%s.csv' % (camIdx, tag))
            np.savetxt(path, proj_arr, delimiter=',',
                    header='frame_idx,corner_idx,u_meas,v_meas,u_proj,v_proj,eu,ev,depth_z',
                    comments='')
            eu   = proj_arr[:, 6]
            ev   = proj_arr[:, 7]
            norm = np.sqrt(eu**2 + ev**2)
            depth = proj_arr[:, 8]
            print("  [proj_export] cam%d [%s] — mean: %.4f  std: %.4f  max: %.4f px" % (
                camIdx, tag, np.mean(norm), np.std(norm), np.max(norm)))
            print("  [proj_export] frames with norm > 100px : %d" % np.sum(norm > 100.0))
            print("  [proj_export] corners with depth_z <= 0: %d" % np.sum(depth <= 0.0))
            print("  [proj_export] corners with u_proj nan  : %d" % np.sum(np.isnan(proj_arr[:, 4])))
            print("  [proj_export] corners with |u_proj| > 1280: %d" % np.sum(np.abs(proj_arr[:, 4]) > 1280))
            print("  [proj_export] corners with |v_proj| > 960 : %d" % np.sum(np.abs(proj_arr[:, 5]) > 960))

        return allReprojectionErrors
    
    def saveCamChainYaml(self, resultFile):

        def inline_matrix(mat):
            """Return 4x4 numpy array as list-of-lists for inline YAML rendering."""
            return [list(float(x) for x in row) for row in mat]

        def inline_vector(vec):
            return [float(x) for x in vec]

        def camDict(cam, dataset):
            P = cam.geometry.projection()

            if isinstance(P, acv.DistortedPinholeProjection):
                cam_type, dist_type = 'pinhole', 'radtan'
            elif isinstance(P, acv.EquidistantPinholeProjection):
                cam_type, dist_type = 'pinhole', 'equidistant'
            elif isinstance(P, acv.FovPinholeProjection):
                cam_type, dist_type = 'pinhole', 'fov'
            elif isinstance(P, acv.PinholeProjection):
                cam_type, dist_type = 'pinhole', 'none'
            elif isinstance(P, acv.OmniProjection):
                cam_type, dist_type = 'omni', 'none'
            else:
                raise RuntimeError("Unknown camera projection type: {}".format(type(P)))

            if cam_type == 'pinhole':
                intrinsics = [P.fu(), P.fv(), P.cu(), P.cv()]
            else:
                intrinsics = [P.xi(), P.fu(), P.fv(), P.cu(), P.cv()]

            dist_coeffs = P.distortion().getParameters().flatten().tolist() if hasattr(P, 'distortion') else []

            return {
                'camera_model':      cam_type,
                'distortion_model':  dist_type,
                'intrinsics':        inline_vector(intrinsics),
                'distortion_coeffs': inline_vector(dist_coeffs),
                'resolution':        [int(P.ru()), int(P.rv())],
                'rostopic':          dataset.topic,
            }

        T_camN_cam0_mat = self.T_camN_cam0_Dvs[self.camIdx].T()
        tau_opt         = float(self.timeshiftDvs[self.camIdx].toScalar())

        cam0_dict = camDict(self.camera0, self.dataset0)
        cam0_dict['cam_overlaps'] = [self.camIdx]

        camN_dict = camDict(self.cameraN, self.datasetN)
        camN_dict['cam_overlaps'] = [0]
        camN_dict['T_cn_cnm1']    = inline_matrix(T_camN_cam0_mat)

        # async-specific metadata — not in standard Kalibr but preserved here
        async_meta = {
            'timeshift_cam0_to_cam{}'.format(self.camIdx): tau_opt,
            'async_calibration': {
                'method':                   'bspline_crosscorr',
                'tau_prior_s':              float(self.timeshiftPrior),
                'tau_optimized_s':          tau_opt,
                'tau_fixed_in_opt':         not self.timeshiftDvs[self.camIdx].isActive(),
                'crosscorr_method':         '3d_vector_dot_product',
                'crosscorr_parabolic_refine': True,
            }
        }
        camN_dict.update(async_meta)

        camchain = {
            'cam0': cam0_dict,
            'cam{}'.format(self.camIdx): camN_dict,
        }

        # custom YAML representer for inline lists 
        # Kalibr uses flow style for vectors/matrices, block style for the rest.
        # We tag inline_vector/inline_matrix results as a custom type and emit them
        # in flow style only.

        class InlineList(list): pass

        def inline_list_representer(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

        def convert_inline(obj):
            """Recursively convert plain lists to InlineList so they render inline."""
            if isinstance(obj, list):
                # 4x4 matrix: list of 4 lists → each row inline
                if all(isinstance(row, list) for row in obj):
                    return InlineList([InlineList(row) for row in obj])
                return InlineList(obj)
            if isinstance(obj, dict):
                return {k: convert_inline(v) for k, v in obj.items()}
            return obj

        camchain_inline = convert_inline(camchain)

        dumper = yaml.Dumper
        dumper.add_representer(InlineList, inline_list_representer)

        with open(resultFile, 'w') as f:
            yaml.dump(camchain_inline, f, Dumper=dumper, default_flow_style=False, sort_keys=False)

        print("Camera chain saved to {}".format(resultFile))
        print("  tau_prior            : {:.6f} s".format(self.timeshiftPrior))
        print("  tau_optimized        : {:.6f} s".format(tau_opt))
        print("  tau_fixed_in_opt     : {}".format(not self.timeshiftDvs[self.camIdx].isActive()))

    def buildAndSolveProblem(self,
                            splineOrder=6,
                            poseKnotsPerSecond=70,
                            maxIterations=20,
                            timeOffsetPadding=0.02,
                            blakeZisserCam=-1,
                            noTimeCalibration=False,
                            verbose=False):

        print("\tSpline order: %d" % splineOrder)
        print("\tPose knots per second: %d" % poseKnotsPerSecond)
        print("\tMax iterations: %d" % maxIterations)
        print("\tTime offset padding: %f" % timeOffsetPadding)

        #######################
        ## timeshift prior
        #######################
        self.findTimeshiftPrior(verbose=verbose)

        #######################
        ## cam0 pose spline
        #######################
        poseSpline = self.initPoseSpline(self.target0Observations, splineOrder, poseKnotsPerSecond,
                                        timeOffsetPadding=timeOffsetPadding, label="cam0_main")

        #######################
        ## design variables
        #######################
        poseSplineDv = asp.BSplinePoseDesignVariable(poseSpline)
        self.addDesignVariables(poseSplineDv)
        self.addCameraDesignVariables(self.camIdx, self.T_camN_cam0, self.timeshiftPrior,
                                    noTimeCalibration=noTimeCalibration)

        #######################
        ## error terms
        #######################

        # cam0 — reference camera, no time shift
        self.cam0ReprojectionErrors = self.addCameraErrorTerms(
            camIdx=0,
            dataset=self.dataset0,
            camera=self.camera0,
            targetobservations=self.target0Observations,
            T_camN_cam0=self.T_cam0_cam0_Dv.toExpression(),
            poseSplineDv=poseSplineDv,
            blakeZissermanDf=blakeZisserCam,
            timeOffsetPadding=timeOffsetPadding,
            applyFrameTimeShift=False,
            tag='pre'
        )

        # camN — non-reference camera, apply time shift
        self.camNReprojectionErrors = self.addCameraErrorTerms(
            camIdx=self.camIdx,
            dataset=self.datasetN,
            camera=self.cameraN,
            targetobservations=self.targetNObservations,
            T_camN_cam0=self.T_camN_cam0_Dvs[self.camIdx].toExpression(),
            poseSplineDv=poseSplineDv,
            blakeZissermanDf=blakeZisserCam,
            timeOffsetPadding=timeOffsetPadding,
            applyFrameTimeShift=True,
            tag='pre'
        )

        ######################
        ## solve
        ######################
        options = aopt.Optimizer2Options()
        options.verbose = True
        options.doLevenbergMarquardt = True
        options.levenbergMarquardtLambdaInit = 10.0
        options.nThreads = max(1, multiprocessing.cpu_count() - 1)
        options.convergenceDeltaX = 1e-5
        options.convergenceDeltaJ = 1e-2
        options.maxIterations = maxIterations
        options.trustRegionPolicy = aopt.LevenbergMarquardtTrustRegionPolicy(options.levenbergMarquardtLambdaInit)
        options.linearSolver = aopt.BlockCholeskyLinearSystemSolver()

        self.optimizer = aopt.Optimizer2(options)
        self.optimizer.setProblem(self.problem)

        self.optimizer.initialize()
        self.optimizer.evaluateError(True)

        self.exportReprojErrors(0, self.cam0ReprojectionErrors, 'pre')
        self.exportReprojErrors(self.camIdx, self.camNReprojectionErrors, 'pre')

        try:
            self.optimizer.optimize()
        except Exception as e:
            sm.logError(str(e))
            raise RuntimeError("Optimization failed!")

        self.exportReprojErrors(0, self.cam0ReprojectionErrors, 'post')
        self.exportReprojErrors(self.camIdx, self.camNReprojectionErrors, 'post')

        ########################
        ## cam1 trajectory validation
        ########################

        poseSplineCam1_indep = self.initPoseSpline(
            self.targetNObservations,
            splineOrder=splineOrder,
            poseKnotsPerSecond=poseKnotsPerSecond,
            timeOffsetPadding=timeOffsetPadding,
            label='cam%d_validation' % self.camIdx
        )

        T_cam1_cam0_mat = np.array(self.T_camN_cam0_Dvs[self.camIdx].T())
        tau_val         = float(self.timeshiftDvs[self.camIdx].toScalar())

        t_spline_min = poseSpline.t_min()
        t_spline_max = poseSpline.t_max()
        t_indep_min  = poseSplineCam1_indep.t_min()
        t_indep_max  = poseSplineCam1_indep.t_max()

        convention_checked = False
        for obs in self.targetNObservations:
            t_check = obs.time().toSec()
            t_q     = t_check + tau_val
            if t_q <= t_spline_min or t_q >= t_spline_max:
                continue
            if t_check <= t_indep_min or t_check >= t_indep_max:
                continue

            T_spline_cam0 = np.array(poseSpline.transformation(t_q))
            T_spline_cam1 = np.array(poseSplineCam1_indep.transformation(t_check))
            T_pnp_cam1    = np.array(obs.T_t_c().T())

            print("")
            print("  [convention check] first valid cam1 frame at t=%.6f" % t_check)
            print("    poseSpline cam0  T[0,:] = %s" % np.array2string(T_spline_cam0[0,:], precision=4))
            print("    indep spline cam1 T[0,:] = %s" % np.array2string(T_spline_cam1[0,:], precision=4))
            print("    PnP T_cam1_target T[0,:] = %s" % np.array2string(T_pnp_cam1[0,:], precision=4))
            print("    indep spline translation : %s" % np.array2string(T_spline_cam1[:3,3], precision=4))
            print("    PnP translation          : %s" % np.array2string(T_pnp_cam1[:3,3],   precision=4))
            convention_checked = True
            break

        rows = []
        outlier_count = 0

        for obs in self.targetNObservations:
            t_cam1  = obs.time().toSec()
            t_query = t_cam1 + tau_val

            if t_query <= t_spline_min or t_query >= t_spline_max:
                continue
            if t_cam1 <= t_indep_min or t_cam1 >= t_indep_max:
                continue

            boundary_margin = 2.0 / poseKnotsPerSecond
            if (t_cam1 < t_indep_min + boundary_margin or
                    t_cam1 > t_indep_max - boundary_margin):
                continue

            T_cam0_w_composed = np.array(poseSpline.transformation(t_query))
            T_cam1_w_composed = T_cam1_cam0_mat @ T_cam0_w_composed
            T_cam1_w_indep    = np.array(poseSplineCam1_indep.transformation(t_cam1))

            delta_T   = np.linalg.inv(T_cam1_w_composed) @ T_cam1_w_indep
            delta_t   = delta_T[:3, 3]
            trans_err = np.linalg.norm(delta_t)

            R_delta   = delta_T[:3, :3]
            cos_angle = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
            rot_err   = np.degrees(np.arccos(cos_angle))

            t_composed = T_cam1_w_composed[:3, 3]
            rv_opt     = rotation_matrix_to_rotvec(T_cam1_w_composed[:3, :3])
            rv_indep   = rotation_matrix_to_rotvec(T_cam1_w_indep[:3, :3])

            if trans_err > 0.05:
                outlier_count += 1
                t_rel      = t_cam1 - t_spline_min
                near_start = t_cam1 - t_indep_min
                near_end   = t_indep_max - t_cam1
                print("  [outlier] t=%.4f (rel=%.2fs)  trans=%.4fm  rot=%.2f°  "
                    "dist_from_start=%.3fs  dist_from_end=%.3fs" % (
                    t_cam1, t_rel, trans_err, rot_err, near_start, near_end))

            rows.append([
                t_cam1,
                t_composed[0],  t_composed[1],  t_composed[2],
                rv_opt[0],      rv_opt[1],       rv_opt[2],
                delta_t[0],     delta_t[1],      delta_t[2],
                rv_indep[0],    rv_indep[1],     rv_indep[2],
                trans_err,      rot_err
            ])

        if rows:
            arr  = np.array(rows)
            path = os.path.join(self.outputDir, 'cam%d_pose_consistency.csv' % self.camIdx)
            np.savetxt(path, arr, delimiter=',',
                    header='t_cam1,'
                            'tx_composed,ty_composed,tz_composed,'
                            'rvx_composed,rvy_composed,rvz_composed,'
                            'delta_tx,delta_ty,delta_tz,'
                            'rvx_indep,rvy_indep,rvz_indep,'
                            'trans_err_m,rot_err_deg',
                    comments='')

            trans_errs = arr[:, 13]
            rot_errs   = arr[:, 14]

            print("")
            print("  [validation] cam%d pose consistency (%d frames, %d outliers >50mm):" % (
                self.camIdx, len(rows), outlier_count))
            print("    translation error — mean: %.4fm  std: %.4fm  max: %.4fm" % (
                np.mean(trans_errs), np.std(trans_errs), np.max(trans_errs)))
            print("    rotation error    — mean: %.4f°  std: %.4f°  max: %.4f°" % (
                np.mean(rot_errs), np.std(rot_errs), np.max(rot_errs)))
            for p in [50, 75, 90, 95, 99]:
                print("    trans p%d: %.4fm   rot p%d: %.4f°" % (
                    p, np.percentile(trans_errs, p),
                    p, np.percentile(rot_errs,   p)))
        else:
            print("  [validation] cam%d pose consistency: no valid frames found" % self.camIdx)

        ########################
        ## return results
        ########################
        T_camN_cam0_opt = sm.Transformation(self.T_camN_cam0_Dvs[self.camIdx].T())
        tau_opt = self.timeshiftDvs[self.camIdx].toScalar()
        return T_camN_cam0_opt, tau_opt

def main():
    signal.signal(signal.SIGINT, signal_exit)

    parsed = parseArgs()

    if parsed.verbose:
        sm.setLoggingLevel(sm.LoggingLevel.Debug)
    else:
        sm.setLoggingLevel(sm.LoggingLevel.Info)

    sm.logInfo("Output directory: %s" % parsed.outputDir)

    numCams = len(parsed.topics)
    targetConfig = kc.CalibrationTargetParameters(parsed.targetYaml)

    # load all configs and datasets
    camConfigs = []
    datasets = []
    for i in range(numCams):
        print("Loading cam%d:" % i)
        camConfigs.append(kc.CameraParameters(parsed.camIntrinsics[i]))
        datasets.append(initBagDataset(parsed.bagfile, parsed.topics[i],
                                       parsed.bag_from_to, parsed.bag_freq))

    # run one AsyncCalibrator per non-reference camera
    results = []
    for camIdx in range(1, numCams):
        print("")
        print("Calibrating cam%d against cam0" % camIdx)

        calibrator = AsyncCalibrator(
            camConfigs[0], camConfigs[camIdx],
            camIdx=camIdx,
            targetConfig=targetConfig,
            dataset0=datasets[0],
            datasetN=datasets[camIdx],
            reprojectionSigma=1.0,
            showCorners=parsed.showExtraction,
            showReproj=False,
            showOneStep=False,
            outputDir=parsed.outputDir
        )

        # override tau prior from CLI if provided
        if parsed.tauPrior is not None:
            tau_override = parsed.tauPrior[camIdx - 1]
            sm.logWarn("cam%d: overriding tau prior with CLI value: %.6f s" % (camIdx, tau_override))
            calibrator.timeshiftPrior = tau_override

        T_camN_cam0, tau = calibrator.buildAndSolveProblem(
            splineOrder=parsed.splineOrder,
            poseKnotsPerSecond=parsed.knotsPerSecond,
            maxIterations=parsed.maxIterations,
            timeOffsetPadding=parsed.timeOffsetPadding,
            noTimeCalibration=parsed.noTimeCalibration,
            blakeZisserCam=-1,
            verbose=parsed.verbose
        )

        bagtag = os.path.splitext(os.path.basename(parsed.bagfile))[0]
        resultFile = os.path.join(parsed.outputDir,
                                  "%s-cam0-cam%d-camchain.yaml" % (bagtag, camIdx))
        calibrator.saveCamChainYaml(resultFile)

        results.append((camIdx, T_camN_cam0, tau, resultFile))

    # summary
    print("Calibration complete, results:")
    for camIdx, T, tau, f in results:
        print("")
        print("  cam%d results -> %s" % (camIdx, f))
        print("  T_cam%d_cam0:" % camIdx)
        print(T.T())
        print("  time shift cam0 -> cam%d: %.6f s" % (camIdx, tau))


if __name__ == "__main__":
    main()