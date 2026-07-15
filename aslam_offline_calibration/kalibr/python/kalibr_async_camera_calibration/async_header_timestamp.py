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
import argparse 
import math
import pylab as pl
import multiprocessing
import os
import yaml

CALIBRATION_GROUP_ID = 0
HELPER_GROUP_ID = 1

#read image topics from the dataset
def initBagDataset(bagfile, topic, from_to, freq):
    print("\tDataset:          {0}".format(bagfile))
    print("\tTopic:            {0}".format(topic))
    reader = kc.BagImageDatasetReader(bagfile, topic, bag_from_to=from_to, bag_freq=freq)
    print("\tNumber of images in the bag: {0}".format(reader.numImages()))
    return reader

#available models
cameraModels = { 'pinhole-radtan': acvb.DistortedPinhole,
                 'pinhole-equi':   acvb.EquidistantPinhole,
                 'pinhole-fov':    acvb.FovPinhole,
                 'omni-none':      acvb.Omni,
                 'omni-radtan':    acvb.DistortedOmni,
                 'eucm-none':      acvb.ExtendedUnified,
                 'ds-none':        acvb.DoubleSphere}

#for interupting pipeline 
def signal_exit(signal, frame):
    sm.logWarn("Shutdown requested! (CTRL+C)")
    sys.exit(2)

def parseArgs():
    class KalibrArgParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            sm.logError('%s' % message)
            sys.exit(2)
        def format_help(self):
            formatter = self._get_formatter()
            formatter.add_text(self.description)
            formatter.add_usage(self.usage, self._actions,
                                self._mutually_exclusive_groups)
            for action_group in self._action_groups:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
            formatter.add_text(self.epilog)
            return formatter.format_help()     
        
    usage = """
    Example usage to calibrate a camera system with two cameras using an aprilgrid. 
    
    cam0: omnidirection model with radial-tangential distortion
    cam1: pinhole model with equidistant distortion
    
    %(prog)s --models omni-radtan pinhole-equi --target aprilgrid.yaml \\
              --bag MYROSBAG.bag --topics /cam0/image_raw /cam1/image_raw
    
    example aprilgrid.yaml:
        target_type: 'aprilgrid'
        tagCols: 6
        tagRows: 6
        tagSize: 0.088  #m
        tagSpacing: 0.3 #percent of tagSize"""
            
    parser = KalibrArgParser(description='Calibrate the intrinsics, extrinsics, time offset of a camera system with non-shared overlapping field of view.', usage=usage)
    parser.add_argument('--models', nargs='+', dest='models', help='The camera model {0} to estimate'.format(list(cameraModels.keys())), required=True)
    
    groupSource = parser.add_argument_group('Data source')
    groupSource.add_argument('--bag', dest='bagfile', help='The bag file with the data')
    groupSource.add_argument('--topics', nargs='+', dest='topics', help='The list of image topics', required=True)
    groupSource.add_argument('--bag-from-to', metavar='bag_from_to', type=float, nargs=2, help='Use the bag data starting from up to this time [s]')
    groupSource.add_argument('--bag-freq', metavar='bag_freq', type=float, help='Frequency to extract features at [hz]')

    groupTarget = parser.add_argument_group('Calibration target configuration')
    groupTarget.add_argument('--target', dest='targetYaml', help='Calibration target configuration as yaml file', required=True)
    
    outputSettings = parser.add_argument_group('Output options')
    outputSettings.add_argument('--verbose', action='store_true', dest='verbose', help='Enable (really) verbose output (disables plots)')
    outputSettings.add_argument('--show-extraction', action='store_true', dest='showextraction', help='Show the calibration target extraction. (disables plots)')
    outputSettings.add_argument('--plot', action='store_true', dest='plot', help='Plot during calibration (this could be slow).')
    outputSettings.add_argument('--dont-show-report', action='store_true', dest='dontShowReport', help='Do not show the report on screen after calibration.')
    outputSettings.add_argument('--export-poses', action='store_true', dest='exportPoses', help='Export the optimized poses into a CSV (time_ns, position, quaterion)')

    #print help if no argument is specified
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(2)
        
    #Parser the argument list
    try:
        parsed = parser.parse_args()
    except:
        sys.exit(2)
    
    #some checks
    if len(parsed.topics) != len(parsed.models):
        sm.logError("Please specify exactly one camera model (--models) for each topic (--topics).")
        sys.exit(2)
    
    #there is a with the gtk plot widget, so we cant plot if we have opencv windows open...
    #--> disable the plots in these special situations
    if parsed.showextraction or parsed.verbose:
        parsed.dontShowReport = True
    
    return parsed


class AsyncCalibrator():
    # camera config, target config, dataset
    def __init__(self, cam0Config, camNConfig, camIdx, targetConfig, dataset0, datasetN, reprojectionSigma=1.0, showCorners=True, showReproj=True, showOneStep=False):
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
        np.savetxt('/data/poses_discrete_%s.csv' % label, 
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
        np.savetxt('/data/poses_spline_dense_%s.csv' % label,
                np.hstack((dense_times.reshape(-1,1), dense_curves.T)),
                delimiter=',',
                header='t,tx,ty,tz,rx,ry,rz')
        
        return pose   
    
    def findTimeshiftPrior(self, verbose=False):
        print("Estimating time shift camera %d to camera 0:" % self.camIdx)
        
        #fit a spline to the camera observations
        poseSplineCam0 = self.initPoseSpline(self.target0Observations, poseKnotsPerSecond=10, timeOffsetPadding=0.0, label='cam0_prior')
        poseSplineCamN = self.initPoseSpline(self.targetNObservations, poseKnotsPerSecond=10, timeOffsetPadding=0.0, label='cam%d_prior' % self.camIdx)

        #################

        #time range diagnostic
        t_start_cam0 = poseSplineCam0.t_min()
        t_end_cam0   = poseSplineCam0.t_max()
        t_start_camN = poseSplineCamN.t_min()
        t_end_camN   = poseSplineCamN.t_max()

        print("  [diag] cam0 spline range : %.6f  to  %.6f  (%.3fs)" % (
            t_start_cam0, t_end_cam0, t_end_cam0 - t_start_cam0))
        print("  [diag] cam%d spline range : %.6f  to  %.6f  (%.3fs)" % (
            self.camIdx, t_start_camN, t_end_camN, t_end_camN - t_start_camN))

        t_start = max(t_start_cam0, t_start_camN)   #overlap check
        t_end   = min(t_end_cam0,   t_end_camN)
        overlap = t_end - t_start
        print("  [diag] overlap window    : %.6f  to  %.6f  (%.3fs)" % (t_start, t_end, overlap))

        dT = 0.01
        uniform_times = np.arange(t_start, t_end, dT)
        effective_dT  = uniform_times[1] - uniform_times[0] if len(uniform_times) > 1 else float('nan')
        print("  [diag] uniform grid      : %d samples,  dT_nominal=%.4fs,  dT_effective=%.6fs" % (
            len(uniform_times), dT, effective_dT))

        #################

        omega_cam0_norm = []
        omega_camN_norm = []

        # Uniform time grid over overlapping range
        t_start = max(poseSplineCam0.t_min(), poseSplineCamN.t_min())
        t_end = min(poseSplineCam0.t_max(), poseSplineCamN.t_max())
        dT = 0.01  # 10ms grid spacing
        uniform_times = np.arange(t_start, t_end, dT)

        for tk in uniform_times:
            omega0 = aopt.EuclideanExpression(np.matrix(poseSplineCam0.angularVelocityBodyFrame(tk)).transpose())
            omegaN = aopt.EuclideanExpression(np.matrix(poseSplineCamN.angularVelocityBodyFrame(tk)).transpose())
            omega_cam0_norm.append(np.linalg.norm(omega0.toEuclidean()))
            omega_camN_norm.append(np.linalg.norm(omegaN.toEuclidean()))

        omega_cam0_norm = np.array(omega_cam0_norm)
        omega_camN_norm = np.array(omega_camN_norm)

        print("  [diag] omega_cam0  : min=%.4f  max=%.4f  mean=%.4f  std=%.4f" % (
            omega_cam0_norm.min(), omega_cam0_norm.max(),
            omega_cam0_norm.mean(), omega_cam0_norm.std()))
        print("  [diag] omega_cam%d : min=%.4f  max=%.4f  mean=%.4f  std=%.4f" % (
            self.camIdx,
            omega_camN_norm.min(), omega_camN_norm.max(),
            omega_camN_norm.mean(), omega_camN_norm.std()))

        pearson_r = np.corrcoef(omega_cam0_norm, omega_camN_norm)[0, 1]
        print("  [diag] Pearson r (zero-lag) : %.4f  (>0.99 = signals nearly identical)" % pearson_r)

        # discrete poses for cross correlation
        np.savetxt('/data/omega_cam0_prior.csv',
            np.column_stack((uniform_times, omega_cam0_norm)),
            delimiter=',',
            header='t,omega_norm')

        np.savetxt('/data/omega_cam%d_prior.csv' % self.camIdx,
            np.column_stack((uniform_times, omega_camN_norm)),
            delimiter=',',
            header='t,omega_norm')

        #verify
        if len(omega_camN_norm) == 0 or len(omega_cam0_norm) == 0:
            sm.logFatal("The time ranges of camera 0 and camera {0} do not overlap. "
                        "Please make sure that your sensors are synchronized correctly.".format(self.camIdx))
            sm.logFatal("Cam0 spline range: {0} to {1}".format(poseSplineCam0.t_min(), poseSplineCam0.t_max()))
            sm.logFatal("Cam{0} spline range: {1} to {2}".format(self.camIdx, poseSplineCamN.t_min(), poseSplineCamN.t_max()))
            sm.logFatal("Cam0 first observation: {0}".format(self.target0Observations[0].time().toSec()))
            sm.logFatal("Cam0 last observation: {0}".format(self.target0Observations[-1].time().toSec()))
            sm.logFatal("Cam{0} first observation: {1}".format(self.camIdx, self.targetNObservations[0].time().toSec()))
            sm.logFatal("Cam{0} last observation: {1}".format(self.camIdx, self.targetNObservations[-1].time().toSec()))
            sys.exit(-1)

        # Cross-correlate
        corr = np.correlate(omega_camN_norm, omega_cam0_norm, "full")
        lag_times = np.arange(-(len(omega_cam0_norm)-1), len(omega_cam0_norm)) * dT
        peak_idx  = corr.argmax()
        peak_val  = corr[peak_idx]
        peak_lag  = lag_times[peak_idx]

        discrete_shift_xcorr = corr.argmax() - (len(omega_cam0_norm) - 1)
        shift_xcorr = -discrete_shift_xcorr * dT

        # secondary peak: mask ±5 samples around primary, find next highest
        mask = np.ones(len(corr), dtype=bool)
        lo = max(0, peak_idx - 5)
        hi = min(len(corr), peak_idx + 6)
        mask[lo:hi] = False
        secondary_val = corr[mask].max()
        snr = peak_val / secondary_val if secondary_val > 0 else float('inf')

        print("  [diag] corr peak           : lag=%.4fs,  value=%.4f" % (peak_lag, peak_val))
        print("  [diag] secondary peak val  : %.4f" % secondary_val)
        print("  [diag] peak SNR            : %.3f  (<1.05 = flat/unreliable peak)" % snr)
        print("  [diag] xcorr shift         : %.6fs  (not used for init)" % shift_xcorr)

        # Header timestamp-based initialization
        # Convention: t_camN = t_cam0 + tau  =>  tau = t_camN_first - t_cam0_first
        t_cam0_first = self.target0Observations[0].time().toSec()
        t_camN_first = self.targetNObservations[0].time().toSec()
        shift_header = t_camN_first - t_cam0_first

        print("  [diag] cam0 first obs t       : %.6f" % t_cam0_first)
        print("  [diag] cam%d first obs t      : %.6f" % (self.camIdx, t_camN_first))
        print("  [diag] header timestamp shift : %.6fs" % shift_header)

        #Create plots
        if verbose:
            fig, axes = pl.subplots(3, 1, figsize=(10, 9), sharex=True)
    
            axes[0].plot(uniform_times, omega_cam0_norm, color='tab:blue')
            axes[0].set_ylabel("Angular vel (rad/s)")
            axes[0].set_title("cam0")
            axes[0].grid(True)

            axes[1].plot(uniform_times, omega_camN_norm, color='tab:orange')
            axes[1].set_ylabel("Angular vel (rad/s)")
            axes[1].set_title("cam%d" % self.camIdx)
            axes[1].grid(True)

            axes[2].plot(uniform_times, omega_cam0_norm, color='tab:blue', label='cam0 original')
            axes[2].plot(uniform_times - shift_header, omega_cam0_norm, color='tab:green',
                        linestyle='--', label='cam0 shifted (τ=%.4fs)' % shift_header)
            axes[2].plot(uniform_times, omega_camN_norm, color='tab:orange', label='cam%d' % self.camIdx)
            axes[2].set_ylabel("Angular vel (rad/s)")
            axes[2].set_title("Alignment check (header timestamp shift)")
            axes[2].set_xlabel("Time (s)")
            axes[2].legend()
            axes[2].grid(True)

            fig.suptitle("Time shift prior: cam0 to cam%d" % self.camIdx)
            pl.tight_layout()

            pl.figure()
            pl.plot(lag_times, corr)
            pl.xlabel("Lag (s)")
            pl.ylabel("Cross-correlation (rad²/s²)")
            pl.title("Cross-correlation ||ω_cam%d|| vs ||ω_cam0|| (diagnostic only)" % self.camIdx)
            pl.axvline(x=peak_lag, color='r', linestyle='--', label="xcorr peak at %.4fs" % peak_lag)
            pl.axvline(x=shift_header, color='g', linestyle='--', label="header shift at %.4fs" % shift_header)
            pl.legend()
            pl.show()

        sm.logDebug("cont. time shift (xcorr):  {0}".format(shift_xcorr))
        sm.logDebug("cont. time shift (header): {0}".format(shift_header))
        sm.logDebug("dT: {0}".format(dT))
        
        # Use header timestamp shift as the prior
        self.timeshiftPrior = shift_header

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
            path = '/data/reproj_errors_cam%d_%s.csv' % (camIdx, tag)
            np.savetxt(path, rows, delimiter=',',
                    header='frame_idx,corner_idx,eu,ev,norm', comments='')
            print("  [%s] cam%d reproj — mean: %.4f  std: %.4f  max: %.4f px" % (
                tag, camIdx, np.mean(rows[:, 4]), np.std(rows[:, 4]), np.max(rows[:, 4])))

    def addCameraErrorTerms(self, camIdx, dataset, camera, targetobservations,
                            T_camN_cam0, poseSplineDv=None, blakeZissermanDf=0.0,
                            timeOffsetPadding=0.0, applyFrameTimeShift=False):
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
                # camN: shift raw timestamp by tau DV so spline is queried at corrected time
                frameTime = self.timeshiftDvs[camIdx].toExpression() + obs.time().toSec()
                frameTimeScalar = frameTime.toScalar()
            else:
                # cam0: reference clock, no shift
                frameTime = aopt.ScalarExpression(obs.time().toSec())
                frameTimeScalar = frameTime.toScalar()

            # skip observations outside spline range
            if frameTimeScalar <= poseSplineDv.spline().t_min() or frameTimeScalar >= poseSplineDv.spline().t_max():
                continue

            ## transformation chain: target frame --> camN frame
            # T_dest_src convention:
            #   T_w_cam0: cam0 pose in world (from spline)
            #   T_camN_w = T_camN_cam0 * T_cam0_w
            T_w_cam0 = poseSplineDv.transformationAtTime(frameTime, timeOffsetPadding, timeOffsetPadding)
            T_cam0_w = T_w_cam0.inverse()
            T_camN_w = T_camN_cam0 * T_cam0_w

            # get 2D image corners and 3D target corners for this observation
            imageCornerPoints = np.array(obs.getCornersImageFrame()).T
            targetCornerPoints = np.array(obs.getCornersTargetFrame()).T

            # build aslam frame — handles distortion via fixed camera geometry
            frame = camera.frameType()
            frame.setGeometry(camera.geometry)

            # populate frame with keypoints
            for pidx in range(0, imageCornerPoints.shape[1]):
                k = camera.keypointType()
                k.setMeasurement(imageCornerPoints[:, pidx])
                k.setInverseMeasurementCovariance(invR)
                frame.addKeypoint(k)

            # build reprojection error terms
            reprojectionErrors = list()
            for pidx in range(0, imageCornerPoints.shape[1]):
                targetPoint = np.insert(targetCornerPoints.transpose()[pidx], 3, 1)
                p = T_camN_w * aopt.HomogeneousExpression(targetPoint)

                rerr = error_t(frame, pidx, p)

                if blakeZissermanDf > 0.0:
                    mest = aopt.BlakeZissermanMEstimator(blakeZissermanDf)
                    rerr.setMEstimatorPolicy(mest)

                # projection export — cam0 only, evaluates chain before optimization
                if camIdx == 0:
                    p_hom = p.toHomogeneous()
                    w = float(p_hom[3])
                    p_euc = np.array([float(p_hom[0])/w, float(p_hom[1])/w, float(p_hom[2])/w])
                    z = p_euc[2]

                    try:
                        kp_proj = camera.geometry.projection().euclideanToKeypoint(p_euc)
                        u_proj = float(kp_proj[0])
                        v_proj = float(kp_proj[1])
                    except Exception as ex:
                        u_proj, v_proj = float('nan'), float('nan')

                    u_meas = float(imageCornerPoints[0, pidx])
                    v_meas = float(imageCornerPoints[1, pidx])

                    proj_rows.append([
                        frameIdx_export, pidx,
                        u_meas, v_meas,
                        u_proj, v_proj,
                        u_meas - u_proj, v_meas - v_proj,
                        z
                    ])
                self.problem.addErrorTerm(rerr)
                reprojectionErrors.append(rerr)

            allReprojectionErrors.append(reprojectionErrors)
            frameIdx_export += 1
            iProgress.sample()

        print("\r  Added {0} camera error terms                      ".format(len(targetobservations)))

        # save projection export for cam0
        if camIdx == 0 and proj_rows:
            proj_arr = np.array(proj_rows)
            path = '/data/projections_cam0_pre.csv'
            np.savetxt(path, proj_arr, delimiter=',',
                    header='frame_idx,corner_idx,u_meas,v_meas,u_proj,v_proj,eu,ev,depth_z',
                    comments='')

            # summary for high-error frames
            eu = proj_arr[:, 6]
            ev = proj_arr[:, 7]
            norm = np.sqrt(eu**2 + ev**2)
            depth = proj_arr[:, 8]

            print("  [proj_export] cam0 — mean: %.4f  std: %.4f  max: %.4f px" % (
                np.mean(norm), np.std(norm), np.max(norm)))
            print("  [proj_export] frames with norm > 100px : %d" % np.sum(norm > 100.0))
            print("  [proj_export] corners with depth_z <= 0: %d" % np.sum(depth <= 0.0))
            print("  [proj_export] corners with u_proj nan  : %d" % np.sum(np.isnan(proj_arr[:, 4])))
            print("  [proj_export] corners with |u_proj| > 1280: %d" % np.sum(np.abs(proj_arr[:, 4]) > 1280))
            print("  [proj_export] corners with |v_proj| > 960 : %d" % np.sum(np.abs(proj_arr[:, 5]) > 960))


        return allReprojectionErrors
    
    def saveCamChainYaml(self, resultFile):
        '''
        Save camchain yaml file with intrinsics, extrinsics and time offset.

        Args:
            resultFile: path to write the yaml file
        
        Reads from:
            self.camera0, self.cameraN   : AslamCamera objects
            self.dataset0, self.datasetN : BagImageDatasetReader objects (for topic names)
            self.T_camN_cam0_Dvs         : dict of optimized extrinsic DVs, indexed by camIdx
            self.timeshiftDvs            : dict of optimized timeshift DVs, indexed by camIdx
            self.camIdx                  : index of non-reference camera
        '''

        def camDict(cam, dataset, name):
            P = cam.geometry.projection()

            # detect camera and distortion type from projection class
            if isinstance(P, acv.DistortedPinholeProjection):
                cam_type = 'pinhole'
                dist_type = 'radtan'
            elif isinstance(P, acv.EquidistantPinholeProjection):
                cam_type = 'pinhole'
                dist_type = 'equidistant'
            elif isinstance(P, acv.FovPinholeProjection):
                cam_type = 'pinhole'
                dist_type = 'fov'
            elif isinstance(P, acv.PinholeProjection):
                cam_type = 'pinhole'
                dist_type = 'none'
            elif isinstance(P, acv.OmniProjection):
                cam_type = 'omni'
                dist_type = 'none'
            else:
                raise RuntimeError("Unknown camera projection type: {}".format(type(P)))

            # intrinsics
            if cam_type == 'pinhole':
                intrinsics = [P.fu(), P.fv(), P.cu(), P.cv()]
            else:
                intrinsics = [P.xi(), P.fu(), P.fv(), P.cu(), P.cv()]

            # distortion coefficients
            dist_coeffs = P.distortion().getParameters().flatten().tolist() if hasattr(P, 'distortion') else []

            return {
                'name': name,
                'rostopic': dataset.topic,
                'camera_model': cam_type,
                'distortion_model': dist_type,
                'intrinsics': intrinsics,
                'distortion_coeffs': dist_coeffs,
                'resolution': [P.ru(), P.rv()]
            }

        # optimized extrinsic and timeshift for camN
        T_camN_cam0_opt = self.T_camN_cam0_Dvs[self.camIdx].T()
        tau_opt = float(self.timeshiftDvs[self.camIdx].toScalar())

        camchain = {
            'cameras': [
                camDict(self.camera0, self.dataset0, 'cam0'),
                camDict(self.cameraN, self.datasetN, 'cam{}'.format(self.camIdx))
            ],
            'baselines': [
                {
                    'cam0_to_cam{}'.format(self.camIdx): T_camN_cam0_opt.tolist()
                }
            ],
            'time_shift_cam0_to_cam{}'.format(self.camIdx): tau_opt
        }

        with open(resultFile, 'w') as f:
            yaml.dump(camchain, f, default_flow_style=False)

        print("Camera chain saved to {}".format(resultFile))
        print("  Optimized time shift cam0 to cam{}: {:.6f} s".format(self.camIdx, tau_opt))

    def buildAndSolveProblem(self,
                            splineOrder=6,
                            poseKnotsPerSecond=70,
                            maxIterations=20,
                            timeOffsetPadding=0.02,
                            blakeZisserCam=-1,
                            verbose=False):

        print("\tSpline order: %d" % splineOrder)
        print("\tPose knots per second: %d" % poseKnotsPerSecond)
        print("\tMax iterations: %d" % maxIterations)
        print("\tTime offset padding: %f" % timeOffsetPadding)

        #######################
        ## timeshift prior
        #######################
        self.findTimeshiftPrior(verbose=verbose)

        if abs(self.timeshiftPrior) < 1e-4:
            t_cam0_first = self.target0Observations[0].time().toSec()
            t_camN_first = self.targetNObservations[0].time().toSec()
            self.timeshiftPrior = t_camN_first - t_cam0_first
            print("  [warn] cross-correlation unreliable, initializing tau from header timestamps: %.6fs" % self.timeshiftPrior)

        #######################
        ## cam0 pose spline
        #######################
        splinePadding = abs(self.timeshiftPrior) + timeOffsetPadding
        poseSpline = self.initPoseSpline(self.target0Observations, splineOrder, poseKnotsPerSecond, timeOffsetPadding=splinePadding, label="cam0_main")

        #######################
        ## design variables
        #######################
        poseSplineDv = asp.BSplinePoseDesignVariable(poseSpline)
        self.addDesignVariables(poseSplineDv)
        self.addCameraDesignVariables(self.camIdx, self.T_camN_cam0, self.timeshiftPrior)

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
            applyFrameTimeShift=False
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
            timeOffsetPadding=abs(self.timeshiftPrior) + timeOffsetPadding,
            applyFrameTimeShift=True
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
        #populate the residual reproj errors
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
        ## return results
        ########################
        T_camN_cam0_opt = sm.Transformation(self.T_camN_cam0_Dvs[self.camIdx].T())
        tau_opt = self.timeshiftDvs[self.camIdx].toScalar()
        return T_camN_cam0_opt, tau_opt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_exit)

    # bagfile = '/data/merged_payload4b/alternateSkipped.bag'
    bagfile = '/data/merged_payload4b/decompressed_ros1.bag'
    cam0YamlFile = '/data/cam0.yaml'
    cam1YamlFile = '/data/cam1.yaml'
    targetYamlFile = '/data/aprilgrid.yaml'

    # load configs
    cam0Config = kc.CameraParameters(cam0YamlFile)
    camNConfig = kc.CameraParameters(cam1YamlFile)
    targetConfig = kc.CalibrationTargetParameters(targetYamlFile)

    # load datasets
    dataset0 = initBagDataset(bagfile, '/cam_sync/cam0/image_raw', None, None)
    datasetN = initBagDataset(bagfile, '/cam_sync/cam1/image_raw', None, None)

    # initialize calibrator
    calibrator = AsyncCalibrator(
        cam0Config, camNConfig,
        camIdx=1,
        targetConfig=targetConfig,
        dataset0=dataset0,
        datasetN=datasetN,
        reprojectionSigma=1.0,
        showCorners=False,
        showReproj=False,
        showOneStep=False
    )

    # build and solve
    T_camN_cam0, tau = calibrator.buildAndSolveProblem(
        splineOrder=6,
        poseKnotsPerSecond=70,
        maxIterations=20,
        timeOffsetPadding=0.02,
        blakeZisserCam=-1,
        verbose=True
    )

    # write results to yaml
    bagtag = os.path.splitext(os.path.basename(bagfile))[0]
    resultFile = bagtag + "-camchain.yaml"
    calibrator.saveCamChainYaml(resultFile)
    print("Results written to:")
    print("  Saving camera chain calibration to file: {0}".format(resultFile))

    print("")
    print("Calibration results:")
    print("T_cam{0}_cam0:".format(calibrator.camIdx))
    print(T_camN_cam0.T())
    print("Time shift cam0 to cam{0} (t_cam{0} = t_cam0 + shift):".format(calibrator.camIdx))
    print(tau)

        

                                 











