import argparse 
import sm
import sys
import os 

from models import cameraModels

## Command line argument parser 
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
    Calibrate extrinsics and time offsets of an asynchronous camera system using an AprilGrid.
    Camera 0 is the reference. All other cameras are calibrated against it.

    %(prog)s --models pinhole-radtan pinhole-radtan --target aprilgrid.yaml \\
              --bag MYROSBAG.bag --topics /cam0/image_raw /cam1/image_raw \\
              --cam-intrinsics cam0.yaml cam1.yaml

    example aprilgrid.yaml:
        target_type: 'aprilgrid'
        tagCols: 6
        tagRows: 9
        tagSize: 0.034
        tagSpacing: 0.3"""

    # parser object
    parser = KalibrArgParser(
        description='Calibrate the extrinsics and time offsets of an asynchronous camera system.',
        usage=usage)

    parser.add_argument('--models', nargs='+', dest='models',
        help='Camera model per topic: {0}'.format(list(cameraModels.keys())),
        required=True)

    # Data source arguments --> bagfile, topics, bag start time, bag frequency
    groupSource = parser.add_argument_group('Data source')
    groupSource.add_argument('--bag', dest='bagfile',
        help='ROS bag file containing the image data', required=True)
    groupSource.add_argument('--topics', nargs='+', dest='topics',
        help='Image topic for each camera, in order (cam0 first)', required=True)
    groupSource.add_argument('--bag-from-to', metavar='bag_from_to', type=float, nargs=2,
        help='Use bag data in this time window [s]')
    groupSource.add_argument('--bag-freq', metavar='bag_freq', type=float,
        help='Subsample bag at this frequency [Hz]')

    # Target arguments --> target yaml file
    groupTarget = parser.add_argument_group('Calibration target')
    groupTarget.add_argument('--target', dest='targetYaml',
        help='Calibration target configuration yaml', required=True)

    # Camera Intrinsic arguments --> instrinsics for each cam in order 
    groupIntrinsics = parser.add_argument_group('Camera intrinsics')
    groupIntrinsics.add_argument('--cam-intrinsics', nargs='+', dest='camIntrinsics',
        help='Kalibr-format intrinsics yaml for each camera, in order', required=True)

    # Calibration settings arguments --> 
    groupCalib = parser.add_argument_group('Calibration settings')
    groupCalib.add_argument('--no-time-calibration', action='store_true',
        dest='noTimeCalibration', default=False,
        help='Fix time offsets at the cross-correlation prior, do not optimize them (default: %(default)s)')
    groupCalib.add_argument('--tau-prior', nargs='+', type=float, dest='tauPrior', default=None,
        help='Override cross-correlation estimate for each non-reference camera [s]. '
             'Provide one value per non-reference camera in order (e.g. --tau-prior 0.05 0.03).')
    groupCalib.add_argument('--spline-order', type=int, dest='splineOrder', default=6,
        help='B-spline order for pose trajectory (default: %(default)s)')
    groupCalib.add_argument('--knots-per-second', type=int, dest='knotsPerSecond', default=100,
        help='Pose spline knots per second (default: %(default)s)')
    groupCalib.add_argument('--max-iter', type=int, dest='maxIterations', default=50,
        help='Maximum optimizer iterations (default: %(default)s)')
    groupCalib.add_argument('--time-offset-padding', type=float, dest='timeOffsetPadding', default=0.02,
        help='Spline boundary padding in seconds (default: %(default)s)')

    # Output arguments --> output directory, verbose, show target extraction video
    groupOutput = parser.add_argument_group('Output')
    groupOutput.add_argument('--output-dir', dest='outputDir', default=None,
        help='Directory for all output files (CSVs, plots, yaml). '
             'Defaults to the directory containing the bag file.')
    groupOutput.add_argument('--verbose', action='store_true', dest='verbose',
        help='Enable verbose output')
    groupOutput.add_argument('--show-extraction', action='store_true', dest='showExtraction',
        help='Show target extraction video (disables parallel processing)')

    # print help if no arguments passed 
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(2)

    try:
        parsed = parser.parse_args()
    except:
        sys.exit(2)

    numCams = len(parsed.topics)  #check number of cameras passed via topics in argument

    if numCams < 2:
        sm.logError("At least 2 topics required (--topics).")
        sys.exit(2)

    if len(parsed.models) != numCams:
        sm.logError("Number of models (%d) must match number of topics (%d)." % (len(parsed.models), numCams))
        sys.exit(2)

    if len(parsed.camIntrinsics) != numCams:
        sm.logError("Number of intrinsics files (%d) must match number of topics (%d)." % (len(parsed.camIntrinsics), numCams))
        sys.exit(2)

    if parsed.tauPrior is not None and len(parsed.tauPrior) != numCams - 1:
        sm.logError("--tau-prior expects %d values (one per non-reference camera), got %d." % (numCams - 1, len(parsed.tauPrior)))
        sys.exit(2)

    # verify camera models passed in argument
    for m in parsed.models:
        if m not in cameraModels:
            sm.logError("Unknown camera model '%s'. Choose from: %s" % (m, list(cameraModels.keys())))
            sys.exit(2)

    # verify bagfile exists
    if not os.path.isfile(parsed.bagfile):
        sm.logError("Bag file not found: %s" % parsed.bagfile)
        sys.exit(2)

    #verify target yaml file exists
    if not os.path.isfile(parsed.targetYaml):
        sm.logError("Target yaml not found: %s" % parsed.targetYaml)
        sys.exit(2)

    # verify camera intrinsic file exists
    for f in parsed.camIntrinsics:
        if not os.path.isfile(f):
            sm.logError("Intrinsics file not found: %s" % f)
            sys.exit(2)

    # if output directory not present --> store in bagfile directory OR make one 
    if parsed.outputDir is None:
        parsed.outputDir = os.path.dirname(os.path.abspath(parsed.bagfile))
    else:
        os.makedirs(parsed.outputDir, exist_ok=True)

    return parsed