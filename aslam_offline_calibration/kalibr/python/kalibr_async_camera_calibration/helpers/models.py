import aslam_cv_backend as acvb

#available camera models
cameraModels = { 'pinhole-radtan': acvb.DistortedPinhole,
                 'pinhole-equi':   acvb.EquidistantPinhole,
                 'pinhole-fov':    acvb.FovPinhole,
                 'omni-none':      acvb.Omni,
                 'omni-radtan':    acvb.DistortedOmni,
                 'eucm-none':      acvb.ExtendedUnified,
                 'ds-none':        acvb.DoubleSphere}