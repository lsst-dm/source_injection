# This file is part of source_injection.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import galsim


import lsst.geom as geom
from lsst.pex.exceptions import InvalidParameterError


def inject_sources(exposure, objects, calibFluxRadius=12.0, logger=None):
    """Inject artificial sources into the given exposure

    Parameters
    ----------
    exposure : `lsst.afw.image.exposure.exposure.ExposureF`
        The exposure into which the fake sources should be added
    objects : `typing.Iterator` [`tuple`]
        Iterator of tuples of [`lsst.geom.SpherePoint`, `galsim.GSObject`]]
        specifying where and what surface brightness profile to inject.
    calibFluxRadius : `float`, optional
        Aperture radius (in pixels) used to define the calibration for this
        exposure+catalog.  This is used to produce the correct instrumental
        fluxes within the radius.  The value should match that of the field
        defined in slot_CalibFlux_instFlux.
    logger : `lsst.log.log.log.Log` or `logging.Logger`, optional
        Logger.
    """
    exposure.mask.addMaskPlane("FAKE")
    bitmask = exposure.mask.getPlaneBitMask("FAKE")
    if logger:
        logger.info(f"Adding mask plane with bitmask {bitmask}")

    wcs = exposure.getWcs()
    psf = exposure.getPsf()

    bbox = exposure.getBBox()
    fullBounds = galsim.BoundsI(bbox.minX, bbox.maxX, bbox.minY, bbox.maxY)
    gsImg = galsim.Image(exposure.image.array, bounds=fullBounds)

    pixScale = wcs.getPixelScale().asArcseconds()

    for spt, gsObj in objects:
        pt = wcs.skyToPixel(spt)
        posd = galsim.PositionD(pt.x, pt.y)
        posi = galsim.PositionI(pt.x//1, pt.y//1)
        if logger:
            logger.debug(f"Adding fake source at {pt}")

        mat = wcs.linearizePixelToSky(spt, geom.arcseconds).getMatrix()
        gsWCS = galsim.JacobianWCS(mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1])

        # This check is here because sometimes the WCS
        # is multivalued and objects that should not be
        # were being included.
        gsPixScale = np.sqrt(gsWCS.pixelArea())
        if gsPixScale < pixScale/2 or gsPixScale > pixScale*2:
            continue

        try:
            psfArr = psf.computeKernelImage(pt).array
        except InvalidParameterError:
            # Try mapping to nearest point contained in bbox.
            contained_pt = geom.Point2D(
                np.clip(pt.x, bbox.minX, bbox.maxX),
                np.clip(pt.y, bbox.minY, bbox.maxY)
            )
            if pt == contained_pt:  # no difference, so skip immediately
                if logger:
                    logger.infof(
                        "Cannot compute Psf for object at {}; skipping",
                        pt
                    )
                continue
            # otherwise, try again with new point
            try:
                psfArr = psf.computeKernelImage(contained_pt).array
            except InvalidParameterError:
                if logger:
                    logger.infof(
                        "Cannot compute Psf for object at {}; skipping",
                        pt
                    )
                continue

        apCorr = psf.computeApertureFlux(calibFluxRadius)
        psfArr /= apCorr
        gsPSF = galsim.InterpolatedImage(galsim.Image(psfArr), wcs=gsWCS)

        conv = galsim.Convolve(gsObj, gsPSF)
        stampSize = conv.getGoodImageSize(gsWCS.minLinearScale())
        subBounds = galsim.BoundsI(posi).withBorder(stampSize//2)
        subBounds &= fullBounds

        if subBounds.area() > 0:
            subImg = gsImg[subBounds]
            offset = posd - subBounds.true_center
            # Note, for calexp injection, pixel is already part of the PSF and
            # for coadd injection, it's incorrect to include the output pixel.
            # So for both cases, we draw using method='no_pixel'.

            conv.drawImage(
                subImg,
                add_to_image=True,
                offset=offset,
                wcs=gsWCS,
                method='no_pixel'
            )

            subBox = geom.Box2I(
                geom.Point2I(subBounds.xmin, subBounds.ymin),
                geom.Point2I(subBounds.xmax, subBounds.ymax)
            )
            exposure[subBox].mask.array |= bitmask
