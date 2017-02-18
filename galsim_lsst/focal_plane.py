
import galsim
import os
import numpy as np

from galsim.config.output import OutputBuilder

class FocalPlaneBuilder(OutputBuilder):
    """Implements the FocalPlane custom output type.

    This type models a full focal plane including multiple CCD images using coherent patterns
    for things like the PSF and sky level.

    The wcs is taken from a reference wcs (e.g. from a set of Fits files), but can reset the
    pointing position to a different location on the sky.
    """

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """Build the images

        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.
        @param file_num         The current file_num.
        @param image_num        The current image_num.
        @param obj_num          The current obj_num.
        @param ignore           A list of parameters that are allowed to be in config that we can
                                ignore here.  i.e. it won't be an error if they are present.
        @param logger           If given, a logger object to log progress.

        @returns a list of the images built
        """
        if 'nobjects' not in config:
            nobjects = galsim.config.ProcessInputNObjects(base)
            config['nobjects'] = nobjects
        req = { 'nobjects' : int, 
                'nchips' : int,
                'output_style' : str,
              }
        opt = {
                'pointing_ra' : galsim.Angle,
                'pointing_dec' : galsim.Angle,
              }
        ignore = ignore + [ 'reference_wcs', 'reference_image' ]

        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore)
        
        nobjects = kwargs['nobjects']
        nchips = kwargs['nchips']
        output_style = kwargs['output_style']

        base['image']['nobjects'] = nobjects

        if 'eval_variables' not in base:
            base['eval_variables'] = {}

        # Read the reference wcs and image size
        if 'reference_wcs' not in config:
            raise ValueError("reference_wcs is required for FocalPlane output type")
        if 'reference_image' not in config:
            raise ValueError("reference_image is required for FocalPlane output type")
        if 'file_name' not in config['reference_image']:
            raise ValueError("file_name is required for reference_image")

        bounds = galsim.BoundsD()
        self.focal_wcs = []
        self.focal_size = []
        w_pos_list = []
        for chip_num in range(nchips):
            base['chip_num'] = chip_num
            base['eval_variables']['ichip_num'] = chip_num
            wcs_type = config['reference_wcs']['type']
            wcs_builder = galsim.config.wcs.valid_wcs_types[wcs_type]
            wcs = wcs_builder.buildWCS(config['reference_wcs'], base)
            self.focal_wcs.append(wcs)
            file_name = galsim.config.ParseValue(config['reference_image'], 'file_name', base, str)[0]
            if 'dir' in config['reference_image']:
                dir = galsim.config.ParseValue(config['reference_image'], 'dir', base, str)[0]
                file_name = os.path.join(dir,file_name)
            header = galsim.FitsHeader(file_name)
            xsize = header['NAXIS1']
            ysize = header['NAXIS2']
            self.focal_size.append((xsize, ysize))

            im_pos1 = galsim.PositionD(0,0)
            im_pos2 = galsim.PositionD(0,ysize)
            im_pos3 = galsim.PositionD(xsize,0)
            im_pos4 = galsim.PositionD(xsize,ysize)
            w_pos_list.append(wcs.toWorld(im_pos1))
            w_pos_list.append(wcs.toWorld(im_pos2))
            w_pos_list.append(wcs.toWorld(im_pos3))
            w_pos_list.append(wcs.toWorld(im_pos4))

        for w_pos in w_pos_list:
            w_pos._set_aux()
        pointing_x = np.mean([w_pos._x for w_pos in w_pos_list ])
        pointing_y = np.mean([w_pos._y for w_pos in w_pos_list ])
        pointing_z = np.mean([w_pos._z for w_pos in w_pos_list ])
        # TODO: Fix this to work when crossing RA = 0.
        pointing_minra = np.min([w_pos.ra for w_pos in w_pos_list ])
        pointing_maxra = np.max([w_pos.ra for w_pos in w_pos_list ])
        pointing_mindec = np.min([w_pos.dec for w_pos in w_pos_list ])
        pointing_maxdec = np.max([w_pos.dec for w_pos in w_pos_list ])
        pointing_ra = np.arctan2(pointing_y,pointing_x)
        pointing_dec = np.arctan2(pointing_z, np.sqrt(pointing_x**2+pointing_y**2))
        pointing = galsim.CelestialCoord(pointing_ra, pointing_dec)
        proj_list = [ pointing.project(w_pos, projection='gnomonic') for w_pos in w_pos_list ]
        for proj in proj_list: bounds += proj
        base['pointing'] = pointing
        base['eval_variables']['fpointing_ra'] = pointing_ra
        base['eval_variables']['fpointing_dec'] = pointing_dec
        base['eval_variables']['fpointing_minra'] = pointing_minra
        base['eval_variables']['fpointing_maxra'] = pointing_maxra
        base['eval_variables']['fpointing_mindec'] = pointing_mindec
        base['eval_variables']['fpointing_maxdec'] = pointing_maxdec
        base['eval_variables']['ifirst_image_num'] = base['image_num']
        base['eval_variables']['ichip_num'] = '$image_num - first_image_num'
        rmax = np.max([proj.x**2 + proj.y**2 for proj in proj_list])**0.5

        seed_offset = [ 1-k*nobjects for k in range(nimages) ]
        base['stamp']['seed_offset'] = {
            'index_key' : 'image_num',
            'type' : 'List',
            'items' : seed_offset
        }

        if 'meta_params' in base:
            for key in base['meta_params']:
                param = galsim.config.ParseValue(base['meta_params'], key, base, float)
                base['eval_variables']['f' + key] = param
        
        base['eval_variables']['ffocal_xmin'] = bounds.xmin
        base['eval_variables']['ffocal_xmax'] = bounds.xmax
        base['eval_variables']['ffocal_ymin'] = bounds.ymin
        base['eval_variables']['ffocal_ymax'] = bounds.ymax
        base['eval_variables']['ffocal_r'] = {
            'type' : 'Eval',
            'str' : "base['pointing'].project(galsim.CelestialCoord(@gal.ra, @gal.dec))"
        }
        base['eval_variables']['ffocal_rmax'] = rmax

        return galsim.config.BuildImages(nimages, base, image_num, obj_num, logger=logger)

galsim.config.output.RegisterOutputType('FocalPlane', FocalPlaneBuilder())
