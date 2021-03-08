# -*- coding: utf-8 -*-
#*************************************************************************
#***File Name: flash_class.py
#***Author: Zhonghai Zhao
#***Mail: zhaozhonghi@126.com 
#***Created Time: 2018年03月25日 星期日 14时39分05秒
#*************************************************************************
class flash_class(object):
    '''
    This class contains some function used in flash_visual module.
    '''
    # initialization
    def __init__(self):
        pass
    def line_set(self):
        '''
        This function stored some predefined line plot parameters.
        Parameters:
            None.
        Returns:
            line_list.
        Raises:
            KeyError.
        '''
        line_list = ['r-', 'g--', 'b-.', 'c:', 'k-', 'y--', 'm-.']
        return line_list
    def get_data(self, prefix='lasslab', filenumber=0, keyword='cnt'):
        '''
        This function is used to get file data.
        Parameters:
            prefix         - FLASH output file prefix.
            filenumber     - FLASH output file number.
            keyword        - FLASH output file keyword.
        Returns:
            data.
        Raises:
            KeyError.
        '''
        import h5py as h5
        if (keyword == 'cnt'):
            s = '_hdf5_plt_cnt_'
        elif (keyword == 'part'):
            s = '_hdf5_part_'
        else:
            s = '_hdf5_chk_'
        filename = prefix + s + str(filenumber).zfill(4)
        # read data
        data = h5.File(filename, 'r')

        # return
        return data

    def get_PI(self, prefix='PItest', filenumber=1):
        '''
        This function is used to read data from Proton Imaging data file.
        Parameters:
            prefix         - FLASH output file prefix.
            filenumber     - FLASH output file number.
            keyword        - FLASH output file keyword.
        Returns:
            data.
        Raises:
            KeyError.
        '''
        s = prefix + '_ProtonDetectorFile' + str(filenumber).zfill(2)
        filedata = open(s, 'r')

        xcoord = []
        ycoord = []
        lines = filedata.readlines()
        length = len(lines)
        for i in range(length):
            line_loc = lines[i]
            xcoord.append(float(line_loc[4:20]))
            ycoord.append(float(line_loc[24:40]))

        return [xcoord, ycoord]

    def get_constant(self, field='dens'):
        '''
        This function is used to get normalization constant.
        Parameters:
            field          - physical field.
        Returns:
            constant.
        Raises:
            KeyError.
        '''
        #from constants import flash_Harris_sheet as const
        from constants import laser_slab as const
        #from constants import Harris_with_Particle as const
        normalize = {'dens':const.N0, 'tele':const.T0, 'tion':const.T0, 'trad':const.T0, 'velx':const.V0, 'vely':const.V0, 'velz':const.V0, \
                     'magx':const.B0, 'magy':const.B0, 'magz':const.B0, 'curx':const.J0, 'cury':const.J0, 'curz':const.J0, 'magp':const.P0, 'pres':const.P0, \
                     'totp':const.P0}
        keys = normalize.keys()
        ifinkeys = False
        for each in keys:
            if (each == field):
                constant = normalize[each]
                ifinkeys = True
        if (ifinkeys == True):
            return constant
        else:
            print('field selected in not in data file.')
            return 1.0
    def get_block(self, data, refine=0, geom=[0., 1., 0, 1., 0, 1], box_scale=0.8, box_panning=[0.1, 0.1, 0.1]):
        '''
        This function is used to find out the necessary block to plot.
        Parameters:
            data           - FLASH output file data.
            refine         - refine level.
            box_scale      - total box scale.
            box_panning    - box panning.
            geom           - geometry axis.
        Returns:
            block.
        Raises:
            KeyError.
        '''
        x = float(geom[1] - geom[0])
        y = float(geom[3] - geom[2])
        z = float(geom[5] - geom[4])
        xmin = geom[0]
        ymin = geom[2]
        zmin = geom[4]
        refine_level = data[u'refine level'][:]
        max_level = max(refine_level)
        length = refine_level.shape[0]
        bounding_box = data[u'bounding box'][:]
        block = []
        if (refine == 0):
            #for j in range(max_level, 0, -1):
            for i in range(length):
                if (i != length-1):
                    ifenter_1 = refine_level[i] == max_level
                    ifenter_2 = (refine_level[i+1] <= refine_level[i]) and (refine_level[i] != max_level)
                    ifenter = ifenter_1 or ifenter_2
                else:
                    ifenter = True
                if (ifenter == True):
                    info = [i]
                    info.append(refine_level[i])
                    info.append(round((bounding_box[i][0, 0] - xmin)/x*box_scale + box_panning[0], 5))
                    info.append(round((bounding_box[i][1, 0] - ymin)/y*box_scale + box_panning[1], 5))
                    info.append(round((bounding_box[i][2, 0] - zmin)/z*box_scale + box_panning[2], 5))
                    info.append(round((bounding_box[i][0, 1] - bounding_box[i][0, 0])/x*box_scale, 5))
                    info.append(round((bounding_box[i][1, 1] - bounding_box[i][1, 0])/y*box_scale, 5))
                    info.append(round((bounding_box[i][2, 1] - bounding_box[i][2, 0])/z*box_scale, 5))
                    block.append(info)
        else:
            for i in range(length):
                if (refine_level[i] == refine):
                    info = [i]
                    info.append(refine_level[i])
                    info.append(round((bounding_box[i][0, 0] - xmin)/x*box_scale + box_panning[0], 5))
                    info.append(round((bounding_box[i][1, 0] - ymin)/y*box_scale + box_panning[1], 5))
                    info.append(round((bounding_box[i][2, 0] - zmin)/z*box_scale + box_panning[2], 5))
                    info.append(round((bounding_box[i][0, 1] - bounding_box[i][0, 0])/x*box_scale, 5))
                    info.append(round((bounding_box[i][1, 1] - bounding_box[i][1, 0])/y*box_scale, 5))
                    info.append(round((bounding_box[i][2, 1] - bounding_box[i][2, 0])/z*box_scale, 5))
                    block.append(info)
        # return
        return block

    def get_figinfo(self, box_scale=0.8, box_panning=[0.1, 0.1, 0.1], geom=[0,1,0,1,0,1], geom_factor=1, axis='z', coordinate=0., ngrid=[8,8,8]):
        '''
        Parameters:
            box_scale      - total box scale.
            box_panning    - box panning.
            geom           - 2d geometry axis.
            geom_factor    - geom factor.
            axis           - coordinate axis.
            coordinate     - coordinate.
            ngrid          - [nxb,nyb,nzb] in flash.
        Returns:
            block.
        Raises:
            KeyError.
        '''
        if (axis == 'x'):
            axes = [box_panning[1], box_panning[2], box_scale, box_scale]
            xlim = [geom[2]*geom_factor, geom[3]*geom_factor]
            ylim = [geom[4]*geom_factor, geom[5]*geom_factor]
            coord = (coordinate - geom[0])/(geom[1]-geom[0]) * box_scale + box_panning[0]
            label = [r'$ \rm Y/mm $', r'$ \rm Z/mm $']
            grids = [ngrid[1], ngrid[2]]
        elif (axis == 'y'):
            axes = [box_panning[0], box_panning[2], box_scale, box_scale]
            xlim = [geom[0]*geom_factor, geom[1]*geom_factor]
            ylim = [geom[4]*geom_factor, geom[5]*geom_factor]
            coord = (coordinate - geom[2])/(geom[3]-geom[2]) * box_scale + box_panning[1]
            label = [r'$ \rm X/mm $', r'$ \rm Z/mm $']
            grids = [ngrid[0], ngrid[2]]
        elif (axis == 'z'):
            axes = [box_panning[0], box_panning[1], box_scale, box_scale]
            xlim = [geom[0]*geom_factor, geom[1]*geom_factor]
            ylim = [geom[2]*geom_factor, geom[3]*geom_factor]
            coord = (coordinate - geom[4])/(geom[5]-geom[4]) * box_scale + box_panning[2]
            label = [r'$ \rm X/mm $', r'$ \rm Y/mm $']
            grids = [ngrid[0], ngrid[1]]
        else:
            print("No match direction")
        return (axes, xlim, ylim, label, coord, grids)

    def get_plane(self, field_data=[], block=[], dimension=3, axis='z', coordinate=0., ngrid=[8,8,8], ifexist=True):
        '''
        Parameters:
            field_data     - FLASH output file data.
            block          - data block.
            dimension      - dimensions.
            axis           - coordinate axis.
            coordinate     - coordinate.
            ngrid          - number of grid in each block.
            ifexist        - if field exist in data file.
        Returns:
            block.
        Raises:
            KeyError.
        '''
        import numpy as np

        length = len(block)
        narray = []
        for i in range(length):
            array_loc = []
            n = block[i][0]
            if (dimension == 2):
                if (axis == 'x'):
                    array_loc.append([block[i][3], block[i][4], block[i][6], block[i][7]])
                    if (ifexist == True):
                        array_loc.append(field_data[n, :,:,0])
                    else:
                        array_loc.append(field_data)
                    narray.append(array_loc)
                elif (axis == 'y'):
                    array_loc.append([block[i][2], block[i][4], block[i][5], block[i][7]])
                    if (ifexist == True):
                        array_loc.append(field_data[n, :,0,:])
                    else:
                        array_loc.append(field_data)
                    narray.append(array_loc)
                elif (axis == 'z'):
                    array_loc.append([block[i][2], block[i][3], block[i][5], block[i][6]])
                    if (ifexist == True):
                        array_loc.append(field_data[n, 0,:,:])
                    else:
                        array_loc.append(field_data)
                    narray.append(array_loc)
                else:
                    print("Error Direction Selected!")
            elif (dimension == 3):
                x0 = block[i][2]
                x1 = block[i][2] + block[i][4]
                y0 = block[i][3]
                y1 = block[i][3] + block[i][5]
                if (axis == 'x'):
                    x0 = block[i][2]
                    x1 = block[i][2] + block[i][5]
                    if ((coordinate >= x0) and (coordinate < x1)):
                        index = int(round((coordinate - x0)/(x1 - x0)*(ngrid[0] - 1)))
                        array_loc.append([block[i][3], block[i][4], block[i][6], block[i][7]])
                        array_loc.append(field_data[n, :,:,index])
                        narray.append(array_loc)
                elif (axis == 'y'):
                    y0 = block[i][3]
                    y1 = block[i][3] + block[i][6]
                    if ((coordinate >= y0) and (coordinate < y1)):
                        index = int(round((coordinate - y0)/(y1 - y0)*(ngrid[1] - 1)))
                        array_loc.append([block[i][2], block[i][4], block[i][5], block[i][7]])
                        array_loc.append(field_data[n, :,index,:])
                        narray.append(array_loc)
                elif (axis == 'z'):
                    z0 = block[i][4]
                    z1 = block[i][4] + block[i][7]
                    if ((coordinate >= z0) and (coordinate < z1)):
                        index = int(round((coordinate - z0)/(z1 - z0)*(ngrid[2] - 1)))
                        array_loc.append([block[i][2], block[i][3], block[i][5], block[i][6]])
                        array_loc.append(field_data[n, index,:,:])
                        narray.append(array_loc)
                else:
                    print("Error Direction Selected!")
        return narray
    def get_line(self, narray=[], dimension=3, box_scale=0.8, box_panning=[0.1, 0.1, 0.1], geom=[0, 1, 0, 1, 0, 1],  axis=[], coordinate=0., ngrid=[8, 8, 8]):
        '''
        This function is used to calculate the line-data in FLASH output file.
        Parameters:
            narray         - 2d plane data, obtained from get_plane.
            dimension      - dimensions.
            box_scale      - total box scale.
            box_panning    - box panning.
            geom           - geometry axis.
            axis           - coordinate axis.
            coordinate     - coordinate.
            ngrid          - number of grid in each block.
        Returns:
            array.
        Raises:
            KeyError.
        '''
        import numpy as np

        length = len(narray)
        nline = []
        for i in range(length):
            bound_size = narray[i][0]
            array = narray[i][1]
            #if_in_block = False
            # set domain
            x0 = bound_size[0]
            x1 = bound_size[0] + bound_size[2]
            y0 = bound_size[1]
            y1 = bound_size[1] + bound_size[3]
            if (dimension == 2):
                if (axis[1] == 'x'):
                    coord = (coordinate - geom[0])/(geom[1]-geom[0]) * box_scale + box_panning[0]
                    if ((coord > x0) and (coord <= x1)):
                        index = int(round((coord - x0)/(x1 - x0)*(ngrid[0] - 1)))
                        data_list = []
                        data_list.append(np.linspace(y0, y1, ngrid[0]))
                        data_list.append(array[:, index])
                        nline.append(data_list)
                elif (axis[1] == 'y'):
                    coord = (coordinate - geom[2])/(geom[3]-geom[2]) * box_scale + box_panning[1]
                    if ((coord > y0) and (coord <= y1)):
                        index = int(round((coord - y0)/(y1 - y0)*(ngrid[1] - 1)))
                        data_list = []
                        data_list.append(np.linspace(x0, x1, ngrid[1]))
                        data_list.append(array[index, :])
                        nline.append(data_list)
            elif (dimension == 3):
                if (axis[1] == 'x'):
                    coord = (coordinate - geom[0])/(geom[1]-geom[0]) * box_scale + box_panning[0]
                    if ((coord > x0) and (coord <= x1)):
                        index = int(round((coord - x0)/(x1 - x0)*(ngrid[0] - 1)))
                        data_list = []
                        data_list.append(np.linspace(y0, y1, ngrid[0]))
                        data_list.append(array[:, index])
                        nline.append(data_list)
                elif (axis[1] == 'y'):
                    coord = (coordinate - geom[2])/(geom[3]-geom[2]) * box_scale + box_panning[1]
                    if (axis[0] == 'x'):
                        if ((coord > x0) and (coord <= x1)):
                            index = int(round((coord - x0)/(x1 - x0)*(ngrid[0] - 1)))
                            data_list = []
                            data_list.append(np.linspace(y0, y1, ngrid[0]))
                            data_list.append(array[index, :])
                            nline.append(data_list)
                    elif (axis[0] == 'z'):
                        if ((coord > y0) and (coord <= y1)):
                            index = int(round((coord - y0)/(y1 - y0)*(ngrid[1] - 1)))
                            data_list = []
                            data_list.append(np.linspace(x0, x1, ngrid[1]))
                            data_list.append(array[index, :])
                            nline.append(data_list)
                elif (axis[1] == 'z'):
                    coord = (coordinate - geom[4])/(geom[5]-geom[4]) * box_scale + box_panning[2]
                    if ((coord > y0) and (coord <= y1)):
                        index = int(round((coord - y0)/(y1 - y0)*(ngrid[2] - 1)))
                        data_list = []
                        data_list.append(np.linspace(x0, x1, ngrid[2]))
                        data_list.append(array[index, :])
                        nline.append(data_list)
            #    pass
        return nline
    def get_coord(self, axis='x', coordinate=0, geom=[0, 1., 0., 1.], geom_factor=10000, box_scale=0.8, box_panning=[0.1, 0.1]):
        '''
        This function is used to calculate coordinate in figure frame.
        Parameters:
            axis           - coordinate axis.
            coordinate     - coordinate.
            geom           - geometry axis.
            geom_factor    - geometry axis factor.
            box_scale      - total box scale.
            box_panning    - box panning
        Returns:
            coord.
        Raises:
            KeyError.
        '''
        coord = []
        x = float(geom[1] - geom[0])
        y = float(geom[3] - geom[2])
        xmin = geom[0]
        ymin = geom[2]

        if (axis == 'x'):
            #coord.append((coordinate-xmin)/x*geom_factor*box_scale + box_panning[0])
            coord.append((coordinate-xmin)/x*box_scale + box_panning[0])
            coord.append([geom[2]*geom_factor, geom[3]*geom_factor])
            coord.append('$ Y/\mu m $')
        elif (axis == 'y'):
            #coord.append((coordinate-ymin)/y*geom_factor*box_scale + box_panning[1])
            coord.append((coordinate-ymin)/y*box_scale + box_panning[1])
            coord.append([geom[0]*geom_factor, geom[1]*geom_factor])
            coord.append('$ X/\mu m $')
        else:
            pass
        return coord
    def reconstruct(self, nline=[], axis=[], ngrid=[8,8,8], geom=[0, 1, 0, 1, 0, 1], geom_factor=10, box_scale=0.8, box_panning=[0.1, 0.1, 0.1]):
        '''
        This function is used to re-construct line-data in FLASH output file.
        Parameters:
            nline          - list obtained from get_line.
            axis           - coordinate axis.
            ngrid          - number of grid in each block.
            geom           - geometry axis.
            geom_factor    - geometry axis factor.
            box_scale      - total box scale.
            box_panning    - box panning.
        Returns:
            (x, y).
        Raises:
            KeyError.
        '''
        import numpy as np

        grids = 0
        for i in 'xyz':
            if (i not in axis):
                break
            grids = grids + 1
        grid_index = ngrid[grids]

        length = len(nline)
        if (length == 0):
            print('Nothing to be reconstructed!')
        else:
            x = []
            y = []
            for j in range(length):
                if (j == 0):
                    for i in range(grid_index-1):
                        x.append(nline[j][0][i])
                        y.append(nline[j][1][i])
                elif (j == length-1):
                    for i in range(grid_index):
                        if (i == 0):
                            x.append(nline[j][0][i])
                            y.append((nline[j][1][i] + nline[j-1][1][grid_index-1])/2.0)
                        else:
                            x.append(nline[j][0][i])
                            y.append(nline[j][1][i])
                else:
                    for i in range(grid_index-1):
                        if (i == 0):
                            x.append(nline[j][0][i])
                            y.append((nline[j][1][i] + nline[j-1][1][grid_index-1])/2.0)
                        else:
                            x.append(nline[j][0][i])
                            y.append(nline[j][1][i])
        # reconstruct
        x = list(((np.array(x) - box_panning[grids])/box_scale * (geom[grids*2+1] - geom[grids*2]) + geom[grids*2]) * geom_factor)
        y = np.array(y)
        return [x, y]
    def compute_energy(self, field_data=[], block=[], ngrid=16):
        '''
        This function is used to compute kinetic, inertial, magnetic, total energy.
        Parameters:
            field_data     - FLASH output file data.
            block          - data block.
            ngrid          - number of grid in each block.
        Returns:
            energy.
        Raises:
            KeyError.
        '''
        import numpy as np

        length = len(block)
        energy = [0, 0, 0, 0, 0]

        blocksize = field_data[u'block size'][:]
        dens = field_data[u'dens'][:]
        vx = field_data[u'velx'][:]
        vy = field_data[u'vely'][:]
        vz = field_data[u'velz'][:]
        bx = field_data[u'magx'][:]
        by = field_data[u'magy'][:]
        bz = field_data[u'magz'][:]
        eint = field_data[u'eint'][:]
        for i in range(length):
            n = block[i][0]
            dx = blocksize[n][0]/float(ngrid)
            dy = blocksize[n][1]/float(ngrid)
            ds = dx*dy

            idens = dens[n]
            # kinetic energy 0.5*rho*(v^2)
            ivx = vx[n]
            ivy = vy[n]
            ivz = vz[n]
            kinetic_energy = 0.5 * np.sum(idens*(ivx*ivx + ivy*ivy + ivz*ivz))
            energy[0] = energy[0] + kinetic_energy*ds
            # internal energy P/(gamma - 1)
            ieint = eint[n]
            internal_energy = np.sum(ieint*idens)
            energy[1] = energy[1] + internal_energy*ds
            # magnetic energy 0.5*(B^2)
            ibx = bx[n]
            iby = by[n]
            ibz = bz[n]
            magnetic_energy = 0.5 * np.sum(ibx*ibx + iby*iby + ibz*ibz)
            energy[2] = energy[2] + magnetic_energy*ds
            # plasma energy (kinetic + internal)
            energy[3] = energy[3] + (kinetic_energy + internal_energy)*ds
            # total energy (kinetic + internal + magnetic)
            energy[4] = energy[4] + (kinetic_energy + internal_energy + magnetic_energy)*ds

        return energy
    #
    def compute_field(self, field_data=[], block=[], field='electric_field', component='z', resistivity=0.):
        '''
        This function is used to compute a series of fluid / EM field data.
        Parameters:
            field_data     - FLASH output file data.
            block          - data block.
            field          - field label to be computed.
            component      - vector component('x', 'y', or 'z').
            resistivity    - electric resistivity.
        Returns:
            array.
        Raises:
            KeyError.
        '''

        import numpy as np
        import computation_class

        compu = computation_class.computation_class()

        length = len(block)
        blocksize = field_data[u'block size'][:]
        array = []
        for i in range(length):
            n = block[i][0]
            # select case
            if (field == 'electric_field'):
                # select component
                if (component == 'x'):
                    # ex = -(vy*Bz - vz*By) + eta*Jx
                    vy = field_data[u'vely'][:][n,0,:,:]
                    vz = field_data[u'velz'][:][n,0,:,:]
                    by = field_data[u'magy'][:][n,0,:,:]
                    bz = field_data[u'magz'][:][n,0,:,:]
                    jx = field_data[u'curx'][:][n,0,:,:]
                    array.append(-(vy*bz-vz*by) + resistivity*jx)
                elif (component == 'y'):
                    # ey = -(vz*Bx - vx*Bz) + eta*Jy
                    vx = field_data[u'velx'][:][n,0,:,:]
                    vz = field_data[u'velz'][:][n,0,:,:]
                    bx = field_data[u'magx'][:][n,0,:,:]
                    bz = field_data[u'magz'][:][n,0,:,:]
                    jy = field_data[u'cury'][:][n,0,:,:]
                    array.append(-(vz*bx-vx*bz) + resistivity*jy)
                else:
                    # ez = -(vx*By - vy*Bx) + eta*Jz
                    vx = field_data[u'velx'][:][n,0,:,:]
                    vy = field_data[u'vely'][:][n,0,:,:]
                    bx = field_data[u'magx'][:][n,0,:,:]
                    by = field_data[u'magy'][:][n,0,:,:]
                    jz = field_data[u'curz'][:][n,0,:,:]
                    array.append(-(vx*by-vy*bx) + resistivity*jz)
            elif (field == 'tension_force'):
                bx = field_data[u'magx'][:][n,0,:,:]
                by = field_data[u'magy'][:][n,0,:,:]
                light_c = 3e10 #cm/s
                shape = bx.shape
                size = blocksize[n]
                dx = size[0]/float(shape[0])
                dy = size[1]/float(shape[1])
                if (component == 'x'):
                    dxbx = compu.derivate(array=bx, delta=dx, order=1, direction='x')
                    dybx = compu.derivate(array=bx, delta=dy, order=1, direction='y')
                    array.append(bx*dxbx + by*dybx)
                elif (component == 'y'):
                    dxby = compu.derivate(array=by, delta=dx, order=1, direction='x')
                    dyby = compu.derivate(array=by, delta=dy, order=1, direction='y')
                    array.append(bx*dxby + by*dyby)
                else:
                    pass
            elif (field == 'magnetic_pressure_force'):
                bx = field_data[u'magx'][:][n,0,:,:]
                by = field_data[u'magy'][:][n,0,:,:]
                bp = bx**2 + by**2 
                light_c = 3e10 #cm/s
                shape = bp.shape
                size = blocksize[n]
                dx = size[0]/float(shape[0])
                dy = size[1]/float(shape[1])
                if (component == 'x'):
                    dxbp = compu.derivate(array=bp, delta=dx, order=1, direction='x')
                    array.append(-dxbp)
                elif (component == 'y'):
                    dybp = compu.derivate(array=bp, delta=dy, order=1, direction='y')
                    array.append(-dybp)
                else:
                    pass
            elif (field == 'thermal_pressure_force'):
                pp = field_data[u'pres'][:][n,0,:,:]
                shape = pp.shape
                size = blocksize[n]
                dx = size[0]/float(shape[0])
                dy = size[1]/float(shape[1])
                if (component == 'x'):
                    dxpp = compu.derivate(array=pp, delta=dx, order=1, direction='x')
                    array.append(-dxpp)
                elif (component == 'y'):
                    dypp = compu.derivate(array=pp, delta=dy, order=1, direction='y')
                    array.append(-dypp)
                else:
                    pass
            elif (field == 'total_pressure_force'):
                tp = field_data[u'totp'][:][n,0,:,:]
                shape = tp.shape
                size = blocksize[n]
                dx = size[0]/float(shape[0])
                dy = size[1]/float(shape[1])
                if (component == 'x'):
                    dxtp = compu.derivate(array=tp, delta=dx, order=1, direction='x')
                    array.append(-dxtp)
                elif (component == 'y'):
                    dytp = compu.derivate(array=tp, delta=dy, order=1, direction='y')
                    array.append(-dytp)
                else:
                    pass
            else:
                pass

        return array
    #

    def get_prop(self, data=[], prop='mass'):
        '''
        This function is used to get particle property index from particle file.
        Parameters:
            data           - file data.
            prop           - particle property.
        Returns:
            index
        Raises:
            KeyError.
        '''

        props = data[u'property'][:]
        length = len(props)

        index = -1
        for i in range(length):
            #strp = props[i][0].replace(' ','')
            strp = props[i][0].decode().replace(' ','')
            if (strp == prop):
                index = i
                break
        if (index == -1):
            print('No such property in File!')
        else:
            pass

        return index

    def get_part_prop(self, prefix='Lorentz', filelist=[], prop='mass', npart=0):
        '''
        This function is used to get particle property from particle file.
        Parameters:
            prefix         - file prefix.
            namelist       - file name list.
            prop           - particle property.
            npart          - index of particle.
        Returns:
            array.
        Raises:
            KeyError.
        '''
        import numpy as np

        data = flash_class.get_data(self, prefix=prefix, keyword='part', filenumber=filelist[0]) 
        index = flash_class.get_prop(self, data=data, prop=prop)
        tag_index = flash_class.get_prop(self, data=data, prop='tag')

        length = len(filelist)
        numprop = []
        for k in range(length):
            data = flash_class.get_data(self, prefix=prefix, keyword='part', filenumber=filelist[k])
            part = data[u'particle'][:]
            part = part[part[:,tag_index].argsort()]
            numprop.append(part[npart, index])


        return np.array(numprop)

    def flash_enspe(self, filenumber=0, prefix='turbPI_mhd_2d', keyword='part', bins=300, iflog=False, ifnormed=True):
        '''
        This function is used to compute particle energy spectrum from FLASH data file.
        Parameters:
            prefix         - file prefix.
            filenumber     - file number.
            keyword        - key word of property.
            bins           - bins of np.histogram.
            iflog          - if use log before histogram.
            ifnormed       - if normed in np.histogram.
        Returns:
            enspe(x, Fx).
        Raises:
            KeyError.
        '''
        import numpy as np

        data = flash_class.get_data(self, prefix=prefix, keyword=keyword, filenumber=filenumber)

        part = data[u'particle'][:]

        index_mass = flash_class.get_prop(self, data=data, prop='mass')
        mass = part[:,index_mass]
        index_velx = flash_class.get_prop(self, data=data, prop='velx')
        velx = part[:,index_velx]
        index_vely = flash_class.get_prop(self, data=data, prop='vely')
        vely = part[:,index_vely]
        index_velz = flash_class.get_prop(self, data=data, prop='velz')
        velz = part[:,index_velz]

        qe = 1.602176462e-19
        ek = 0.5 * mass * (velx**2 + vely**2 + velz**2) * 1.e-4 / qe

        if (iflog == True):
            #ek = np.log10(ek)
            ek = np.log(ek)
            Fx,x = np.histogram(ek, bins=bins, normed=ifnormed)
            for k in range(len(Fx)):
                #Fx[k] = Fx[k] / (10**x[k+1] - 10**x[k])
                Fx[k] = Fx[k] / (np.e**x[k+1] - np.e**x[k])
            x[:] = np.e**x[:]
        else:
            Fx,x = np.histogram(ek, bins=bins, normed=ifnormed)

        x = 0.5 * (x[1:] + x[:-1])

        return (x, Fx)

    #
    def particle_to_grid(self, prefix='Harris_mhd_2d', filenumber=0, prop='mass', geom=[0,1,0,1,0,1], shape=[512,512]):

        '''
        This function is used to put particle number to grid density.
        Parameters:
            prefix         - file prefix.
            filenumber     - file number.
            prop           - particle property.
            geom           - box geometry.
            shape          - shape of array to hold property.
        Returns:
            array.
        Raises:
            KeyError.
        '''
        import numpy as np

        # get data
        data = flash_class.get_data(self, prefix=prefix, keyword='part', filenumber=filenumber)
        particle = data[u'particle'][:]
        index = flash_class.get_prop(self, data=data, prop='posx')
        xcoord = particle[:,index]
        index = flash_class.get_prop(self, data=data, prop='posy')
        ycoord = particle[:,index]
        index = flash_class.get_prop(self, data=data, prop=prop)
        mainpart = particle[:,index]

        # set coordinates and array
        nx = shape[1]
        ny = shape[0]
        dx = (geom[1] - geom[0])/nx
        dy = (geom[3] - geom[2])/ny
        x_min = geom[0] -1.5 * dx
        y_min = geom[2] -1.5 * dy

        # shape function
        gx = [0,0,0]
        gy = [0,0,0]
        wx = [0., 0., 0.]
        wy = [0., 0., 0.]

        # put particle to grid
        array = np.zeros([ny, nx])
        npart = np.zeros([ny, nx])
        length = particle.shape[0]

        for i in range(length):
            # x direcion
            x_par = xcoord[i]
            ix = (x_par - x_min)/dx - 1.5
            ix_int = int(round(ix))
            gx = [ix_int-1, ix_int, ix_int+1]
            # y direcion
            y_par = ycoord[i]
            iy = (y_par - y_min)/dy - 1.5
            iy_int = int(round(iy))
            gy= [iy_int-1, iy_int, iy_int+1]

            # center
            wx[1] = 0.75 - (ix - gx[1])**2
            wy[1] = 0.75 - (iy - gy[1])**2
            # side
            for ng in range(0, 3, 2):
                wx[ng] = 0.5 * (1.5 - np.abs(ix - gx[ng]))**2
                wy[ng] = 0.5 * (1.5 - np.abs(iy - gy[ng]))**2
            # add to array
            for ip in range(3):
                for iq in range(3):
                    gx_loc = gx[iq]
                    gy_loc = gy[ip]
                    if ((gx_loc >=0) and (gx_loc < nx) and (gy_loc >= 0) and (gy_loc < ny)):
                        weight = wx[iq] * wy[ip]
                        #array[ix_int+iq-1, iy_int+ip-1] = array[ix_int+iq-1, iy_int+ip-1] + weight*mainpart[i]
                        #npart[ix_int+iq-1, iy_int+ip-1] = npart[ix_int+iq-1, iy_int+ip-1] + weight
                        array[iy_int+ip-1, ix_int+iq-1] = array[iy_int+ip-1, ix_int+iq-1] + weight*mainpart[i]
                        npart[iy_int+ip-1, ix_int+iq-1] = npart[iy_int+ip-1, ix_int+iq-1] + weight
        for j in range(nx):
            for i in range(ny):
                if (npart[i,j] > 0):
                    array[i,j] = array[i,j] / npart[i,j]
                else:
                    array[i,j] = 0.

        return array

    def grid_UG(self, narray=[], nblock=[12,12,1], ngrid=[256,256,1]):
        '''
        This function convert UG grid to an array.
        Parameters:
            narray         - array list.
            nblock         - blocks in each direction.
            ngrid          - grids of block in each direction.
        Returns:
            array.
        Raises:
            None.
        '''
        import numpy as np

        bx = nblock[0]
        by = nblock[1]
        bz = nblock[2]
        gx = ngrid[0]
        gy = ngrid[1]
        gz = ngrid[2]

        if (len(narray) != bx*by*bz):
            print('Array dimensions not be consistent!')
        else:
            if ( bz == 1 ):
                array = np.zeros([by*gy, bx*gx], np.float)
            else:
                pass
            # convert to uniform array
            total = 0
            for i in range(0, by, 1):
                for j in range(0, bx, 1):
                    x_low = j * gx
                    x_high= (j+1) * gx
                    y_low = (by-i-1) * gy
                    y_high= (by-i) * gy
                    y_low = i * gy
                    y_high= (i+1) * gy

                    array[y_low:y_high, x_low:x_high] = narray[total][0]

                    total = total + 1
        
        return array

    def turb_energy(self, array=[], ngrid=500):
        '''
        Parameters:
            array          - array list.
            ngrid          - length of energy array.
        Returns:
            spec.
        Raises:
            None.
        '''
        import numpy as np

        nx, ny = array.shape

        listk = []
        weight = []

        for j in range(ny):
            for i in range(nx):
                k = np.sqrt(i**2 + j**2)
                listk.append(k)
                weight.append(array[i,j])

        # without weight
        y_number, x = np.histogram(listk, bins=ngrid, normed=False)
        # with weight
        y_energy, x = np.histogram(listk, bins=ngrid, normed=False, weights=weight)
        # compute average energy of each k
        y_unit = np.array(y_energy) / np.array(y_number)

        # compute spectrum
        x = (x[1:] + x[:-1]) / 2.
        y_energy = 2. * np.pi * np.array(x) * np.array(y_unit)

        return (x, y_energy)

    def dynamic_alignment(self, bx=[], by=[], ux=[], uy=[], info='theta', dr=0.1, npoint=1000, ngrid=500):
        '''
        Parameters:
            bx             - Bx array.
            by             - By array.
            ux             - Ux array.
            uy             - Uy array.
            info           - information.
            dr             - spatial dr.
            npoint         - number of sample point.
            ngrid          - length of energy array.
        Returns:
            spec.
        Raises:
            None.
        '''

        import random
        import numpy as np

        nx, ny = bx.shape
        
        dx = float(1. / nx)
        length = int(dr / dx)
        tpi = 2 * np.pi

        list_r = []
        list_dutdb = []
        list_ducdb = []

        for i in range(npoint):
            # point 1
            x1 = random.randint(0, nx-1)
            y1 = random.randint(0, ny-1)
            # point 2
            dnxy = random.randint(0, length)
            rand_theta = random.uniform(0, tpi)
            x2 = x1 + int(dnxy * np.cos(rand_theta))
            y2 = y1 + int(dnxy * np.sin(rand_theta))

            if ((x2 >= nx) or (x2 < 0)):
                continue
            if ((y2 >= ny) or (y2 < 0)):
                continue

            r = np.sqrt((x2-x1)**2 + (y2-y1)**2) * dx
            dbx = bx[x2, y2] - bx[x1, y1]
            dby = by[x2, y2] - by[x1, y1]
            dux = ux[x2, y2] - ux[x1, y1]
            duy = uy[x2, y2] - uy[x1, y1]

            du2 = np.sqrt(dux**2 + duy**2)
            db2 = np.sqrt(dbx**2 + dby**2)
            dutdb = du2 * db2
            ducdb = np.abs(dux*dby - duy*dbx)
            
            if (info == 'theta'):
                list_r.append(r)
                list_dutdb.append(dutdb)
                list_ducdb.append(ducdb)
            else:
                pass

        # compute spectrum
        if (info == 'theta'):
            # without weight
            y_number, x = np.histogram(list_r, bins=ngrid, normed=False)
            # dutdb with weight
            y_dutdb, x = np.histogram(list_r, bins=ngrid, normed=False, weights=list_dutdb)
            # ducdb with weight
            y_ducdb, x = np.histogram(list_r, bins=ngrid, normed=False, weights=list_ducdb)
            # compute average theta  of each r
            y_theta = np.array(y_ducdb) / np.array(y_dutdb)
        else:
            pass

        # compute spectrum
        x = (x[1:] + x[:-1]) / 2.
        #y_theta = np.array(y_theta)


        return (x, y_theta)

    def turb_pdf(self, data=[], info='theta', dr=0.01, npoint=1000, ngrid=500):
        '''
        This function is used to compute turbulence PDF.
        Parameters:
            data           - data array.
            info           - information.
            dr             - spatial dr.
            npoint         - number of sample point.
            ngrid          - length of energy array.
        Returns:
            spec.
        Raises:
            None.
        '''

        import random
        import numpy as np

        nx, ny = data[0].shape

        dx = float(1. / nx)
        length = dr / dx
        tpi = 2 * np.pi

        if (info == 'theta'):
            bx = data[0]
            by = data[1]
            ux = data[2]
            uy = data[3]
        elif (info == 'bx'):
            bx = data[0]
        elif (info == 'by'):
            by = data[0]
        elif (info == 'de'):
            de = data[0]

        list_theta = []
        list_db = []

        for i in range(npoint):
            # point 1
            x1 = random.randint(0, nx-1)
            y1 = random.randint(0, ny-1)
            # theta
            rand_theta = random.uniform(0, tpi)
            # point 2
            dnx = int(length * np.cos(rand_theta))
            dny = int(length * np.sin(rand_theta))
            x2 = x1 + dnx
            y2 = y1 + dny
            # select
            if ((x2 >= nx) or (x2 < 0)):
                continue
            if ((y2 >= ny) or (y2 < 0)):
                continue

            if (info == 'theta'):
                dbx = bx[x2, y2] - bx[x1, y1]
                dby = by[x2, y2] - by[x1, y1]
                dux = ux[x2, y2] - ux[x1, y1]
                duy = uy[x2, y2] - uy[x1, y1]

                du2 = np.sqrt(dux**2 + duy**2)
                db2 = np.sqrt(dbx**2 + dby**2)
                dutdb = du2 * db2
                ducdb = np.abs(dux*dby - duy*dbx)
                theta = np.arcsin(ducdb / (dutdb + 1.e-30))

                list_theta.append(theta)
            elif (info == 'bx'):
                dbx = bx[x2, y2] - bx[x1, y1]
                list_db.append(dbx)
            elif (info == 'by'):
                dby = by[x2, y2] - by[x1, y1]
                list_db.append(dby)
            elif (info == 'de'):
                dde = de[x2, y2] - de[x1, y1]
                list_db.append(dde)
            else:
                pass

        if (info == 'theta'):
            y_theta, x = np.histogram(list_theta, bins=ngrid, normed=False)
            x = (x[1:] + x[:-1]) / 2.
        elif ((info == 'bx') or (info == 'by') or (info == 'de')):
            mean = np.mean(list_db)
            std = np.std(list_db)

            list_db = np.array(list_db)
            list_db = (list_db - mean) / std
            total = len(list_db)

            y_theta, x = np.histogram(list_db, bins=ngrid, normed=False)
            x = (x[1:] + x[:-1]) / 2.
            y_theta = y_theta / total

        return (x, y_theta)

    def epoch_de(self, field_data1=[], field_data2=[]):
        '''
        This function is used to compute De(only in Z direction) in EPOCH simulation.
        Parameters:
            field_data1    - field data from EPOCH, E field and B field.
            field_data2    - field data from EPOCH, particle statistical field, J, U, N.
        Returns:
            D_ele
        Raises:
            None.
        '''

        import numpy as np
        from constants import turbMR as const

        J0 = const.J0
        E0 = const.E0
        # field1
        ez = field_data1['Electric_Field_Ez_averaged'].data
        bx = field_data1['Magnetic_Field_Bx_averaged'].data
        by = field_data1['Magnetic_Field_By_averaged'].data
        # field2
        jz = field_data2['Current_Jz_averaged'].data
        ux = field_data2['Derived_Velocity_Ux_averaged_ele'].data
        uy = field_data2['Derived_Velocity_Uy_averaged_ele'].data

        D_ele = jz * (ez + (ux*by - uy*bx))
        #D_ele = D_ele / (J0 * E0)

        return D_ele

    def condition_de(self, de=[], cond=[], threshold=[1]):
        '''
        This function is used to compute conditionally averaged de.
        Parameters:
            de             - dissipation scalar.
            cond           - condition.
            threshold      - threshold of condition.
        Returns:
            condde
        Raises:
            None.
        '''

        import numpy as np

        # find rms of cond
        nx, ny = cond.shape
        rms = np.sqrt(np.sum(cond **2) / float(nx * ny))
        aver = np.sum(de) / float(nx * ny)
        print(rms)
        print(aver)

        threshold = rms * threshold
        length = len(threshold)
        condde = []

        for k in range(length):
            thred = threshold[k]
            total = 0
            number = 0
            for i in range(nx):
                for j in range(ny):
                    if (np.abs(cond[i,j]) >= thred):
                        total = total + de[i,j]
                        number = number + 1
            if (number == 0):
                condde.append(0)
            else:
                condde.append(total / float(number) / aver)

        return condde

    def potential_phi(self, bx=[], by=[], direction='x'):
        '''
        This function is used to integrate magnetic potential phi.
        Parameters:
            bx             - bx.
            by             - by.
            direction      - integrated direction, 'x' or 'y'.
        Returns:
            phi
        Raises:
            None.
        '''

        import numpy as np

        
        #bx = np.transpose(bx)
        #by = np.transpose(by)

        nx, ny = bx.shape
        phi = np.zeros([nx, ny])
        phi[0,0] = (-by[0,0] + bx[0,0]) / 2.
        #phi[0,0] = bx[0,0]
        if (direction == 'x'):
            for j in range(0, ny):
                if (j == 0):
                    pass
                else:
                    phi[0,j] = phi[0,j-1] - (by[0,j-1] + by[0,j])/2.

                for i in range(1, nx):
                    phi[i,j] = phi[i-1,j] + (bx[i-1,j] + bx[i,j])/2.
        if (direction == 'y'):
            for i in range(0, nx):
                if (i == 0):
                    pass
                else:
                    phi[i,0] = phi[i-1,0] + (bx[i-1,0] + bx[i,0])/2.
                    #phi[i,0] = phi[i-1,0] + bx[i,0]
                    #phi[i,0] = phi[i-1,0] + (bx[i-1,0] + bx[i,0] + bx[i+1,0])/3.

                for j in range(1, ny):
                    phi[i,j] = phi[i,j-1] - (by[i,j-1] + by[i,j])/2.
                    #phi[i,j] = phi[i,j-1] - by[i,j]
                    #phi[i,j] = phi[i,j-1] - (by[i,j-1] + by[i,j] + by[i,j+1])/3.
        # compute bx
        bx_comp = np.zeros([nx, ny])
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                bx_comp[i,j] = (phi[i+1,j] - phi[i-1, j]) / 2.

        # compute by
        by_comp = np.zeros([nx, ny])
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                by_comp[i,j] = -(phi[i,j+1] - phi[i, j-1]) / 2.


        #return (bx_comp, by_comp)
        return phi

    def phi_egenvalue(self, phi=[], threshold=1.5e-4):
        '''
        This function is used to compute egenvalue at neutral point.
        Parameters:
            bx             - bx.
            by             - by.
            threshold      - threshold value for neutral point.
        Returns:
            (xp, lo, ho)
        Raises:
            None.
        '''

        import numpy as np

        xpi = []
        xpj = []
        loi = []
        loj = []
        hoi = []
        hoj = []

        #bx = np.transpose(bx)
        #by = np.transpose(by)
        nx, ny = phi.shape

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                #bx_loc = bx[i,j]
                #by_loc = by[i,j]
                # find nuetral point
                #if ((np.abs(bx_loc) <= threshold) and (np.abs(by_loc) <= threshold)):
                #    # compute Hessian matrix
                #    a = -(by[i,j+1] - by[i,j-1]) / 2.
                #    b =  (bx[i,j+1] - bx[i,j-1]) / 2.
                #    c = -(by[i+1,j] - by[i-1,j]) / 2.
                #    d =  (bx[i+1,j] - bx[i-1,j]) / 2.

                #    sq = np.sqrt((a-d)**2 + 4*b*c)
                #    x1 = (a+d) - sq
                #    x2 = (a+d) + sq

                #    if ((x1 < 0) and (x2 < 0)):
                #        hoi.append(i)
                #        hoj.append(j)
                #    elif ((x1 > 0) and (x2 > 0)):
                #        loi.append(i)
                #        loj.append(j)
                #    elif ( x1*x2 < 0):
                #        xpi.append(i)
                #        xpj.append(j)

                dxphi = (phi[i+1,j+1] - phi[i+1,j-1] + phi[i,j+1] - phi[i,j-1] + phi[i-1,j+1] - phi[i-1,j-1])/ 6. / (phi[i+1,j] + phi[i,j] + phi[i-1,j]) * 3
                dyphi = (phi[i+1,j+1] - phi[i-1,j+1] + phi[i+1,j] - phi[i-1,j] + phi[i+1,j-1] - phi[i-1,j-1])/ 6. / (phi[i+1,j] + phi[i,j] + phi[i-1,j]) * 3

                rate = np.sqrt(dxphi**2 + dyphi**2)

                if (rate < threshold):
                    # compute Hessian matrix
                    a = (phi[i,j+1] + phi[i,j-1] - 2*phi[i,j])
                    b = ((phi[i+1,j+1] - phi[i-1,j+1]) - (phi[i+1,j-1] - phi[i-1,j-1])) / 4.
                    c = b
                    d = (phi[i+1,j] + phi[i-1,j] - 2*phi[i,j])

                    sq = np.sqrt((a-d)**2 + 4*b*c)
                    x1 = (a+d) - sq
                    x2 = (a+d) + sq

                    if ((x1 < 0) and (x2 < 0)):
                        hoi.append(j)
                        hoj.append(i)
                    elif ((x1 > 0) and (x2 > 0)):
                        loi.append(j)
                        loj.append(i)
                    elif ( x1*x2 < 0):
                        xpi.append(j)
                        xpj.append(i)
                
        length = 51.2
        xpi = np.array(xpi) / float(nx / length)
        xpj = np.array(xpj) / float(nx / length)
        loi = np.array(loi) / float(nx / length)
        loj = np.array(loj) / float(nx / length)
        hoi = np.array(hoi) / float(nx / length)
        hoj = np.array(hoj) / float(nx / length)

        return ((xpi, xpj), (loi, loj), (hoi, hoj))
        #return (xpi, xpj)





#############################   end of file ###############################
