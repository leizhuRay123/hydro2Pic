# -*- coding: utf-8 -*-
#*************************************************************************
#***File Name: binary_io.py
#***Author: Zhonghai Zhao
#***Mail: zhaozhonghi@126.com 
#***Created Time: 2018年03月25日 星期日 14时39分05秒
#*************************************************************************
class binary_io(object):
    '''
    This class contains some functions to read data from sdf file and write data to anothet file.
    '''
    # initialization
    def __init__(self):
        pass
    # write particle information
    def particle_write(self, filenumber=0, subset='tracer_p', particle='tracer_pro', prefix='3', mode=1):
        '''
        This function is used to write particle data to file.
        Parameters:
            filenumber    - sdf file number, an integer or an integer list.
            subset        - particle subset to read an write.
            particle      - particle name.
            prefix        - sdf file prefix.
        Returns:
            None.
        Raises:
            KeyError.
        '''
        #import numpy as np
        import struct
        from epoch_class import epoch_class as mysdf
        sc = mysdf()
        namelist = sc.get_list(filenumber)
        field = ['Px', 'Py', 'Pz']
        position = ['X', 'Y', 'Z']
        for i in range(len(namelist)):
            data_dict = sc.get_data(namelist[i], prefix=prefix)
            # read momentum
            p = []
            for j in range(len(field)):
                data = data_dict['Particles_' + field[j] + '_subset_' + subset + '_' + particle].data
                p = p + [data]
            # read grid
            grid = data_dict['Grid_Particles_subset_' + subset + '_' + particle].data
            weight = data_dict['Particles_Weight_subset_' + subset + '_' + particle].data
            # write data to file
            n = len(p[0])
            if (mode == 1):
                order = [0, 1, 2]
                grid_mul = [1., 1., 1.]
                p_mul = [1., 1., 1.]
            elif (mode == 2):
                order = [1, 0, 2]
                grid_mul = [-1., 1., 1.]
                p_mul = [-1., 1., 1.]
            elif (mode == 3):
                order = [2, 1, 0]
                grid_mul = [1., 1., -1.]
                p_mul = [1., 1., -1.]
            else:
                pass
            files = open(str(namelist[i]) + '.dat', 'wb')
            for j in range(n):
                for k in order:
                    files.write(struct.pack('d', grid[k][j]*grid_mul[k]))
                    files.write(struct.pack('d', p[k][j]*p_mul[k]))
                files.write(struct.pack('d', weight[j]))
            files.close()
    # write field information
    def field_write(self, filenumber=0, field='bx', prefix='1'):
        '''
        This function is used to write field data to file.
        Parameters:
            filenumber    - sdf file number, an integer or an integer list.
            field         - field information.
            prefix        - sdf file prefix.
        Returns:
            None.
        Raises:
            KeyError.
        '''
        import struct
        import numpy as np
        from epoch_class import epoch_class as mysdf
        sc = mysdf()
        namelist = sc.get_list(filenumber)
        data_dict = sc.get_data(namelist[0], prefix=prefix)
        keys = data_dict.keys()
        fields = sc.get_field(field=field, keys=keys)
        narray = []
        for i in range(len(namelist)):
            data_dict = sc.get_data(namelist[i], prefix=prefix)
            data = data_dict[fields].data
            narray.append(np.transpose(data))
        shape = narray[0].shape
        dims = len(shape)
        for each in range(len(namelist)):
            files = open(field + '_' + str(namelist[each]).zfill(4) + '.dat', 'wb')
            if(dims == 2):
                for j in range(shape[1]):
                    for i in range(shape[0]):
                        files.write(struct.pack('d', narray[each][i, j]))
            elif(dims == 3):
                for k in range (shape[2]):
                    for j in range(shape[1]):
                        for i in range(shape[0]):
                            files.write(struct.pack('d', narray[each][i, j, k]))
            else:
                pass
            files.close()
    def csv_write(self, data=[], filename='pro_dose.csv'):
        '''
        This function is used to write data to csv file.
        Parameters:
            data          - data array.
            filenumber    - sdf file number, an integer or an integer list.
        Returns:
            None.
        Raises:
            KeyError.
        '''
        import numpy as np
        import csv

        shape = data.shape
        length = len(shape)

        f = open(filename, 'wb')
        csv_writer = csv.writer(f
                )
        if (length == 1):
            psss
        elif (length == 2):
            for i in range(shape[0]):
                csv_writer.writerow(data[i,:])
        elif (length == 3):
            pass
        else:
            pass

        f.close()

    def reconstruct_field(self, info='epoch', filenumber=0, filename='by.dat', fileshape=[400,400], field='bx', prefix='1', length=200, field_range=[5, 65], filewrite='by', \
                          temp_factor=1., nreduce=1):
        '''
        This function is used to reconstruct field which used in proton radiography.
        Parameters:
            info          - information.
            filenumber    - sdf file number, an integer or an integer list.
            filename      - binary file name.
            fileshape     - data shape.
            field         - field information.
            prefix        - sdf file prefix.
            length        - length in third direction.
            field_range   - field range in third direction.
            filename      - binary file name.
            temp_factor   - temperature factor.
            nreduce       - reduce factor
        Returns:
            None.
        Raises:
            KeyError.
        '''
        import struct
        import numpy as np
        import epoch_class
        ec = epoch_class.epoch_class()

        if (info == 'epoch'):
            data_dict = ec.get_data(filenumber, prefix=prefix)
            keys = data_dict.keys()
            field = ec.get_field(field=field, keys=keys)
            data = data_dict[field].data
            #data = np.transpose(data)
            files = open(filewrite + '_' + str(filenumber).zfill(4) + '.dat', 'wb')
        elif (info == 'fortran'):
            data = binary_io.binary_read(self, filename=filename, shape=fileshape)
            files = open(filewrite + '.dat', 'wb')
        shape = data.shape
        dims = len(shape)
        nx = length
        if (dims == 2):
            ny = int(shape[0] / nreduce)
            nz = int(shape[1] / nreduce)
        elif (dims == 3):
            ny = int(shape[1] / nreduce)
            nz = int(shape[2] / nreduce)
        else:
            pass
        low = field_range[0]-1
        up  = field_range[1]-1
        print(nx, ny, nz)

        temp = np.sqrt(temp_factor)

        # write to file
        if (dims == 2):
            for k in range(nz):
                for j in range(ny):
                    output = np.sum(data[nreduce*j:nreduce*(j+1), nreduce*k:nreduce*(k+1)])/(nreduce**2)
                    for i in range(nx):
                        if ((i >= low) and (i < up)):
                            files.write(struct.pack('d', -output/temp))
                        else:
                            files.write(struct.pack('d', 0.))
        elif (dims == 3):
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        if ((i >= low) and (i < up)):
                            files.write(struct.pack('d', data[i-low, j, k]/temp))
                        else:
                            files.write(struct.pack('d', 0.))
        else:
            pass
        files.close()
    # read binary file
    def binary_read(self, filename=r'density.dat', shape=[300, 200, 200], axis='x', index=100):
        '''
        This function is used to read binary file.
        Parameters:
            filename      - binary file name.
            shape         - array shape.
            axis          - if 3d, axis is slice.
            index         - if 3d, slice at index
        Returns:
            array
        Raises:
            KeyError.
        '''
        import numpy as np
        import struct
        length = len(shape)
        files = open(filename, 'rb')
        if (length == 1):
            files.seek(0, 0)
            array = np.zeros(shape, np.float)
            for i in range(shape[0]):
                element = struct.unpack('d', files.read(8))
                array[i] = element[0]
        elif(length == 2):
            files.seek(0, 0)
            array = np.zeros(shape, np.float)
            for j in range(shape[1]):
                for i in range(shape[0]):
                    element = struct.unpack('d', files.read(8))
                    array[i, j] = element[0]
        elif(length == 3):
            if (axis == 'x'):
                array = np.zeros([shape[1], shape[2]], np.float)
                files.seek((index-1)*8, 0)
                for k in range(shape[2]):
                    for j in range(shape[1]):
                        element = struct.unpack('d', files.read(8))
                        array[j, k] = element[0]
                        files.seek((shape[0]-1)*8, 1)
            elif (axis == 'y'):
                array = np.zeros([shape[0], shape[2]], np.float)
                files.seek(shape[0]*(index-1)*8, 0)
                for k in range(shape[2]):
                    for i in range(shape[0]):
                        element = struct.unpack('d', files.read(8))
                        array[i, k] = element[0]
                    files.seek(shape[0]*(shape[1]-1)*8, 1)
            else:
                array = np.zeros([shape[0], shape[1]], np.float)
                files.seek(shape[0]*shape[1]*(index-1)*8, 0)
                for j in range(shape[1]):
                    for i in range(shape[0]):
                        element = struct.unpack('d', files.read(8))
                        array[i, j] = element[0]
            #for k in range(shape[2]):
            #    for j in range(shape[1]):
            #        for i in range(shape[0]):
            #            element = struct.unpack('d', files.read(8))
            #            array[i, j, k] = element[0]
        else:
            pass
        files.close()
        return array
    def get_extent(self, axis='z', shape=[300, 200, 200], constant=1.0):
        '''
        This function is used to get extent.
        Parameters:
            axis          - slice axis.
            shape         - array shape.
            constant      - dimensionless constants.
        Returns:
            None.
        Raise:
            KeyError.
        '''
        dimen = len(shape)
        x = binary_io.binary_read(self, filename='x.dat', shape=[shape[0]])
        y = binary_io.binary_read(self, filename='y.dat', shape=[shape[1]])
        if (dimen == 3):
            z = binary_io.binary_read(self, filename='z.dat', shape=[shape[2]])
        extent = []
        if(axis == 'x'):
            extent.append(y[0]/constant)
            extent.append(y[shape[1]-1]/constant)
            extent.append(z[0]/constant)
            extent.append(z[shape[2]-1]/constant)
        elif (axis == 'y'):
            extent.append(x[0]/constant)
            extent.append(x[shape[0]-1]/constant)
            extent.append(z[0]/constant)
            extent.append(z[shape[2]-1]/constant)
        else:
            extent.append(x[0]/constant)
            extent.append(x[shape[0]-1]/constant)
            extent.append(y[0]/constant)
            extent.append(y[shape[1]-1]/constant)
        return extent
    def polar2line(self, filename='energy', axis='x', index=30, shape=[50, 2200, 2200]):
        '''
        This function is used to average a polar plane to 1d line.
        Parameters:
            filename      - binary file name.
            axis          - slice axis.
            index         - slice index.
            shape         - array shape.
        Returns:
            None.
        Raise:
            KeyError.
        '''
        import numpy as np
        data = binary_io.binary_read(self, filename=filename, shape=shape, axis=axis, index=index)
        n = int(shape[2]/2)
        center = [n, n]
        array = np.zeros(n, np.float)
        number = np.zeros(n, np.int)
        for i in range(n):
            for j in range(n):
                if (data[i, j] != 0.0):
                    length = int(round(np.sqrt((i-n)**2 + (j-n)**2)))
                    if (length < n):
                        array[length] = array[length] + data[i,j]
                        number[length] = number[length] + 1
        for i in range(n):
            if(number[i] != 0):
                array[i] = array[i]/np.float(number[i])
        return array
    def txt_read(self, filename='al', index=10, start=6):
        '''
        This function used to read txt file of stopping power.
        Parameters:
            filename      - txt file name.
            inde          - middle character(\t).
            start         - start line
        Returns:
            data.
        Raise:
            KeyError.
        '''
        import numpy as np
        energy = []
        stopping_power = []
        
        f = open(filename + '.txt')
        flines = f.readlines()
        length = len(flines)

        for i in range(start, length-1):
            energy.append(float(flines[i][:index]))
            stopping_power.append(float(flines[i][index:]))

        return (energy, stopping_power)

    def flash_field_write(self, prefix='lasslab', filenumber=[0], keyword='cnt', field='dens', refine=0, box_panning=[0.1, 0.1, 0.1], \
                   box_scale=0.8, geom=[0., 1., 0., 1., 0., 1.], geom_factor=1, dimension=2, axis='z', coordinate=0., ngrid=[8,8,8], box_grid=[256,256], \
                   targ='none', species = 'none', Abar=1., Zbar=1., mass_ratio=25, temp_factor=1.0, dens_factor=1.0, grid_small=10,\
                   ifexit=True, ifAMR=True, ifreturn=True, iflog=False, method=1,selec_box=False, \
                   grid_x_min=0, grid_x_max=1, grid_y_min=0,grid_y_max=1):
        '''
        This function is used to visualize 2d FLASH output data.
        Parameters:
            prefix         - FLASH output file prefix.
            filenumber     - FLASH output file number.
            keyword        - FLASH output file keyword.
            field          - physical field.
            refine         - refine level.
            box_scale      - total box scale.
            box_panning    - box panning
            geom           - 2d geometry axis.
            geom_factor    - 2d geometry axis factor.
            dimension      - geom demensins.
            axis           - slice axis.
            coordinate     - slice coordinate.
            ngrid          - [nxb,nyb,nzb] in flash.
            ifexit         - if data exit in file.
            ifAMR          - if AMR or UG.
            box_grid       - box grid.
            targ           - target material name.
            species        - particle species.
            Abar           - averaged A.
            Zbar           - averaged Z.
            mass_ratio     - mass ratio of ion and electron.
            iflog          - if use log correction in dens.
            temp_factor    - temperature factor.
            dens_factor    - number density factor.
            method         - interpolate method.
        Returns:
            None.
        Raises:
            KeyError.
        '''

        import numpy as np
        import flash_class
        import struct
        from scipy import interpolate

        fc = flash_class.flash_class()

        dims = len(box_grid)

        data = fc.get_data(prefix=prefix, filenumber=filenumber[0], keyword=keyword)
        block = fc.get_block(data, refine=refine, geom=geom, box_scale=box_scale, box_panning=box_panning)

        if (ifexit == True):
            if (field == 'dens+ye'):
                field_data1 = data['dens'][:]
                field_data2 = data['ye'][:]
                if (targ == 'none'):
                    field_data = field_data1 * field_data2
                else:
                    field_data3 = data[targ][:]
                    field_data = field_data1 * field_data2 * field_data3
            else:
                if (targ == 'none'):
                    field_data = data[field][:]
                else:
                    field_data1 = data[field][:]
                    field_data2 = data[targ][:]
                    field_data = field_data1 * field_data2
        else:
            pass
        if (dims == 2):
            figinfo = fc.get_figinfo(box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, axis=axis, coordinate=coordinate, ngrid=ngrid)

            narray = fc.get_plane(field_data=field_data, block=block, dimension=dimension, axis=axis, coordinate=figinfo[4], ngrid=ngrid)
        elif (dims == 3):
            pass

        if (ifAMR == True):
            if (dims == 1):
                pass
            elif (dims == 2):
                nx = box_grid[0]
                ny = box_grid[1]
                array = np.zeros([nx, ny])

                x_min = figinfo[0][0]
                x_max = figinfo[0][0] + figinfo[0][2]
                dx = (x_max - x_min)/float(nx)
                y_min = figinfo[0][1]
                y_max = figinfo[0][1] + figinfo[0][3]
                dy = (y_max - y_min)/float(ny)
                length = len(narray)
                #print(x_min, x_max, y_min, y_max)
                for i in range(length):
                    array_loc = np.transpose(narray[i][1])
                    x_min_loc = narray[i][0][0]
                    x_max_loc = narray[i][0][0] + narray[i][0][2]
                    dx_loc = (x_max_loc - x_min_loc)/float(figinfo[5][0])
                    y_min_loc = narray[i][0][1]
                    y_max_loc = narray[i][0][1] + narray[i][0][3]
                    #print(x_min_loc, x_max_loc, y_min_loc, y_max_loc)
                    dy_loc = (y_max_loc - y_min_loc)/float(figinfo[5][1])
                    nx_start = int(round((x_min_loc - x_min)/dx))
                    nx_end = int(round((x_max_loc - x_min)/dx))
                    ny_start = int(round((y_min_loc - y_min)/dy))
                    ny_end = int(round((y_max_loc - y_min)/dy))
                    # interpolation
                    # method 1
                    if (method == 1):
                        for iy in range(ny_start, ny_end):
                            for ix in range(nx_start, nx_end):
                                x_loc = x_min + dx*(ix - 0.5)
                                y_loc = y_min + dy*(iy - 0.5)
                                ix_loc = int((x_loc - x_min_loc)/dx_loc)
                                iy_loc = int((y_loc - y_min_loc)/dy_loc)
                                #array[ix,iy] = (array_loc[ix_loc,iy_loc] + array_loc[ix_loc-1,iy_loc] + \
                                #                array_loc[ix_loc,iy_loc] + array_loc[ix_loc,iy_loc-1])*0.25
                                array[ix,iy] = array_loc[ix_loc,iy_loc]
                    # method 2
                    else:
                        xaxis = np.linspace(x_min_loc, x_max_loc, ngrid[0]+1, endpoint=True)
                        xaxis = 0.5 * (xaxis[1:] + xaxis[:-1])
                        yaxis = np.linspace(y_min_loc, y_max_loc, ngrid[1]+1, endpoint=True)
                        yaxis = 0.5 * (yaxis[1:] + yaxis[:-1])
                        f = interpolate.interp2d(xaxis, yaxis, array_loc, kind='linear')

                        xaxis = np.linspace(x_min_loc, x_max_loc, nx_end-nx_start+1, endpoint=True)
                        xaxis = 0.5 * (xaxis[1:] + xaxis[:-1])
                        yaxis = np.linspace(y_min_loc, y_max_loc, ny_end-ny_start+1, endpoint=True)
                        yaxis = 0.5 * (yaxis[1:] + yaxis[:-1])
                        array[nx_start:nx_end, ny_start:ny_end] = f(xaxis, yaxis)
            elif (dims == 3):
                nx = box_grid[0]
                ny = box_grid[1]
                nz = box_grid[2]
                array = np.zeros([nx, ny, nz])

                x_min = box_panning[0]
                x_max = x_min + box_scale
                dx = (x_max - x_min)/float(nx)
                y_min = box_panning[1]
                y_max = y_min + box_scale
                dy = (y_max - y_min)/float(ny)
                z_min = box_panning[2]
                z_max = z_min + box_scale
                dz = (z_max - z_min)/float(nz)

                length = len(block)
                for i in range(length):
                    array_loc = field_data[block[i][0]]

                    x_min_loc = block[i][2]
                    x_max_loc = x_min_loc + block[i][5]
                    dx_loc = (x_max_loc - x_min_loc)/float(ngrid[0])
                    y_min_loc = block[i][3]
                    y_max_loc = y_min_loc + block[i][6]
                    dy_loc = (y_max_loc - y_min_loc)/float(ngrid[1])
                    z_min_loc = block[i][4]
                    z_max_loc = z_min_loc + block[i][7]
                    dz_loc = (z_max_loc - z_min_loc)/float(ngrid[2])

                    nx_start = int(round((x_min_loc - x_min)/dx))
                    nx_end = int(round((x_max_loc - x_min)/dx))
                    ny_start = int(round((y_min_loc - y_min)/dy))
                    ny_end = int(round((y_max_loc - y_min)/dy))
                    nz_start = int(round((z_min_loc - z_min)/dz))
                    nz_end = int(round((z_max_loc - z_min)/dz))

                    # interpolation
                    # method 1
                    if (method == 1):
                        for iz in range(nz_start, nz_end):
                            z_loc = z_min + dz*(iz - 0.5)
                            iz_loc = int((z_loc - z_min_loc)/dz_loc)
                            for iy in range(ny_start, ny_end):
                                y_loc = y_min + dy*(iy - 0.5)
                                iy_loc = int((y_loc - y_min_loc)/dy_loc)
                                for ix in range(nx_start, nx_end):
                                    x_loc = x_min + dx*(ix - 0.5)
                                    ix_loc = int((x_loc - x_min_loc)/dx_loc)

                                    array[ix, iy, iz] = array_loc[iz_loc, iy_loc, ix_loc]
                    # method 2
                    else:
                        pass
                        xaxis = np.linspace(x_min_loc, x_max_loc, ngrid[0], endpoint=True)
                        #xaxis = 0.5 * (xaxis[1:] + xaxis[:-1])
                        yaxis = np.linspace(y_min_loc, y_max_loc, ngrid[1], endpoint=True)
                        #yaxis = 0.5 * (yaxis[1:] + yaxis[:-1])
                        zaxis = np.linspace(z_min_loc, z_max_loc, ngrid[2], endpoint=True)
                        #zaxis = 0.5 * (zaxis[1:] + zaxis[:-1])

                        xaxis_new = np.linspace(x_min_loc, x_max_loc, nx_end-nx_start+1, endpoint=True)
                        xaxis_new = 0.5 * (xaxis_new[1:] + xaxis_new[:-1])
                        yaxis_new = np.linspace(y_min_loc, y_max_loc, ny_end-ny_start+1, endpoint=True)
                        yaxis_new = 0.5 * (yaxis_new[1:] + yaxis_new[:-1])
                        zaxis_new = np.linspace(z_min_loc, z_max_loc, nz_end-nz_start+1, endpoint=True)
                        zaxis_new = 0.5 * (zaxis_new[1:] + zaxis_new[:-1])
                        #print(xaxis.shape, yaxis.shape, zaxis.shape)
                        #print(xaxis_new.shape, yaxis_new.shape, zaxis_new.shape)
                        #print(array_loc.shape)
                        #print(x_min_loc, x_max_loc, y_min_loc, y_max_loc, z_min_loc, z_max_loc)

                        for iz in range(nz_start, nz_end):
                            for iy in range(ny_start, ny_end):
                                for ix in range(nx_start, nx_end):
                                    array[ix, iy, iz] = interpolate.interpn((zaxis, yaxis, xaxis), array_loc, \
                                                        xi=[zaxis_new[iz-nz_start], yaxis_new[iy-ny_start], xaxis_new[ix-nx_start]], method='linear')[0]
                                    #array[ix, iy, iz] = interpolate.interpn((zaxis, yaxis, xaxis), array_loc, xi=[0.12, 0.12, 0.12], method='linear')[0]

            else:
                pass
        else:
            pass
        #modify the  array in the need box
        if selec_box :
            array = array[grid_x_min:grid_x_max, grid_y_min:grid_y_max]
            array = array[::grid_small, ::grid_small]
        else:
            pass

        print("矩阵大小为：", array.shape)

        nx,ny = array.shape

        # normalization
        if (field == 'ye'):
            #array =  np.sum(array)/float(nx * ny)
            pass
        elif (field == 'dens' and species == 'ion'):
            array = array * 1e3 / (Abar * 1.67e-27) * dens_factor
        elif (field == 'dens' and species == 'ele'):
            array = array * 1e3 / (Abar * 1.67e-27) * Zbar * dens_factor
        elif (field == 'dens+ye'):
            array = array * 1e3 / (Abar * 1.67e-27) * Abar * dens_factor
        elif (field == 'velx' and species == 'ion'):
            array = array * 1e-2 * (mass_ratio * 9.11e-31) * np.sqrt(temp_factor/(mass_ratio / 1836. / Abar))
        elif (field == 'velx' and species == 'ele'):
            array = array * 1e-2 * (9.11e-31) * np.sqrt(temp_factor/(mass_ratio / 1836. / Abar))
        elif (field == 'vely' and species == 'ion'):
            array = array * 1e-2 * (mass_ratio * 9.11e-31) * np.sqrt(temp_factor/(mass_ratio / 1836. / Abar))
        elif (field == 'vely' and species == 'ele'):
            array = array * 1e-2 * (9.11e-31) * np.sqrt(temp_factor/(mass_ratio / 1836. / Abar))
        elif (field == 'velz' and species == 'ion'):
            array = array * 1e-2 * (mass_ratio * 9.11e-31) * np.sqrt(temp_factor/(mass_ratio / 1836. / Abar))
        elif (field == 'velz' and species == 'ele'):
            array = array * 1e-2 * (9.11e-31) * np.sqrt(temp_factor/(mass_ratio / 1836. /Abar ))
        elif (field == 'magx'):
            array = array * 1e-4 * np.sqrt(temp_factor * dens_factor)
        elif (field == 'magy'):
            array = array * 1e-4 * np.sqrt(temp_factor * dens_factor)
        elif (field == 'magz'):
            array = array * 1e-4 * np.sqrt(temp_factor * dens_factor)
        elif (field == 'tele'):
            array = array * temp_factor
        elif (field == 'tion'):
            array = array * temp_factor

        # use log
        #if (iflog == True):
        #    max_value = np.max(array)
        #    min_value = np.min(array)
        #    factor = np.log10(max_value / min_value)
        #    for j in range(ny):
        #        for i in range(nx):
        #            array[i,j] = max_value / (np.log10(max_value / array[i,j]) + 1)

        # return or write to file.
        if (ifreturn == True):
            return array
        else:
            if (species == 'none'):
                files = open('flash_' + field + '_' + str(filenumber[0]).zfill(4) + '.dat', 'wb')
            else:
                if (targ == 'none'):
                    files = open('flash_' + field + '_' + species + '_' + str(filenumber[0]).zfill(4) + '.dat', 'wb')
                else:
                    files = open('flash_' + field + '_' + species + '_' + targ + '_' + str(filenumber[0]).zfill(4) + '.dat', 'wb')
            if (dims == 1):
                pass
            elif (dims == 2):
                for j in range(ny):
                    for i in range(nx):
                        files.write(struct.pack('d', array[i,j]))
            elif (dims == 3):
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            files.write(struct.pack('d', array[i,j,k]))

    def flash2epoch(self, prefix='MR', filenumber=[5], keyword='cnt', box_scale=0.7, box_panning=[0.15,0.15,0.15], geom=[-0.15,0.15,-0.25,0.25,-0.02,0.13], geom_factor=10, \
                    dimension=3, ngrid=[8,8,8], axis='z', coordinate=0.02, box_grid=[600,1000], Abars=[64.0, 6.5, 4.0], mi=[250, 25, 20], targs=['tar1', 'tar2', 'cham'], temp_factor=10., dens_factor=1., \
                    write_density=True, write_temperature=True, write_velocity=True, write_field=True, \
                    selec_box=False, \
                    grid_x_min=0, grid_x_max=1, grid_y_min=0,grid_y_max=1):
        '''
        This function is used to convert data format, from flash to epoch.
        Parameters:
            prefix         - FLASH output file prefix.
            filenumber     - FLASH output file number.
            keyword        - FLASH output file keyword.
            box_scale      - total box scale.
            box_panning    - box panning
            geom           - 2d geometry axis.
            geom_factor    - 2d geometry axis factor.
            dimension      - geom demensins.
            axis           - slice axis.
            coordinate     - slice coordinate.
            ngrid          - [nxb,nyb,nzb] in flash.
            box_grid       - box grid.
            Abars          - atomic number.
            mi             - mi in PIC.
            targs          - target name in FLASH.
            temp_factor    - temperature factor.
            dens_factor    - number density factor.
        Returns:
            None.
        Raises:
            KeyError.
        '''
        import numpy as np

        length = len(targs)
        ye = binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='ye', species='ele', \
                                dens_factor=dens_factor, ifreturn=True,selec_box=selec_box, \
                                grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                grid_y_min=grid_y_min, grid_y_max=grid_y_max)
        sumy = binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                        dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='sumy', species='ion', \
                                        dens_factor=dens_factor, ifreturn=True,selec_box=selec_box, \
                                        grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                        grid_y_min=grid_y_min, grid_y_max=grid_y_max)
        Zbar_all = ye/sumy
        Zbars = []
        for ispecies in range(length):
            # find Zbar

            targ_species = binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                        dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field=targs[ispecies], species='ele',  \
                                        dens_factor=dens_factor, selec_box=selec_box, \
                                        grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                        grid_y_min=grid_y_min, grid_y_max=grid_y_max)

            Zbar = round(Zbar_all[np.where(targ_species > 0.95)].mean())
            print("Zbar is ",Zbar)
            if (Zbar < 1.0):
                Zbar = 1.0
            Zbars.append(Zbar)

            if (write_density):
                # write electron density
                binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                        dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='dens', species='ele', targ=targs[ispecies], \
                                        Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                                            grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                            grid_y_min=grid_y_min, grid_y_max=grid_y_max)
                # write ion density
                binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                        dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='dens', species='ion', targ=targs[ispecies], \
                                        Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                                            grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                            grid_y_min=grid_y_min, grid_y_max=grid_y_max)
            if (write_temperature):
                # write electron temperature
                binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                        dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='tele', species='ele', targ=targs[ispecies], \
                                        Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], temp_factor=temp_factor, dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                                            grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                            grid_y_min=grid_y_min, grid_y_max=grid_y_max)
                # write ion temperature
                binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                        dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='tion', species='ion', targ=targs[ispecies], \
                                        Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], temp_factor=temp_factor, dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                                            grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                            grid_y_min=grid_y_min, grid_y_max=grid_y_max)
            if (write_velocity):
                # write electron velocity : vx
                binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                        dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='velx', species='ele', targ=targs[ispecies], \
                                        Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], temp_factor=temp_factor, dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                                            grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                            grid_y_min=grid_y_min, grid_y_max=grid_y_max)
                # write electron velocity : vy
                binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                        dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='vely', species='ele', targ=targs[ispecies], \
                                        Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], temp_factor=temp_factor, dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                                            grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                            grid_y_min=grid_y_min, grid_y_max=grid_y_max)
                # write electron velocity : vz
                # binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                #                         dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='velz', species='ele', targ=targs[ispecies], \
                #                         Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], temp_factor=temp_factor, dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                #                             grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                #                             grid_y_min=grid_y_min, grid_y_max=grid_y_max)
                # write ion velocity : vx
                binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                        dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='velx', species='ion', targ=targs[ispecies], \
                                        Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], temp_factor=temp_factor, dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                                            grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                            grid_y_min=grid_y_min, grid_y_max=grid_y_max)
                # write ion velocity : vy
                binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                        dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='vely', species='ion', targ=targs[ispecies], \
                                        Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], temp_factor=temp_factor, dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                                            grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                            grid_y_min=grid_y_min, grid_y_max=grid_y_max)
                # # write ion velocity : vz
                # binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                #                         dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='velz', species='ion', targ=targs[ispecies], \
                #                         Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], temp_factor=temp_factor, dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                #                             grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                #                             grid_y_min=grid_y_min, grid_y_max=grid_y_max)
        if (write_field):
            # write magnetic field
            # write Bx
            binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                    dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='magx', species='none', targ='none', \
                                    Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], temp_factor=temp_factor, dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                                            grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                            grid_y_min=grid_y_min, grid_y_max=grid_y_max)
            # write By
            binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                    dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='magy', species='none', targ='none', \
                                    Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], temp_factor=temp_factor, dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                                            grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                            grid_y_min=grid_y_min, grid_y_max=grid_y_max)
            # write Bz
            binary_io.flash_field_write(self, prefix=prefix, filenumber=filenumber, keyword=keyword, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, \
                                    dimension=dimension, axis=axis, coordinate=coordinate, ngrid=ngrid, box_grid=box_grid, field='magz', species='none', targ='none', \
                                    Abar=Abars[ispecies], Zbar=Zbar, mass_ratio=mi[ispecies], temp_factor=temp_factor, dens_factor=dens_factor, ifreturn=False,selec_box=selec_box, \
                                            grid_x_min=grid_x_min, grid_x_max=grid_x_max, \
                                            grid_y_min=grid_y_min, grid_y_max=grid_y_max)

        print(Zbars)
        return Zbars
        

    def SRIM_convert(self, filename='Al.txt', newfile='new.txt', line_bound=[1,100], material='ACTIVE LAYER'):
        '''
        This function is used to convert SRIM stopping power to friendly format.
        Parameters:
            filename       - SRIM data file.
            newfile        - new file to generate.
            line_bound     - lines boundary to read.
            material       - name of material.
        Returns:
            None.
        Raises:
            KeyError.
        '''
        import numpy as np

        f_read = open(filename, 'r')
        lines = f_read.readlines()
        f_read.close()

        low = line_bound[0] - 1
        up = line_bound[1] - 1

        array = np.zeros([up-low+1, 2])

        # convert data
        for i in range(low, up+1):
            s = lines[i]
            energy = s[:7]
            unit = s[8:11]
            sp1 = s[14:23]
            sp2 = s[25:34]

            if (unit == 'keV'):
                array[i-low,0] = float(energy) * 1e-3
            elif (unit == 'MeV'):
                array[i-low,0] = float(energy) * 1.0
            elif (unit == 'GeV'):
                array[i-low,0] = float(energy) * 1e3
            else:
                array[i-low,0] = float(energy) * 1e-6

            array[i-low,1] = (float(sp1) + float(sp2)) * 1e3
        # write file
        f_write = open(newfile, 'w')
        f_write.write('PSTAR: Stopping Powers and Range Tables for Protons\n')
        f_write.write('\n')
        f_write.write(material + '                              \n')
        f_write.write('\n')
        f_write.write('Kinetic   Total     \n')
        f_write.write('Energy    Stp. Pow. \n')
        f_write.write('MeV       MeV cm2/g \n')
        f_write.write('\n')

        for j in range(up-low+1):
            f_write.write(format(array[j,0], '.3E'))
            f_write.write(' ')
            f_write.write(format(array[j,1], '.3E'))
            f_write.write(' \n')
        f_write.close()
        
       # return array

    #def PEOS2cn4(self, source='Cu.prp', dest='Cu-PEOS.cn4', Abar=64.0, Zbar=29.0, nT1=36, nD1=91, nT2=36, nD2=46, nR=10, \
    #             lT1=[40,43], lD1=[45,54], lT2=[61,64], lD2=[66,70], lZ=[9906,10233], lPi=[11723,12050], lPe=[12052,12379], lEi=[11065,11392], lEe=[11394,11721], \
    #             lR=[73,74], lO=[12380,22315], iflog=False, ifwrite=False, ifreconstruct=False, method=2, rpoint=6, initindex=0, radcoeff=1.0):
    def PEOS2cn4(self, source='Au.prp', dest='Au-PEOS.cn4', Abar=197.0, Zbar=79.0, nT1=40, nD1=55, nT2=20, nD2=20, nR=1, \
                 lT1=[40,43], lD1=[45,50], lT2=[57,58], lD2=[60,61], lZ=[17668,17887], lPi=[18675,18894], lPe=[18896,19115], lEi=[18233,18452], lEe=[18454,18673], \
                 lR=[64,64], lO=[19116,21515], iflog=False, ifwrite=False, ifreconstruct=False, method=2, rpoint=6, initindex=0, radcoeff=1.0):
        '''
        This function is used to convert PROPACEOS to IONMIX4 EOS.
        Parameters:
            source         - source file in PROPACEOS format..
            dest           - dest file in IONMIX4 EOS.
            Abar           - Abar.
            Zbar           - Zbar.
            nT1            - point of temperature, EOS.
            nD1            - point of density, EOS.
            nT2            - point of temperature, OPACITY.
            nD2            - point of density, OPACITY.
            nR             - point of radiation.
            lT1            - lines of radiation, EOS.
            lD1            - lines of density, EOS.
            lT2            - lines of radiation, OPACITY.
            lD2            - lines of density, OPACITY.
            lZ             - lines of zbar, EOS.
            lPi            - lines of Pion, EOS.
            lPe            - lines of Pele, EOS.
            lEi            - lines of Eion, EOS.
            lEe            - lines of Eele, EOS.
            lR             - lines of radiation, OPACITY.
            lO             - lines of opacity, OPACITY.
            iflog          - if use log in interpolate.
            ifwrite        - if write to file.
            ifreconstruct  - if reconstruct radiation temperature.
            method         - read opacity method.
            rpoint         - radiation temperature point.
            initindex      - init index of temperature.
            radcoeff       - radiation opacity coefficient.
        Returns:
            None.
        Raises:
            KeyError.
        '''
        import numpy as np
        from scipy import interpolate

        f_read = open(source, 'r')
        lines = f_read.readlines()
        f_read.close()

        # read temperature point, EOS
        temp1 = []
        for i in range(lT1[0]-1, lT1[1]):
            temp1 = temp1 + lines[i].strip().split('  ')
        for i in range(len(temp1)):
            temp1[i] = float(temp1[i])
        if (len(temp1) != nT1):
            print("Warning: Not match teperature point, EOS!")
        else:
            temp1 = np.array(temp1)

        # read density point, EOS
        dens1 = []
        for i in range(lD1[0]-1, lD1[1]):
            dens1 = dens1 + lines[i].strip().split('  ')
        for i in range(len(dens1)):
            dens1[i] = float(dens1[i])
        if (len(dens1) != nD1):
            print("Warning: Not match density point, EOS!")
        else:
            dens1 = np.array(dens1)

        # read temperature point, OPACITY
        temp2 = []
        for i in range(lT2[0]-1, lT2[1]):
            temp2 = temp2 + lines[i].strip().split('  ')
        for i in range(len(temp2)):
            temp2[i] = float(temp2[i])
        if (len(temp2) != nT2):
            print("Warning: Not match teperature point, OPACITY!")
        else:
            temp2 = np.array(temp2)

        # read density point, OPACITY
        dens2 = []
        for i in range(lD2[0]-1, lD2[1]):
            dens2 = dens2 + lines[i].strip().split('  ')
        for i in range(len(dens2)):
            dens2[i] = float(dens2[i])
        if (len(dens2) != nD2):
            print("Warning: Not match density point, OPACITY!")
        else:
            dens2 = np.array(dens2)

        # read zbar point, EOS
        zbar = []
        for i in range(lZ[0]-1, lZ[1]):
            zbar = zbar + lines[i].strip().split('  ')
        for i in range(len(zbar)):
            zbar[i] = float(zbar[i])
        if (len(zbar) != nT1*nD1):
            print("Warning: Not match Zbar point, EOS!")
        else:
            zbar = np.array(zbar)
            zbar = np.reshape(zbar, (nT1,nD1), 'F')

        # read dZ/dT point, EOS, ignored by FLASH, set zero.
        dZdT = np.zeros((nT1, nD1), np.float)

        # read Pion point, EOS
        pion = []
        for i in range(lPi[0]-1, lPi[1]):
            pion = pion + lines[i].strip().split('  ')
        pion = [x.strip() for x in pion if (x.strip() != '')]
        for i in range(len(pion)):
            #pion[i] = np.abs(float(pion[i]))
            pion[i] = float(pion[i])
        if (len(pion) != nT1*nD1):
            print("Warning: Not match Pion point, EOS!")
        else:
            pion = 1e-7 * np.array(pion)
            pion = np.reshape(pion, (nT1,nD1), 'F')

        # read Pele point, EOS
        pele = []
        for i in range(lPe[0]-1, lPe[1]):
            pele = pele + lines[i].strip().split(' ')
        pele = [x.strip() for x in pele if (x.strip() != '')]
        for i in range(len(pele)):
            #pele[i] = np.abs(float(pele[i]))
            pele[i] = float(pele[i])
        if (len(pele) != nT1*nD1):
            print("Warning: Not match Pele point, EOS!")
        else:
            pele = 1e-7 * np.array(pele)
            pele = np.reshape(pele, (nT1,nD1), 'F')

        # read dPi/dT point, EOS, ignored by FLASH, set zero.
        dPidT = np.zeros([nT1, nD1], np.float)

        # read dPe/dT point, EOS, ignored by FLASH, set zero.
        dPedT = np.zeros([nT1, nD1], np.float)

        # read Eion point, EOS
        eion = []
        for i in range(lEi[0]-1, lEi[1]):
            eion = eion + lines[i].strip().split(' ')
        eion = [x.strip() for x in eion if (x.strip() != '')]
        for i in range(len(eion)):
            #eion[i] = np.abs(float(eion[i]))
            eion[i] = float(eion[i])
        if (len(eion) != nT1*nD1):
            print("Warning: Not match Eion point, EOS!")
        else:
            eion = np.array(eion)
            eion = np.reshape(eion, (nT1,nD1), 'F')
        # upshift
        #for i in range(nD1):
        #    low = np.min(eion[:,i])
        #    if (low <= 0):
        #        eion[:,i] = eion[:,i] + 1.1 * np.abs(low)

        # read Eele point, EOS
        eele = []
        for i in range(lEe[0]-1, lEe[1]):
            eele = eele + lines[i].strip().split(' ')
        eele = [x.strip() for x in eele if (x.strip() != '')]
        for i in range(len(eele)):
            #eele[i] = np.abs(float(eele[i]))
            eele[i] = float(eele[i])
        if (len(eele) != nT1*nD1):
            print("Warning: Not match Eele point, EOS!")
        else:
            eele = np.array(eele)
            eele = np.reshape(eele, (nT1,nD1), 'F')
        #for i in range(nD1):
        #    low = np.min(eele[:,i])
        #    if (low <= 0):
        #        eele[:,i] = eele[:,i] + 1.1 * np.abs(low)

        # read dEi/dT point, EOS, ignored by FLASH, set zero.
        dEidT = np.zeros([nT1, nD1], np.float)

        # read dEe/dT point, EOS, ignored by FLASH, set zero.
        dEedT = np.zeros([nT1, nD1], np.float)

        # read dEi/dN point, EOS, ignored by FLASH, set zero.
        dEidN = np.zeros([nT1, nD1], np.float)

        # read dEe/dN point, EOS, ignored by FLASH, set zero.
        dEedN = np.zeros([nT1, nD1], np.float)

        # read radiation point, OPACITY
        rad = []
        for i in range(lR[0]-1, lR[1]):
            rad = rad + lines[i].strip().split('  ')
        for i in range(len(rad)):
            rad[i] = float(rad[i])
        if (len(rad) != (nR+1)):
            print("Warning: Not match radiation point, OPACITY!")

        # re-construct radiation temperature
        if (ifreconstruct == True):
            rad_log = np.linspace(np.log10(rad[0]), np.log10(rad[nR]), nR+1)
            rad_mid = 0.5 * (rad_log[1:] + rad_log[:-1])
            rad_mid = 10**rad_mid
            rx = np.linspace(np.log10(rad[0]), np.log10(rad[nR]), rpoint+1)
            rx_mid = 0.5 * (rx[1:] + rx[:-1])
            rx = 10**rx
            rx_mid = 10**rx_mid


        # read opacityi point, OPACITY
        ##############################################
        if (method == 1):
            rosseland = []
            planckemi = []
            planckabs = []
            for i in range(lO[0], lO[1], 6):
                local1 = lines[i].strip().split('  ')
                local2 = lines[i+2].strip().split('  ')
                local3 = lines[i+4].strip().split('  ')
                for i in range(len(local1)):
                    local1[i] = float(local1[i])
                    local2[i] = float(local2[i])
                    local3[i] = float(local3[i])
                rosseland.append(local1)
                planckemi.append(local2)
                planckabs.append(local3)
            rosseland = np.reshape(np.array(rosseland), (nT2,nD2), 'F')
            planckemi = np.reshape(np.array(planckemi), (nT2,nD2), 'F')
            planckabs = np.reshape(np.array(planckabs), (nT2,nD2), 'F')

            # interpolation of OPACITY
            rosseland_new = np.zeros([nT1,nD1,nR], np.float)
            planckemi_new = np.zeros([nT1,nD1,nR], np.float)
            planckabs_new = np.zeros([nT1,nD1,nR], np.float)

            if (iflog == True):
                f1 = interpolate.interp2d(np.log10(dens2), np.log10(temp2), rosseland, kind='linear')
                rosseland_new[:,:,0] = f1(np.log10(dens1), np.log10(temp1))
                f2 = interpolate.interp2d(np.log10(dens2), np.log10(temp2), planckemi, kind='linear')
                planckemi_new[:,:,0] = f2(np.log10(dens1), np.log10(temp1))
                f3 = interpolate.interp2d(np.log10(dens2), np.log10(temp2), planckabs, kind='linear')
                planckabs_new[:,:,0] = f3(np.log10(dens1), np.log10(temp1))
            else:
                f1 = interpolate.interp2d(dens2, temp2, rosseland, kind='linear')
                rosseland_new[:,:,0] = f1(dens1, temp1)
                f2 = interpolate.interp2d(dens2, temp2, planckemi, kind='linear')
                planckemi_new[:,:,0] = f2(dens1, temp1)
                f3 = interpolate.interp2d(dens2, temp2, planckabs, kind='linear')
                planckabs_new[:,:,0] = f3(dens1, temp1)
        ##################################################
        elif (method == 2):
            rosseland = []
            planckemi = []
            planckabs = []

            for i in range(lO[0], lO[1], 6):
                local1 = lines[i].strip().split('  ')
                local2 = lines[i+2].strip().split('  ')
                local3 = lines[i+4].strip().split('  ')
                for i in range(len(local1)):
                    local1[i] = float(local1[i])
                    local2[i] = float(local2[i])
                    local3[i] = float(local3[i])
                
                # reconstruct
                if (ifreconstruct == True):
                    if (nR == 1):
                        local4 = np.repeat(local1[0], rpoint)
                        local5 = np.repeat(local2[0], rpoint)
                        local6 = np.repeat(local3[0], rpoint)
                    else:
                        f1 = interpolate.interp1d(np.log10(rad_mid), local1, kind='linear')
                        local4 = f1(np.log10(rx_mid))
                        f2 = interpolate.interp1d(np.log10(rad_mid), local2, kind='linear')
                        local5 = f2(np.log10(rx_mid))
                        f3 = interpolate.interp1d(np.log10(rad_mid), local3, kind='linear')
                        local6 = f3(np.log10(rx_mid))

                    rosseland.append(local4)
                    planckemi.append(local5)
                    planckabs.append(local6)
                else:
                    rosseland.append(local1)
                    planckemi.append(local2)
                    planckabs.append(local3)

            
            if (ifreconstruct == True):
                nR = rpoint

            rosseland = np.reshape(np.array(rosseland), (nT2,nD2,nR), 'F')
            planckemi = np.reshape(np.array(planckemi), (nT2,nD2,nR), 'F')
            planckabs = np.reshape(np.array(planckabs), (nT2,nD2,nR), 'F')

            # interpolation of OPACITY
            rosseland_new = np.zeros([nT1,nD1,nR], np.float)
            planckemi_new = np.zeros([nT1,nD1,nR], np.float)
            planckabs_new = np.zeros([nT1,nD1,nR], np.float)

            if (iflog == True):
                for i in range(nR):
                    f1 = interpolate.interp2d(np.log10(dens2), np.log10(temp2), rosseland[:,:,i], kind='linear')
                    rosseland_new[:,:,i] = f1(np.log10(dens1), np.log10(temp1))
                    f2 = interpolate.interp2d(np.log10(dens2), np.log10(temp2), planckemi[:,:,i], kind='linear')
                    planckemi_new[:,:,i] = f2(np.log10(dens1), np.log10(temp1))
                    f3 = interpolate.interp2d(np.log10(dens2), np.log10(temp2), planckabs[:,:,i], kind='linear')
                    planckabs_new[:,:,i] = f3(np.log10(dens1), np.log10(temp1))
            else:
                for i in range(nR):
                    f1 = interpolate.interp2d(dens2, temp2, rosseland[:,:,i], kind='linear')
                    rosseland_new[:,:,i] = f1(dens1, temp1)
                    f2 = interpolate.interp2d(dens2, temp2, planckemi[:,:,i], kind='linear')
                    planckemi_new[:,:,i] = f2(dens1, temp1)
                    f3 = interpolate.interp2d(dens2, temp2, planckabs[:,:,i], kind='linear')
                    planckabs_new[:,:,i] = f3(dens1, temp1)


        # write to file
        if (ifwrite == True):
            fcn4 = open(dest, 'w+')
            # 1, 2: write the numbver of temperature and density points.
            print("%10d%10d" %(nT1-initindex, nD1), file=fcn4)
            # 3: write the atomic number of each element in this material
            print(" atomic #s of gases: %10d" %(Zbar), file=fcn4)
            # 4: write the fraction (by number of ions) of each element in this material
            print(" relative fractions:   %.2E" %(1.0), file=fcn4)
            # 5: write the number of radiation energy groups
            print("%12d" %(nR), file=fcn4)
            # 6: write the temperature points (in ev)
            count = 0
            for i in range(initindex, nT1):
                print("%12.6E" %(temp1[i]), file=fcn4, end='')
                count = count + 1
                if (count%4 == 0):
                    print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 7: write the number density points (in cm^-3)
            count = 0
            for i in range(nD1):
                print("%12.6E" %(dens1[i]), file=fcn4, end='')
                count = count + 1
                if (count%4 == 0):
                    print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 8: write out the zbar at each temperature/density ponit (nele/nion)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(zbar[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 9: write d(zbar)/dT
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(dZdT[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 10: write the ion pressure
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(pion[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 11: write the electron pressure
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(pele[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 12: write the d(pion)/dT (in Joules/cm^3/eV)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(dPidT[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 13: write the d(pele)/dT (in Joules/cm^3/eV)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(dPedT[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 14: write out the ion specific internal energy (in Joules/gram)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(eion[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 15: write out the electron specific internal energy (in Joules/gram)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(eele[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 16: write out the ion specific heat (in Joules/gram/eV)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(dEidT[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 17: write out the electron specific heat (in Joules/gram/eV)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(dEedT[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 18: write out d(eion)/d(nion) (not sure)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(dEidN[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 19: write out d(eele)/d(nele) (not sure)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(dEedN[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 21: write out the energy group boundaries (in ev)
            count = 0
            for i in range(nR+1):
                if (ifreconstruct == False):
                    print("%12.6E" %(rad[i]), file=fcn4, end='')
                else:
                    print("%12.6E" %(rx[i]), file=fcn4, end='')
                count = count + 1
                if (count%4 == 0):
                    print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 22: write the Rosseland group opacities
            count = 0
            for k in range(nR):
                for j in range(nD1):
                    for i in range(initindex, nT1):
                        print("%12.6E" %(radcoeff * rosseland_new[i,j,k]), file=fcn4, end='')
                        count = count + 1
                        if (count%4 == 0):
                            print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 23: write the Planck absorption opacities
            count = 0
            for k in range(nR):
                for j in range(nD1):
                    for i in range(initindex, nT1):
                        print("%12.6E" %(radcoeff * planckabs_new[i,j,k]), file=fcn4, end='')
                        count = count + 1
                        if (count%4 == 0):
                            print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 24: write the Planck emission opacities
            count = 0
            for k in range(nR):
                for j in range(nD1):
                    for i in range(initindex, nT1):
                        print("%12.6E" %(radcoeff * planckemi_new[i,j,k]), file=fcn4, end='')
                        count = count + 1
                        if (count%4 == 0):
                            print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)


            # close
            fcn4.close()



        #return (temp1, dens1, temp2, dens2)
        return (temp1, dens1, zbar, pion, pele, eion, eele)
        #return (rosseland, planckemi, planckabs)
        #return (rosseland, planckemi, planckabs, rosseland_new, planckemi_new, planckabs_new)
        #return (rad, rx, nR)

    def FEOS2cn4(self, feos='Al.feos', opacity='Al-TOPS.txt', dest='Al-FEOS-TOPS.cn4', Abar=26.9815, Zbar=13.0, nT1=50, nD1=45, species=1, initindex=2, \
                 nT2=71, nD2=66, nR=10, lT2=[13,24], lD2=[26,36], multi_init=4868, useopacity=False, ifwrite=False):
    #def FEOS2cn4(self, feos='Cu.feos', opacity='Cu-TOPS.txt', dest='Cu-FEOS-TOPS.cn4', Abar=63.546, Zbar=29.0, nT1=50, nD1=45, species=1, initindex=3, \
    #             nT2=71, nD2=66, nR=10, lT2=[13,24], lD2=[26,36], multi_init=4868, useopacity=False, ifwrite=False):
    #def FEOS2cn4(self, feos='Ti.feos', opacity='Ti-TOPS.txt', dest='Ti-FEOS-TOPS.cn4', Abar=47.867, Zbar=22.0, nT1=50, nD1=45, species=1, initindex=2, \
    #             nT2=71, nD2=66, nR=10, lT2=[13,24], lD2=[26,36], multi_init=4868, useopacity=False, ifwrite=False):
        '''
        This function is used to convert FEOS EOS to cn4 format.
        Parameters:
            feos           - source file in FEOS format.
            opacity        - source file of TOPS opacity.
            dest           - dest file in IONMIX4 EOS.
            Abar           - Abar.
            Zbar           - Zbar.
            nT1            - point of temperature, EOS.
            nD1            - point of density, EOS.
            species        - n species.
            initindex      - init index of temperature.
            nT2            - point of temperature, OPACITY.
            nD2            - point of density, OPACITY.
            nR             - point of radiation.
            lT2            - lines of radiation, OPACITY.
            lD2            - lines of density, OPACITY.
            multi_init     - line of Multigroup opacities.
            useopacity     - if use opacity table.
            ifwrite        - if write to file.
        Returns:
            None.
        Raises:
            KeyError.
        '''
        import numpy as np
        from scipy import interpolate

        # handle EOS

        feos = np.loadtxt(feos)
        ny,nx = feos.shape
        feos = np.reshape(feos, [ny*nx])
        total = (nD1+1) + (nT1+1) + (species + 17) * ((nD1+1) * (nT1+1))
        feos = feos[:total]

        # density, convert g/cc to /cc.
        dens1 = feos[1:nD1+1]
        dens1 = dens1 / (Abar / 6.02214076e23)
        print(dens1)
        # temperature, in eV
        temp1 = feos[(nD1+2):(nD1+1+nT1+1)]
        print(temp1)

        # reshape
        block = int((ny*nx - (nD1+1) - (nT1+1)) / ((nD1+1) * (nT1+1)))
        feos = np.reshape(feos[(nD1+nT1+2):], [block, nT1+1, nD1+1])

        # total pressure, in dyne/cm2 or erg/cm3, and convert to J/cm3
        ptot = feos[0,1:,1:] / 1.e7
        # total energy, in erg/g, and convert to J/g
        etot = feos[1,1:,1:] / 1.e7
        # total entropy, in erg/eVg
        stot = feos[2,1:,1:]
        # total Helmholtz free energy, in erg/g
        ftot = feos[3,1:,1:]

        # electron pressure, in dyne/cm2 or erg/cm3, and convert to J/cm3
        pele = feos[4,1:,1:] / 1.e7
        # electron energy, in erg/g, and convert to J/g
        eele = feos[5,1:,1:] / 1.e7
        # electron entropy, in erg/eVg
        sele = feos[6,1:,1:]
        # electron Helmholtz free energy, in erg/g
        fele = feos[7,1:,1:]

        # ion pressure, in dyne/cm2 or erg/cm3, and convert to J/cm3
        pion = feos[8,1:,1:] / 1.e7
        # ion energy, in erg/g, and convert to J/g
        eion = feos[9,1:,1:] / 1.e7
        # ion entropy, in erg/eVg
        sion = feos[10,1:,1:]
        # ion Helmholtz free energy, in erg/g
        fion = feos[11,1:,1:]

        # Thomas-Fermi EOS
        # TF pressure, in dyne/cm2 or erg/cm3
        pTF = feos[12,1:,1:]
        # TF energy, in erg/g
        eTF = feos[13,1:,1:]
        # TF entropy, in erg/eVg
        sTF = feos[14,1:,1:]
        # TF Helmholtz free energy, in erg/g
        fTF = feos[15,1:,1:]

        # charge states
        zbar = feos[16,1:,1:]

        # charge states of every elements
        # leave behind

        if (useopacity == True):
            # handle OPACITY
            fopa = open(opacity)
            lines = fopa.readlines()
            fopa.close()

            # read temperature point, OPACITY
            temp2 = []
            for i in range(lT2[0]-1, lT2[1]):
                temp2 = temp2 + lines[i].strip().split('  ')
            for i in range(len(temp2)):
                temp2[i] = float(temp2[i])
            if (len(temp2) != nT2):
                print("Warning: Not match teperature point, OPACITY!")
            else:
                temp2 = np.array(temp2)
            # convert from keV to eV
            temp2[:] = temp2[:] * 1.e3

            # read density point, OPACITY
            dens2 = []
            for i in range(lD2[0]-1, lD2[1]):
                dens2 = dens2 + lines[i].strip().split('  ')
            for i in range(len(dens2)):
                dens2[i] = float(dens2[i])
            if (len(dens2) != nD2):
                print("Warning: Not match density point, OPACITY!")
            else:
                dens2 = np.array(dens2)
            # convert from g/cc to /cc
            dens2[:] = dens2[:] / (Abar / 6.02214076e23)

            # read opacity
            rosseland = np.zeros([nT2, nD2, nR], np.float)
            planck = np.zeros([nT2, nD2, nR], np.float)
            count = 0
            for j in range(nT2):
                for i in range(nD2):
                    index = multi_init + count * (nR + 1)
                    count = count + 1
                    local = []
                    for k in range(nR):
                        local = local + lines[index+1+k].strip().split('  ')
                    for k in range(len(local)):
                        local[k] = float(local[k])
                    local = np.reshape(local, [10,3])

                    rosseland[j,i,:] = local[:,1]
                    planck[j,i,:] = local[:,2]
            # interpolation of OPACITY
            rosseland_new = np.zeros([nT1,nD1,nR], np.float)
            planck_new = np.zeros([nT1,nD1,nR], np.float)

            for i in range(nR):
                f1 = interpolate.interp2d(np.log10(dens2), np.log10(temp2), rosseland[:,:,i], kind='linear')
                rosseland_new[:,:,i] = f1(np.log10(dens1), np.log10(temp1))
                f2 = interpolate.interp2d(np.log10(dens2), np.log10(temp2), planck[:,:,i], kind='linear')
                planck_new[:,:,i] = f2(np.log10(dens1), np.log10(temp1))


        # write cn4 file
        if (ifwrite == True):
            # write temperature in TOPS format
            ftemp = open('temperature.txt', 'w+')
            count = 0
            for i in range(initindex, nT1):
                print("%9.3E " %(temp1[i]/1.e3), file=ftemp, end='')
                count = count + 1
                if (count%4 == 0):
                    print("", file=ftemp)
            if (count%4 != 0):
                print("", file=ftemp
            )
            ftemp.close()
            # write density in TOPS format
            fdens = open('density.txt', 'w+')
            count = 0
            for i in range(nD1):
                print("%9.3E " %(dens1[i] * (Abar / 6.02214076e23)), file=fdens, end='')
                #print("%15.8E" %(dens1[i] * (Abar / 6.02214076e23)), file=fdens, end='')
                count = count + 1
                if (count%4 == 0):
                    print("", file=fdens)
            if (count%4 != 0):
                print("", file=fdens)
            fdens.close()

            # write radiation in TOPS format
            # in keV
            x = np.linspace(-4, 2, 11)
            rad = 10**x
            frad = open('radiation.txt', 'w+')
            count = 0
            for i in range(nR+1):
                print("%9.3E " %(rad[i]), file=frad, end='')
                count = count + 1
                if (count%4 == 0):
                    print("", file=frad)
            if (count%4 != 0):
                print("", file=frad)
            frad.close()

            # write IONMIX4 format
            fcn4 = open(dest, 'w+')
            # 1, 2: write the numbver of temperature and density points.
            print("%10d%10d" %(nT1-initindex, nD1), file=fcn4)
            # 3: write the atomic number of each element in this material
            print(" atomic #s of gases: %10d" %(Zbar), file=fcn4)
            # 4: write the fraction (by number of ions) of each element in this material
            print(" relative fractions:   %.2E" %(1.0), file=fcn4)
            # 5: write the number of radiation energy groups
            print("%12d" %(nR), file=fcn4)
            # 6: write the temperature points (in ev)
            count = 0
            for i in range(initindex, nT1):
                print("%12.6E" %(temp1[i]), file=fcn4, end='')
                count = count + 1
                if (count%4 == 0):
                    print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 7: write the number density points (in cm^-3)
            count = 0
            for i in range(nD1):
                print("%12.6E" %(dens1[i]), file=fcn4, end='')
                count = count + 1
                if (count%4 == 0):
                    print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 8: write out the zbar at each temperature/density ponit (nele/nion)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(zbar[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 9: write d(zbar)/dT
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 10: write the ion pressure
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(pion[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 11: write the electron pressure
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(pele[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 12: write the d(pion)/dT (in Joules/cm^3/eV)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 13: write the d(pele)/dT (in Joules/cm^3/eV)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 14: write out the ion specific internal energy (in Joules/gram)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(eion[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 15: write out the electron specific internal energy (in Joules/gram)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(eele[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 16: write out the ion specific heat (in Joules/gram/eV)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 17: write out the electron specific heat (in Joules/gram/eV)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 18: write out d(eion)/d(nion) (not sure)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 19: write out d(eele)/d(nele) (not sure)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 21: write out the energy group boundaries (in ev)
            count = 0
            for i in range(nR+1):
                print("%12.6E" %(rad[i]*1.e3), file=fcn4, end='')
                count = count + 1
                if (count%4 == 0):
                    print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 22: write the Rosseland group opacities
            count = 0
            for k in range(nR):
                for j in range(nD1):
                    for i in range(initindex, nT1):
                        print("%12.6E" %(rosseland_new[i,j,k]), file=fcn4, end='')
                        count = count + 1
                        if (count%4 == 0):
                            print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 23: write the Planck absorption opacities
            count = 0
            for k in range(nR):
                for j in range(nD1):
                    for i in range(initindex, nT1):
                        print("%12.6E" %(planck_new[i,j,k]), file=fcn4, end='')
                        count = count + 1
                        if (count%4 == 0):
                            print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 24: write the Planck emission opacities
            count = 0
            for k in range(nR):
                for j in range(nD1):
                    for i in range(initindex, nT1):
                        print("%12.6E" %(planck_new[i,j,k]), file=fcn4, end='')
                        count = count + 1
                        if (count%4 == 0):
                            print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)

            # close
            fcn4.close()


        #return feos
        #return (temp1, dens1, temp2, dens2)
        #return (rosseland, planck, rosseland_new, planck_new)
        return (temp1, dens1, zbar, eele, pele)


    def BADGER2cn4(self, opacity='Al-TOPS.txt', dest='Al-BADGER-TOPS.cn4', tableformat='TOPS', Abar=26.9815, Zbar=[10], nT1=50, nD1=45, species=1, initindex=0, lR=[10,15], \
                   nT2=71, nD2=66, nR=10, lT2=[13,24], lD2=[26,36],multi_init=4868, useopacity=True, ifwrite=True, massfrac=[1.0]):
        '''
        This function is used to convert BADGER EOS to cn4 format.
        Parameters:
            opacity        - source file of TOPS opacity.
            dest           - dest file in IONMIX4 EOS.
            tableformat    - opacity table format.
            Abar           - Abar.
            Zbar           - Zbar.
            nT1            - point of temperature, EOS.
            nD1            - point of density, EOS.
            species        - n species.
            initindex      - init index of temperature.
            nT2            - point of temperature, OPACITY.
            nD2            - point of density, OPACITY.
            nR             - point of radiation.
            lT2            - lines of radiation, OPACITY.
            lD2            - lines of density, OPACITY.
            multi_init     - line of Multigroup opacities.
            useopacity     - if use opacity table.
            ifwrite        - if write to file.
        Returns:
            None.
        Raises:
            KeyError.
        '''
        import numpy as np
        from scipy import interpolate

        # handle EOS
        # temperature, in eV
        temp1 = np.loadtxt('temperature.txt')
        if (len(temp1) != nT1):
            print('Warning, shape of temperature not match !')

        # density, in /cm^3
        dens1 = np.loadtxt('density.txt')
        if (len(dens1) != nD1):
            print('Warning, shape of density not match !')

        # zbar
        zbar = np.loadtxt('zbar.txt')
        zbar = np.reshape(zbar, [nT1, nD1])

        # dz/dT
        dzdT = np.loadtxt('dzdt.txt')
        dzdT = np.reshape(dzdT, [nT1, nD1])

        # pion, in J/cm^3
        pion = np.loadtxt('i_pressure.txt')
        pion = np.reshape(pion, [nT1, nD1])

        # pele, in J/cm^3
        pele = np.loadtxt('e_pressure.txt')
        pele = np.reshape(pele, [nT1, nD1])

        # eion, in J/g
        eion = np.loadtxt('i_energy.txt')
        eion = np.reshape(eion, [nT1, nD1])

        # eele, in J/g
        eele = np.loadtxt('e_energy.txt')
        eele = np.reshape(eele, [nT1, nD1])
        # for positive heat capacity
        for k in range(100):
            ifcontinue = False
            for i in range(nD1):
                for j in range(nT1-1):
                    if (eele[j,i] >= eele[j+1,i]):
                        ifcontinue = True
                        if (j == 0):
                            eele[j,i] = 0.8 * eele[j+1,i]
                        else:
                            eele[j,i] = (eele[j+1,i] - eele[j-1,i]) / (np.log10(temp1[j+1]) - np.log10(temp1[j-1])) * (np.log10(temp1[j]) - np.log10(temp1[j-1])) + eele[j-1,i]
            print(ifcontinue)
            if (ifcontinue == False):
                break

        # e_cv, in J/g/eV
        e_cv = np.loadtxt('e_cv.txt')
        e_cv = np.reshape(e_cv, [nT1, nD1])

        # i_cv, in J/g/eV
        i_cv = np.loadtxt('i_cv.txt')
        i_cv = np.reshape(i_cv, [nT1, nD1])


        if (useopacity == True):
            if (tableformat == 'TOPS'):
                # handle OPACITY
                fopa = open(opacity)
                lines = fopa.readlines()
                fopa.close()

                # read temperature point, OPACITY
                temp2 = []
                for i in range(lT2[0]-1, lT2[1]):
                    temp2 = temp2 + lines[i].strip().split('  ')
                for i in range(len(temp2)):
                    temp2[i] = float(temp2[i])
                if (len(temp2) != nT2):
                    print("Warning: Not match teperature point, OPACITY!")
                else:
                    temp2 = np.array(temp2)
                # convert from keV to eV
                temp2[:] = temp2[:] * 1.e3

                # read density point, OPACITY
                dens2 = []
                for i in range(lD2[0]-1, lD2[1]):
                    dens2 = dens2 + lines[i].strip().split('  ')
                for i in range(len(dens2)):
                    dens2[i] = float(dens2[i])
                if (len(dens2) != nD2):
                    print("Warning: Not match density point, OPACITY!")
                else:
                    dens2 = np.array(dens2)
                # convert from g/cc to /cc
                dens2[:] = dens2[:] / (Abar / 6.02214076e23)

                # read opacity
                rosseland = np.zeros([nT2, nD2, nR], np.float)
                planck = np.zeros([nT2, nD2, nR], np.float)
                count = 0
                for j in range(nT2):
                    for i in range(nD2):
                        index = multi_init + count * (nR + 1)
                        count = count + 1
                        local = []
                        for k in range(nR):
                            local = local + lines[index+1+k].strip().split('  ')
                        #print(local)
                        for k in range(len(local)):
                            local[k] = float(local[k])
                        local = np.reshape(local, [nR,3])

                        rosseland[j,i,:] = local[:,1]
                        planck[j,i,:] = local[:,2]
                # interpolation of OPACITY
                rosseland_new = np.zeros([nT1,nD1,nR], np.float)
                planck_new = np.zeros([nT1,nD1,nR], np.float)

                for i in range(nR):
                    f1 = interpolate.interp2d(np.log10(dens2), np.log10(temp2), rosseland[:,:,i], kind='linear')
                    rosseland_new[:,:,i] = f1(np.log10(dens1), np.log10(temp1))
                    f2 = interpolate.interp2d(np.log10(dens2), np.log10(temp2), planck[:,:,i], kind='linear')
                    planck_new[:,:,i] = f2(np.log10(dens1), np.log10(temp1))
            elif (tableformat == 'PEOS'):
                # rosseland
                rosseland = np.loadtxt('Au-rosseland.txt')
                rosseland = np.reshape(rosseland, [nD2, nT2])
                planck = np.loadtxt('Au-planckemi.txt')
                planck = np.reshape(planck, [nD2, nT2])

                temp2 = np.loadtxt('Au-temperature.txt')
                temp2 = np.reshape(temp2, nT2)
                dens2 = np.loadtxt('Au-density.txt')
                dens2 = np.reshape(dens2, nD2)

                # interpolate
                rosseland_new = np.zeros([nT1, nD1, 1], np.float)
                planck_new = np.zeros([nT1, nD1, 1], np.float)
                f1 = interpolate.interp2d(np.log10(temp2), np.log10(dens2), rosseland, kind='linear')
                rosseland_new[:,:,0] = np.transpose(f1(np.log10(temp1), np.log10(dens1)))
                f2 = interpolate.interp2d(np.log10(temp2), np.log10(dens2), planck, kind='linear')
                planck_new[:,:,0] = np.transpose(f2(np.log10(temp1), np.log10(dens1)))

        
        if (ifwrite == True):
            if (tableformat == 'TOPS'):
                # write radiation in TOPS format
                # in keV
                #x = np.linspace(-4, 2, 11)
                #rad = 10**x

                rad = []
                for i in range(lR[0]-1, lR[1]):
                    print(lR)
                    rad = rad + lines[i].strip().split('  ')
                    print(lines[i])
                print(rad)
                for i in range(len(rad)):
                    rad[i] = float(rad[i])
                if (len(rad) != (nR)):
                    print("Warning: Not match radiation point, OPACITY!")
                else:
                    rad = np.array(rad)

                frad = open('radiation.txt', 'w+')
                count = 0
                for i in range(nR):
                    print("%9.3E " %(rad[i]), file=frad, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=frad)
                if (count%4 != 0):
                    print("", file=frad)
                frad.close()
            elif (tableformat == 'PEOS'):
                rad = np.array([1.e-4, 1.e2])

            # write IONMIX4 format
            fcn4 = open(dest, 'w+')
            # 1, 2: write the numbver of temperature and density points.
            print("%10d%10d" %(nT1-initindex, nD1), file=fcn4)
            # 3: write the atomic number of each element in this material
            #print(" atomic #s of gases: %10d" %(Zbar), file=fcn4)
            print(" atomic #s of gases: %10d%10d" %(Zbar[0],Zbar[1]), file=fcn4)
            # 4: write the fraction (by number of ions) of each element in this material
            print(" relative fractions:   %10.2E%10.2E" %(massfrac[0],massfrac[1]), file=fcn4)
            # 5: write the number of radiation energy groups
            print("%12d" %(nR), file=fcn4)
            # 6: write the temperature points (in ev)
            count = 0
            for i in range(initindex, nT1):
                print("%12.6E" %(temp1[i]), file=fcn4, end='')
                count = count + 1
                if (count%4 == 0):
                    print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 7: write the number density points (in cm^-3)
            count = 0
            for i in range(nD1):
                print("%12.6E" %(dens1[i]), file=fcn4, end='')
                count = count + 1
                if (count%4 == 0):
                    print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 8: write out the zbar at each temperature/density ponit (nele/nion)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(zbar[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 9: write d(zbar)/dT
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(dzdT[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 10: write the ion pressure
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(pion[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 11: write the electron pressure
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(pele[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 12: write the d(pion)/dT (in Joules/cm^3/eV)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 13: write the d(pele)/dT (in Joules/cm^3/eV)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 14: write out the ion specific internal energy (in Joules/gram)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(eion[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 15: write out the electron specific internal energy (in Joules/gram)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(eele[i,j]), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 16: write out the ion specific heat (in Joules/gram/eV)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 17: write out the electron specific heat (in Joules/gram/eV)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 18: write out d(eion)/d(nion) (not sure)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 19: write out d(eele)/d(nele) (not sure)
            count = 0
            for j in range(nD1):
                for i in range(initindex, nT1):
                    print("%12.6E" %(0.), file=fcn4, end='')
                    count = count + 1
                    if (count%4 == 0):
                        print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 21: write out the energy group boundaries (in ev)
            count = 0
            for i in range(nR):
                print("%12.6E" %(rad[i]*1.e3), file=fcn4, end='')
                count = count + 1
                if (count%4 == 0):
                    print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 22: write the Rosseland group opacities
            count = 0
            for k in range(nR):
                for j in range(nD1):
                    for i in range(initindex, nT1):
                        print("%12.6E" %(rosseland_new[i,j,k]), file=fcn4, end='')
                        count = count + 1
                        if (count%4 == 0):
                            print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 23: write the Planck absorption opacities
            count = 0
            for k in range(nR):
                for j in range(nD1):
                    for i in range(initindex, nT1):
                        print("%12.6E" %(planck_new[i,j,k]), file=fcn4, end='')
                        count = count + 1
                        if (count%4 == 0):
                            print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)
            # 24: write the Planck emission opacities
            count = 0
            for k in range(nR):
                for j in range(nD1):
                    for i in range(initindex, nT1):
                        print("%12.6E" %(planck_new[i,j,k]), file=fcn4, end='')
                        count = count + 1
                        if (count%4 == 0):
                            print("", file=fcn4)
            if (count%4 != 0):
                print("", file=fcn4)

            fcn4.close()

        #return (rosseland, planck, rosseland_new, planck_new) 
        return (temp1, dens1, zbar, eele, pele, eion, pion, e_cv, i_cv)
        #return (temp1, dens1, temp2, dens2, rosseland, planckemi, planckabs)


    def RCF2dose(self, filenumber=1, dosefile='dose.dat', background=65535, density=1.25, dpi=300):
        '''
        Parameters:
            filenumber     - RCF slice number.
            dosefile       - generate dose file.
            background     - background dose.
            density        - RCF active layer density.
            dpi            - scanner dpi.
        Returns:
            None.
        Raises:
            KeyError.
        '''
        import numpy as np
        import cv2 as cv
        import struct

        dv = (2.54/dpi)**2 * 0.0008 # in cm^3
        dE = dv * density * 1e-3    # in kg

        narray = []
        for k in range(1, filenumber+1):
            # read file
            filename = str(k) + '.tif'
            img = cv.imread(filename, 2)
            # convert to optical density(OD)
            OD = -np.log10(img/65535.)
            # minor background
            OD = OD + np.log10(background/65535.)
            # HD-V2 log(OD) = 0.8658*(log(Dose)) - 2.927
            dims = OD.shape
            Dose = np.zeros(dims)
            for j in range(dims[1]):
                for i in range(dims[0]):
                    if (OD[i,j] > 0):
                        Dose[i,j] = 10**((np.log10(OD[i,j]) + 2.927)/0.8658)
                    else:
                        Dose[i,j] = 0.
                    # convert to energy
                    Dose[i,j] = Dose[i,j] * dE   # in J
            narray.append(Dose)
        # write to file
        files = open(dosefile, 'wb')
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(filenumber):
                    files.write(struct.pack('d', narray[k][i,j]))
        files.close()

        return narray
