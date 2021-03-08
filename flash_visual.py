# -*- coding: utf-8 -*-
#*************************************************************************
#***File Name: flash_visual.py
#***Author: Zhonghai Zhao
#***Mail: zhaozhonghi@126.com 
#***Created Time: 2018年03月25日 星期日 14时39分05秒
#*************************************************************************
class flash_visual(object):
    '''
    This class contains some functions to visualize FLASH output data.
    '''
    # initialization
    def __init__(self):
        pass
    def plot_block(self, prefix='lasslab', filenumber=[0], keyword='cnt', field='dens', refine=0, vrange=[0, 2.7], box_panning=[0.1, 0.1, 0.1], \
                   box_scale=0.8, geom=[0., 1., 0., 1., 0., 1.], dimension=2, axis='z', coordinate=0., ngrid=[8,8,8], \
                   figure_size=(4, 8), geom_factor=10000, cb_position=[0.20, 0.05, 0.70, 0.02], component='z', resistivity=1e-3, rotation='horizontal', \
                   cb_label=r'$ KGs $', cb_ticks=[(-1e3,0,1e3), ('-1','0','1')], text=[0.1, 0.1, 'Te', False], \
                   iflog=False, ifexist=True, ifdisplay=True, axisoff=False, dpi=300):
        '''
        This function is used to visualize 2d FLASH output data.
        Parameters:
            prefix         - FLASH output file prefix.
            filenumber     - FLASH output file number.
            keyword        - FLASH output file keyword.
            field          - physical field.
            refine         - refine level.
            vrange         - plot range.
            box_scale      - total box scale.
            box_panning    - box panning
            geom           - 2d geometry axis.
            figure_size    - figure size.
            geom_factor    - 2d geometry axis factor.
            cb_position    - colorbar position.
            component      - vector component('x', 'y' or 'z').
            resistivity    - electric resistivity.
            ifdisplay      - if display figure.
            ifexist        - if the field exist or need to compute.
            axisoff        - if turn on axis.
            dpi            - figuredpi.
        Returns:
            None.
        Raises:
            KeyError.
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        import matplotlib.colorbar as colorbar
        import flash_class
        fc = flash_class.flash_class()

        #constant = fc.get_constant(field)
        constant = 1
        length = len(filenumber)
        for j in range(length):
            data = fc.get_data(prefix=prefix, filenumber=filenumber[j], keyword=keyword)
            block = fc.get_block(data, refine=refine, geom=geom, box_scale=box_scale, box_panning=box_panning)
            # compute
            if (ifexist == True):
                field_data = data[field][:]
            else:
                if (field == 'electric_field'):
                    field_data = fc.compute_field(field_data=data, block=block, field=field, component=component, resistivity=resistivity)
                elif (field == 'tension_force'):
                    field_data = fc.compute_field(field_data=data, block=block, field=field, component=component, resistivity=resistivity)
                elif (field == 'magnetic_pressure_force'):
                    field_data = fc.compute_field(field_data=data, block=block, field=field, component=component, resistivity=resistivity)
                elif (field == 'thermal_pressure_force'):
                    field_data = fc.compute_field(field_data=data, block=block, field=field, component=component, resistivity=resistivity)
                elif (field == 'total_pressure_force'):
                    field_data = fc.compute_field(field_data=data, block=block, field=field, component=component, resistivity=resistivity)
                else:
                    print('No match field data!')
            # plot
            font1 = 20
            font2 = 12
            fontdict1 = {
                    'family':'serif',
                    'style' :'italic',
                    'weight':'normal',
                    'color' :'black',
                    'size'  :font1}
            fontdict2 = {
                    'family':'serif',
                    'style' :'italic',
                    'weight':'normal',
                    'color' :'black',
                    'size'  :font2}
            n = len(block)
            fig = plt.figure(figsize=figure_size)
            figinfo = fc.get_figinfo(box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, axis=axis, coordinate=coordinate, ngrid=ngrid)
            #plt.axes([box_panning[0], box_panning[1], box_scale, box_scale], xlim=[geom[0]*geom_factor, geom[1]*geom_factor], ylim=[geom[2]*geom_factor, geom[3]*geom_factor])
            plt.axes(figinfo[0], xlim=figinfo[1], ylim=figinfo[2])
            #plt.title(field + ' ' + r'$ time = ' + str(filenumber[j]) + ' t_0 $', fontsize=20)
            plt.xlabel(figinfo[3][0], fontsize=16)
            plt.ylabel(figinfo[3][1], fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            color_map = cm.RdBu_r
            #color_map = cm.rainbow
            #color_map = cm.Blues
            narray = fc.get_plane(field_data=field_data, block=block, dimension=dimension, axis=axis, coordinate=figinfo[4], ngrid=ngrid, ifexist=ifexist)
            for i in range(len(narray)):
                ax = plt.axes(narray[i][0])
                if (ifexist == True):
                    if (iflog == False):
                        ax = plt.imshow(narray[i][1]/constant, origin='lower', cmap=color_map, aspect='auto', vmin=vrange[0], vmax=vrange[1])
                    else:
                        #sign = narray[i][1] / np.abs(narray[i][1])
                        #lognarray = np.abs(np.log10(np.abs(narray[i][1])))
                        #ax = plt.imshow(sign * lognarray/constant, origin='lower', cmap=color_map, aspect='auto', vmin=vrange[0], vmax=vrange[1])
                        ax = plt.imshow(np.log10(np.abs(narray[i][1]))/constant, origin='lower', cmap=color_map, aspect='auto', vmin=vrange[0], vmax=vrange[1])
                else:
                    if (iflog == False):
                        ax = plt.imshow(field_data[i]/constant, origin='lower', cmap=color_map, aspect='auto', vmin=vrange[0], vmax=vrange[1])
                    else:
                        ax = plt.imshow(np.log10(field_data[i]/constant), origin='lower', cmap=color_map, aspect='auto', vmin=vrange[0], vmax=vrange[1])
                setax = plt.gca()
                setax.spines['top'].set_linewidth(0.1)
                setax.spines['bottom'].set_linewidth(0.1)
                setax.spines['left'].set_linewidth(0.1)
                setax.spines['right'].set_linewidth(0.1)
                plt.xticks([])
                plt.yticks([])
                if (axisoff == True):
                    plt.axis('off')
            # add color bar
            color_bar = fig.add_axes(cb_position)
            norm = colors.Normalize(vmin=vrange[0], vmax=vrange[1])
            cb = colorbar.ColorbarBase(color_bar, cmap=color_map, norm=norm, orientation=rotation)
            #cb.set_label(cb_label, fontdict=fontdict2)
            cb.set_ticks(cb_ticks[0])
            cb.set_ticklabels(cb_ticks[1])
            cb.ax.tick_params(labelsize=12)
            if (text[3] == True):
                plt.text(text[0], text[1], text[2], fontsize=font2)
            if (ifdisplay == True):
                plt.show()
            else:
                s1 = 'figure/'
                s2 = field + '_' + axis + '_ '+ str(filenumber[j]).zfill(4) + '.png'
                path = s1 + s2
                plt.savefig(path, dpi=dpi)
            plt.close()

        return narray

    def plot_line(self, prefix='lasslab', filenumber=[0], keyword='cnt', info='none', field='dens', refine=0, box_panning=[0.1, 0.1, 0.1], \
                  box_scale=0.8, geom=[0, 1, 0, 1, 0, 1], dimension=2, axis=['z', 'x'], coordinate=[0., 0.], ngrid=[8,8,8], \
                  figure_size=(8, 5), xlim=[0,1], ylim=[0,1], geom_factor=1,\
                  ifdirect=True, ifexist=True, ifrecontruct=True, ifdisplay=True):
        '''
        This function is used to visualize 2d FLASH output data.
        Parameters:
            prefix         - FLASH output file prefix.
            filenumber     - FLASH output file number.
            keyword        - FLASH output file keyword.
            info           - information.
            field          - physical field.
            refine         - refine level.
            box_scale      - total box scale.
            box_panning    - box panning
            geom           - 2d geometry axis.
            dimension      - dimensions.
            axis           - coordinate axis.
            coordinate     - coordinate to be ploted.
            ngrid          - number of grid in each block.
            figure_size    - figure size
            xlim           - x axis range.
            ylim           - y axis range.
            geom_factor    - 2d geometry axis factor.
            ifreconstruct  - if reconstruct data, always True.
            ifdisplay      - if display figure.
        Returns:
            None.
        Raises:
            KeyError.
        '''
        import matplotlib.pyplot as plt
        import flash_class
        fc = flash_class.flash_class()

        line_list = fc.line_set()
        constant = fc.get_constant(field)
        plot_lines = []
        length = len(filenumber)
        for j in range(length):
            data = fc.get_data(prefix=prefix, filenumber=filenumber[j], keyword=keyword)
            block = fc.get_block(data, refine=refine, geom=geom, box_scale=box_scale, box_panning=box_panning)
            field_data = data[field][:]
            n = len(block)
            figinfo = fc.get_figinfo(box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor, axis=axis[0], coordinate=coordinate[0], ngrid=ngrid)
            narray = fc.get_plane(field_data=field_data, block=block, dimension=dimension, axis=axis[0], coordinate=figinfo[4], ngrid=ngrid, ifexist=ifexist)
            if (ifdirect == True):
                nline = fc.get_line(narray=narray, dimension=dimension, box_scale=box_scale, box_panning=box_panning, geom=geom, axis=axis, coordinate=coordinate[1], ngrid=ngrid)
                lines = fc.reconstruct(nline=nline, axis=axis, ngrid=ngrid, box_scale=box_scale, box_panning=box_panning, geom=geom, geom_factor=geom_factor)
                plot_lines.append(lines)
            else:
                # convert block to array
                pass
        return plot_lines
        # plot
        font1 = 20
        font2 = 16
        for axis_plot in 'xyz':
            if (axis_plot not in axis):
                break
        fig = plt.figure(figsize=figure_size)
        for j in range(length):
            plt.plot(plot_lines[j][0], plot_lines[j][1], linewidth=2, label='time = ' + str(filenumber[j]))
        plt.xlabel(axis_plot, fontsize=font1)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(fontsize=font2)
        plt.yticks(fontsize=font2)
        plt.legend(fontsize=font1)
        plt.show()

        return plot_lines
    #
    def time_evolution(self, prefix='Harris_mhd_2d', filenumber=[0], keyword='cnt', case='energy', refine=0, geom=[-20, 20, -20, 20], box_scale=0.8, box_panning=[0.1, 0.1], ngrid=16):
        '''
        '''

        import numpy as np
        import matplotlib.pyplot as plt
        import flash_class
        fc = flash_class.flash_class()
        length = len(filenumber)

        # calculation
        array = np.zeros([10, length])
        for i in range(length):
            data = fc.get_data(prefix=prefix, filenumber=filenumber[i], keyword=keyword)
            block = fc.get_block(data, refine=refine, geom=geom, box_scale=box_scale, box_panning=box_panning)
            # select case
            if (case == 'energy'):
                energy = fc.compute_energy(field_data=data, block=block, ngrid=ngrid)
                for j in range(len(energy)):
                    array[j, i] = energy[j]
            else:
                pass
        # plot
        plt.plot(array[0,:])
        plt.plot(array[1,:])
        plt.plot(array[2,:])
        plt.plot(array[3,:])
        plt.plot(array[4,:])
        plt.show()

    #
    def plot_array(self, array=[], info='PI', prefix='PItest', filenumber=[1], geom=[0,1,0,1], geom_factor=1., vrange=[0,1], shape=[512,512], figure_size=(8,8), \
                   fig_position=[0.15, 0.15, 0.70, 0.70], rotation='vertical', cb_label=r'$ KGs $', cb_ticks=[(-1e3,0,1e3), ('-1','0','1')], extent=[0,1,0,1], \
                   plottype='imshow', text=[1.5, 1.02, r'$ \times 10^2 $', False], postfix='cal', ifdisplay=True, ifcolorbar=False):
        '''
        This function is used to plot Proton Imaging data in FLASH.
        Parameters:
            array          - data array.
            info           - case information.
            prefix         - FLASH output file prefix.
            filenumber     - FLASH output file number.
            geom           - 2d geometry axis.
            geom_factor    - 2d geometry axis factor.
            vrange         - plot range.
            shape          - shape of array to hold density.
            figure_size    - figure size.
            fig_position   - figure position.
            rotation       - colorbar direction.
            cb_label       - colorbar label.
            cb_ticks       - colorbar ticks.
            ifdisplay      - if display figure.
        '''
        import numpy as np
        import copy
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        import matplotlib.colorbar as colorbar
        import flash_class
        fc = flash_class.flash_class()
        #from constants import turbMR as const

        lens = len(filenumber)
        for k in range(lens):
            if (info == 'PI'):
                color_map = cm.RdBu_r
                extent = [geom[0]*geom_factor, geom[1]*geom_factor, geom[2]*geom_factor, geom[3]*geom_factor]
                coord = fc.get_PI(prefix=prefix, filenumber=filenumber[k])
                array = fc.particle_to_grid(shape=shape, coord=coord)
            elif (info == 'epoch_de'):
                color_map = cm.RdBu_r
                #array = array / (const.J0 * const.E0)
                array = np.transpose(array)
                extent = [0, 51.2, 0, 51.2]
            elif (info == 'problem_image'):
                if (plottype == 'imshow'):
                    #color_map = cm.OrRd
                    color_map = cm.RdBu_r
                    extent = [-1.5,1.5,-1.5,1.5]
                    xlabel = r'$ \rm X/mm $'
                    ylabel = r'$ \rm Y/mm $'
                    xticks = ([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5], ['-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5'])
                    yticks = ([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5], ['-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5'])
                    cb_label = r'$ \rm Path-Int-Mag \quad B(Gs \cdot cm) $'
                    cb_ticks = ([-1500, -1000, -500, 0, 500, 1000, 1500], ['-15', '-10', '-5', '0', '5', '10', '15'])
                    #cb_ticks = ([0, 500, 1000, 1500, 2000], ['0', '5', '10', '15', '20'])
                    #cb_ticks = ([-1500, -1000, -500, 0, 500, 1000, 1500], ['-15', '-10', '-5', '0', '5', '10', '15'])
                elif (plottype == 'stream'):
                    color_map = cm.RdYlGn_r
                    xlabel = r'$ \rm X/mm $'
                    ylabel = r'$ \rm Y/mm $'
                    xticks = ([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5], ['-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5'])
                    yticks = ([-1.0, -0.5, 0, 0.5, 1.0], ['-1.0', '-0.5', '0.0', '0.5', '1.0'])
                    cb_label = r'$ \rm Path-Int-Mag \quad B(Gs \cdot cm) $'
                    cb_ticks = ([-1500, -1000, -500, 0, 500, 1000, 1500], ['-15', '-10', '-5', '0', '5', '10', '15'])
            elif (info == 'contrast'):
                #color_map = cm.OrRd
                color_map = cm.viridis
                extent = [-1.5,1.5,-1,1]
                xlabel = r'$ \rm X/mm $'
                ylabel = r'$ \rm Y/mm $'
                xticks = ([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5], ['-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5'])
                yticks = ([-1.0, -0.5, 0, 0.5, 1.0], ['-1.0', '-0.5', '0.0', '0.5', '1.0'])
                cb_label = r'$ \rm contrast \quad \delta \Psi / \Psi_{0} $'
                cb_ticks = ([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3], ['-0.3', '-0.2', '-0.1', '0.0', '0.1', '0.2', '0.3'])
            elif (info == 'proton_radiography'):
                if (plottype == 'imshow'):
                    color_map = cm.RdBu_r
                    extent = [-1,1,-1,1]
                    xlabel = r'$ \rm X $'
                    ylabel = r'$ \rm Y $'
                    xticks = ([-1, -0.5, 0, 0.5, 1], ['-1.0', '-0.5', '0.0', '0.5', '1.0'])
                    yticks = ([-1, -0.5, 0, 0.5, 1], ['-1.0', '-0.5', '0.0', '0.5', '1.0'])
                    cb_label = r'$ \rm contrast \quad (\psi - \psi_0) / \psi_0 $'
                    cb_ticks = ([-1, 0, 1], ['-1.0', '0', '1.0'])
                elif (plottype == 'stream'):
                    color_map = cm.RdYlGn_r
                    xlabel = r'$ \rm X $'
                    ylabel = r'$ \rm Y $'
                    xticks = ([-1, -0.5, 0, 0.5, 1], ['-1.0', '-0.5', '0.0', '0.5', '1.0'])
                    yticks = ([-1, -0.5, 0, 0.5, 1], ['-1.0', '-0.5', '0.0', '0.5', '1.0'])
                    cb_label = r'$ \rm Path-Int-Mag \quad B(Gs \cdot cm) $'
                    cb_ticks = ([0, 500, 1000, 1500], ['0', '5', '10', '15'])
            else:
                pass
        if (plottype == 'stream'):
            # data
            U = array[0]
            V = array[1]
            X = np.linspace(-1.5,1.5, U.shape[1])
            Y = np.linspace(-1,1, U.shape[0])
            carray = np.sqrt(U**2 + V**2)

        font1 = 20
        font2 = 16
        fontdict1 = {
                'family':'serif',
                'style' :'italic',
                'weight':'normal',
                'color' :'black',
                'size'  :font1}
        fontdict2 = {
                'family':'serif',
                'style' :'italic',
                'weight':'normal',
                'color' :'black',
                'size'  :font2}

        fig = plt.figure(figsize=figure_size)
        plt.axes(fig_position)
        if (plottype == 'imshow'):
            #plt.imshow(array, extent=extent, origin='lower', cmap=color_map, aspect='auto', vmin=vrange[0], vmax=vrange[1])
            plt.imshow(array, extent=extent, cmap=color_map, aspect='auto', vmin=vrange[0], vmax=vrange[1])
        elif (plottype == 'stream'):
            plt.streamplot(X, Y, U, V, color=carray, cmap=color_map, linewidth=2, density=2)
        plt.xlabel(xlabel, fontdict=fontdict1)
        plt.ylabel(ylabel, fontdict=fontdict1)
        plt.xticks(xticks[0], xticks[1], fontsize=font2)
        plt.yticks(yticks[0], yticks[1], fontsize=font2)
        # add color bar
        cb_position = copy.deepcopy(fig_position)
        cb_position[0] = fig_position[0] + + fig_position[2] + 0.02
        cb_position[1] = fig_position[1]
        cb_position[2] = 0.02
        cb_position[3] = fig_position[3]
        color_bar = fig.add_axes(cb_position)
        norm = colors.Normalize(vmin=vrange[0], vmax=vrange[1])
        cb = colorbar.ColorbarBase(color_bar, cmap=color_map, norm=norm, orientation=rotation)
        cb.set_label(cb_label, fontdict=fontdict1)
        cb.set_ticks(cb_ticks[0])
        cb.set_ticklabels(cb_ticks[1])
        cb.ax.tick_params(labelsize=12)
        if (text[3] == True):
            plt.text(text[0], text[1], text[2], fontsize=font2)

        if (ifdisplay == True):
            plt.show()
        else:
            figname = info + '_' + postfix + '.png'
            plt.savefig(figname, dpi=300)
        plt.close()


#        return coord

    def calcu_field(self, prefix='lasslab', filenumber=0, keyword='cnt', info='turb_bs', refine=0, field_data=[], \
                    box_panning=[0.1, 0.1, 0.1], box_scale=0.8, geom=[0., 1., 0., 1., 0., 1.], dimension=2, axis='z', coordinate=0., \
                    ngrid=[8,8,8], nblock=[12,12,1], geom_factor=10000, ifexist=True):
        '''
        This function is used to calculate fields inner informaion.
        Parameters:
            prefix         - FLASH output file prefix.
            filenumber     - FLASH output file number.
            keyword        - FLASH output file keyword.
            refine         - refine level.
            box_scale      - total box scale.
            box_panning    - box panning
            geom           - 2d geometry axis.
            geom_factor    - 2d geometry axis factor.
            ngrid          - grids of block in each direction.
            nblock         - block in each direction.
            ifexist        - if read data file.
            field_data     - data external.
        Returns:
            line.
        Raises:
            KeyError.
        '''
        import numpy as np
        import flash_class
        fc = flash_class.flash_class()

        if (ifexist == True):
            data = fc.get_data(prefix=prefix, filenumber=filenumber, keyword=keyword)
            block = fc.get_block(data=data, box_scale=box_scale, box_panning=box_panning, geom=geom)
        else:
            pass
        if(info == 'turb_bs'):
            ibx = data['magx'][:]
            nbx = fc.get_plane(field_data=ibx, block=block, dimension=dimension, axis=axis, ngrid=ngrid)
            bx = fc.grid_UG(narray=nbx, ngrid=ngrid, nblock=nblock)
            fftbx = np.fft.fft2(bx)
            mbx = np.abs(fftbx)**2
            iby = data['magy'][:]
            nby = fc.get_plane(field_data=iby, block=block, dimension=dimension, axis=axis, ngrid=ngrid)
            by = fc.grid_UG(narray=nby, ngrid=ngrid, nblock=nblock)
            fftby = np.fft.fft2(by)
            mby = np.abs(fftby)**2
            modeb = mbx + mby
            nx, ny = modeb.shape
            nx = int(nx/2)
            ny = int(ny/2)
            spec = fc.turb_energy(array=modeb[:nx, :ny], ngrid=500)
        elif (info == 'epoch_turb_bs'):
            bx = field_data[0]
            fftbx = np.fft.fft2(bx)
            mbx = np.abs(fftbx)**2
            by = field_data[1]
            fftby = np.fft.fft2(by)
            mby = np.abs(fftby)**2
            modeb = mbx + mby
            nx, ny = modeb.shape
            nx = int(nx/2)
            ny = int(ny/2)
            spec = fc.turb_energy(array=modeb[:nx, :ny], ngrid=500)
        else:
            pass



        return spec

    def multi_line(self, x=[], y=[], xlim=[], ylim=[], info='dynamic_alignment', iflog=False, ifdisplay=True, iftwin=False):
        '''
        This function is used to plot multi-lines.
        Parameters:
            x              - list of X.
            y              - list of Y.
            xlim           - x axis range.
            ylim           - y axis range.
            iflog          - if compute log.
            ifdisplay      - if display figure.
        Returns:
            None.
        Raises:
            KeyError.
        '''
        import numpy as np
        import matplotlib.pyplot as plt

        length = len(x)

        if (info == 'dynamic_alignment'):
            xlabel = r'$ \rm \Delta r $'
            ylabel = r'$ \rm \theta_r $'
            xticks = ([1e-4, 1e-3, 1e-2, 1e-1], [r'$ 10^{-4} $', r'$ 10^{-3} $', r'$ 10^{-2} $', r'$ 10^{-1} $'])
            yticks = ([1e-1, 1e0], [r'$ 10^{-1} $', r'$ 10^{0} $'])
            legend = (
                    r'$ R_m = 10^6 $',
                    r'$ \Delta r^{0.25} $'
                    )
        elif (info == 'turb_energy'):
            xlabel = r'$ \rm k $'
            ylabel = r'$ \rm E(k) $'
            xticks = ([1e1, 1e2, 1e3, 1e4], [r'$ 10^{1} $', r'$ 10^{2} $', r'$ 10^{3} $', r'$ 10^{4} $'])
            yticks = ([1e6, 1e8, 1e10, 1e12, 1e14], [r'$ 10^{6} $', r'$ 10^{8} $', r'$ 10^{10} $', r'$ 10^{12} $', r'$ 10^{14} $'])
            legend = (
                    r'$ E_B(k) $',
                    r'$ k^{-1.7} $',
                    r'$ k^{-2.5} $'
                    )
        elif (info == 'pdf'):
            xlabel = r'$ \rm (\Delta B_x - \overline{\Delta B_x}) / \Sigma_{\Delta B_x} $'
            ylabel = r'$ \rm PDF $'
            xticks = ([-30, -20, -10, 0, 10, 20, 30], [r'$ -30 $', r'$ -20 $', r'$ -10 $', r'$ 0 $', r'$ 10 $', r'$ 20 $', r'$ 30 $'])
            yticks = ([1e2, 1e3, 1e4, 1e5, 1e6, 1e7], [r'$ 10^{2} $', r'$ 10^{3} $', r'$ 10^{4} $', r'$ 10^{5} $', r'$ 10^{6} $', r'$ 10^{7} $'])
            legend = (
                    r'$ \Delta r = 10^{-1} $',
                    r'$ \Delta r = 10^{-2} $',
                    r'$ \Delta r = 10^{-3} $',
                    )
        elif (info == 'pdf_theta'):
            xlabel = r'$ \rm \tilde{\theta}_{r} $'
            ylabel = r'$ \rm PDF $'
            xticks = ([1e-3, 1e-2, 1e-1, 1e0], [r'$ 10^{-3} $', r'$ 10^{-2} $', r'$ 10^{-1} $', r'$ 10^{0} $'])
            #yticks = ([1e5, 1e6], [r'$ 10^{5} $', r'$ 10^{6} $'])
            yticks = ([3e5, 4e5, 5e5, 6e5], [r'$ 3 \ times 10^{5} $', r'$ 4 \times 10^{5} $', r'$ 5 \times 10^{5} $', r'$ 6 \times 10^{5} $'])
            legend = (
                    r'$ \Delta r = 10^{-1} $',
                    r'$ \Delta r = 10^{-2} $',
                    r'$ \Delta r = 10^{-3} $',
                    )
        elif (info == 'multi_spec'):
            xlabel = r'$ \rm k $'
            ylabel = r'$ \rm E(k) $'
            xticks = ([1e1, 1e2, 1e3], [r'$ 10^{1} $', r'$ 10^{2} $', r'$ 10^{3} $'])
            yticks = ([1e6, 1e8, 1e10, 1e12, 1e14], [r'$ 10^{6} $', r'$ 10^{8} $', r'$ 10^{10} $', r'$ 10^{12} $', r'$ 10^{14} $'])
            legend = (
                    r'$ R_m = 10^4 $',
                    r'$ R_m = 10^5 $',
                    r'$ R_m = 10^6 $',
                    r'$ k = -3/2 $'
                    )
        elif (info == 'epoch_spec'):
            xlabel = r'$ \rm k $'
            ylabel = r'$ \rm E(k) $'
            xticks = ([1e1, 1e2, 1e3], [r'$ 10^{1} $', r'$ 10^{2} $', r'$ 10^{3} $'])
            yticks = ([1e6, 1e8, 1e10, 1e12, 1e14], [r'$ 10^{6} $', r'$ 10^{8} $', r'$ 10^{10} $', r'$ 10^{12} $', r'$ 10^{14} $'])
            legend = (
                    r'$ t = 10 \omega_{ci}^{-1} $',
                    r'$ t = 20 \omega_{ci}^{-1} $',
                    r'$ t = 30 \omega_{ci}^{-1} $',
                    r'$ t = 40 \omega_{ci}^{-1} $',
                    )
        elif (info == 'epoch_enspec'):
            xlabel = r'$ \rm E/MeV $'
            ylabel = r'$ \rm dN /dE $'
            xticks = ([1e-3, 1e-2, 1e-1], [r'$ 10^{-3} $', r'$ 10^{-2} $', r'$ 10^{-1} $'])
            yticks = ([1e13, 1e15, 1e17, 1e19], [r'$ 10^{13} $', r'$ 10^{15} $', r'$ 10^{17} $', r'$ 10^{19} $'])
            legend = (
                    r'$ t = 0 \omega_{ci}^{-1} $',
                    r'$ t = 80 \omega_{ci}^{-1} $'
                    )
        elif (info == 'epoch_de'):
            xlabel = r'$ \rm (\Delta D_e - \overline{\Delta D_e}) / \Sigma_{\Delta D_e} $'
            ylabel = r'$ \rm PDF $'
            xticks = ([-15, -10, -5, 0, 5, 10, 15], [r'$ -15 $', r'$ -10 $', r'$ -5 $', r'$ 0 $', r'$ 5 $', r'$ 10 $', r'$ 15 $'])
            yticks = ([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], [r'$ 10^{-5} $', r'$ 10^{-4} $', r'$ 10^{-3} $', r'$ 10^{-2} $', r'$ 10^{-1} $'])
            legend = (
                    r'$ \Delta r = 0.1 d_i $',
                    r'$ \Delta r = 1.0 d_i $',
                    r'$ \Delta r = 10.0 d_i $',
                    )
        elif (info == 'epoch_bx'):
            xlabel = r'$ \rm (\Delta B_x - \overline{\Delta B_x}) / \Sigma_{\Delta B_x} $'
            ylabel = r'$ \rm PDF $'
            xticks = ([-10, -5, 0, 5, 10], [r'$ -10 $', r'$ -5 $', r'$ 0 $', r'$ 5 $', r'$ 10 $'])
            yticks = ([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], [r'$ 10^{-5} $', r'$ 10^{-4} $', r'$ 10^{-3} $', r'$ 10^{-2} $', r'$ 10^{-1} $'])
            legend = (
                    r'$ \Delta r = 0.1 d_i $',
                    r'$ \Delta r = 1.0 d_i $',
                    r'$ \Delta r = 10.0 d_i $'
                    )
        elif (info == 'ne_te'):
            xlabel = r'$ \rm X/\mu m $'
            ylabel1 = r'$ \rm n_e/cm^{-3} $'
            ylabel2 = r'$ \rm T_e/eV $'
            xticks = ([0, 400, 800, 1200], [r'$ 0 $', r'$ 400 $', r'$ 800 $', r'$ 1200 $'])
            yticks1 = ([1e18, 1e19, 1e20, 1e21, 1e22, 1e23], [r'$ 10^{18} $', r'$ 10^{19} $', r'$ 10^{20} $', r'$ 10^{21} $', r'$ 10^{22} $', r'$ 10^{23} $'])
            yticks2 = ([1e3, 1e4, 1e5, 1e6, 1e7], [r'$ 10^{3} $', r'$ 10^{4} $', r'$ 10^{5} $', r'$ 10^{6} $', r'$ 10^{7} $'])
            legend = (
                    r'$ \rm n_e(CH) $',
                    r'$ \rm n_e(Cu) $',
                    r'$ \rm T_e(CH) $',
                    r'$ \rm T_e(Cu) $'
                    )
        elif (info == 'lambda_de'):
            xlabel = r'$ \rm X/\mu m $'
            ylabel = r'$ \rm \lambda_{mfp} / de $'
            xticks = ([0, 400, 800, 1200], [r'$ 0 $', r'$ 400 $', r'$ 800 $', r'$ 1200 $'])
            yticks = ([0, 50, 100, 150, 200, 250], [r'$ 0 $', r'$ 50 $', r'$ 100 $', r'$ 150 $', r'$ 200 $', r'$ 250 $'])
            legend = (
                    r'$ \rm CH $',
                    r'$ \rm Cu $'
                    )
        elif (info == 'thermalinstability'):
            xlabel = r'$ \rm X/\mu m $'
            ylabel = r'$ \rm \gamma_{m} / ns^{-1} $'
            xticks = ([0, 400, 800, 1200], [r'$ 0 $', r'$ 400 $', r'$ 800 $', r'$ 1200 $'])
            yticks = ([0, 3, 6, 9, 12, 15], [r'$ 0 $', r'$ 3 $', r'$ 6 $', r'$ 9 $', r'$ 12 $', r'$ 15 $'])
            legend = (
                    r'$ \rm CH $',
                    r'$ \rm Cu $'
                    )
        elif (info == 'threashold'):
            xlabel = r'$ \rm \chi_{e}(\omega_{ce}\tau_{e}) $'
            ylabel = r'$ \rm f_{\chi} $'
            xticks = ([0, 5, 10, 15, 20], [r'$ 0 $', r'$ 5 $', r'$ 10 $', r'$ 15 $', r'$ 20 $'])
            yticks = ([0, 0.5, 1.0, 1.5], [r'$ 0 $', r'$ 0.5 $', r'$ 1.0 $', r'$ 1.5 $'])
            legend = (
                    r'$ \rm CH $',
                    r'$ \rm Cu $'
                    )
        elif (info == 'velocity'):
            xlabel = r'$ \rm X/\mu m $'
            ylabel = r'$ \rm velocity/(km/s) $'
            xticks = ([0, 100, 200, 300, 400, 500], [r'$ 0 $', r'$ 100 $', r'$ 200 $', r'$ 300 $', r'$ 400 $', r'$ 500 $'])
            yticks = ([-2e5, -1e5, 0, 1e5, 2e5, 3e5, 4e5, 5e5], [r'$ -200 $', r'$ -100 $', r'$ 0 $', r'$ 100 $', r'$ 200 $', r'$ 300 $', r'$ 400 $', r'$ 500 $'])
            legend = (
                    r'$ \rm v_{adv}(CH) $',
                    r'$ \rm v_{adv}(Cu) $',
                    r'$ \rm v_{i}(CH) $',
                    r'$ \rm v_{i}(Cu) $',
                    r'$ \rm v_{N}(CH) $',
                    r'$ \rm v_{N}(Cu) $'
                    )
        elif (info == 'flash_spec'):
            xlabel = r'$ \rm E_{k}(eV) $'
            ylabel = r'$ \rm dN/dE_{k} $'
            xticks = ([1e1, 1e2, 1e3, 1e4, 1e5], [r'$ 10^{1} $', r'$ 10^{2} $', r'$ 10^{3} $', r'$ 10^{4} $', r'$ 10^{5} $'])
            yticks = ([1e-1, 1e-2, 1e-3, 1e-4, 1e-5], [r'$ 10^{-1} $', r'$ 10^{-2} $', r'$ 10^{-3} $', r'$ 10^{-4} $', r'$ 10^{-5} $', r'$ 10^{-6} $'])
            legend = (
                    r'$ t = 0 ns $',
                    r'$ t = 20 ns (\beta=10) $',
                    r'$ t = 20 ns (\beta=2.5) $',
                    r'$ t = 20 ns (\beta=0.4) $',
                    r'$ t = 4 ns (\beta=2.5) $'
                    )
        else:
            pass

        lw = 2
        font1 = 20
        font2 = 16
        fontdict1 = {
                'family':'serif',
                'style' :'italic',
                'weight':'normal',
                'color' :'black',
                'size'  :font1}
        fontdict2 = {
                'family':'serif',
                'style' :'italic',
                'weight':'normal',
                'color' :'black',
                'siize'  :font2}

        plt.figure(figsize=(8, 5))
        if (iftwin == False):
            color_set = ['r-*', 'g-+', 'b-^', 'k->', 'c--']
            ax = plt.axes([0.15,0.15,0.8,0.8])
            plt.xlabel(xlabel, fontsize=font1)
            plt.ylabel(ylabel, fontsize=font1)
            plt.xticks(xticks[0], xticks[1], fontsize=font2)
            plt.yticks(yticks[0], yticks[1], fontsize=font2)
            plt.xlim(xlim)
            plt.ylim(ylim)
            for i in range(length):
                if (i < 7):
                    plt.plot(x[i], y[i], color_set[i], label=legend[i])
                else:
                    plt.plot(x[i], y[i], 'k--')
            plt.legend(fontsize=font2)
            if (iflog == True):
                plt.xscale('log')
                plt.yscale('log')
        else:
            color_set = ['r--', 'r-.', 'g--', 'g-.', 'b--', 'b-.']
            ax = plt.axes([0.14,0.15,0.75,0.8])
            plt.xlabel(xlabel, fontsize=font1)
            plt.ylabel(ylabel1, fontsize=font1)
            plt.xticks(xticks[0], xticks[1], fontsize=font2)
            plt.yticks(yticks1[0], yticks1[1], fontsize=font2)
            plt.xlim(xlim)
            for i in range(2):
                plt.plot(x[i], y[i], color_set[i], label=legend[i])
            plt.yscale('log')
            #ax.legend(loc=0, fontsize=font2)
            ax2 = plt.twinx()
            plt.ylabel(ylabel2, fontsize=font1)
            plt.yticks(yticks2[0], yticks2[1], fontsize=font2)
            for i in range(2,4):
                plt.plot(x[i], y[i], color_set[i], label=legend[i])
            plt.yscale('log')
            #ax2.legend(loc=0, fontsize=font2)
            lin1 = ax.get_lines()
            lin2 = ax2.get_lines()
            plt.legend(lin1+lin2, legend, fontsize=font2)


        plt.text(5, 3e15, r'$ k_{\perp} = - 1.4 $', fontsize=12)
        plt.text(25, 8e13, r'$ k_{\perp} = - 3.5 $', fontsize=12)

        if (ifdisplay == True):
            plt.show()
        else:
            #path = 'figure/' + info + '.png'
            path = info + '.png'
            plt.savefig(path, dpi=300)

        plt.close()


