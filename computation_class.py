# -*- coding: utf-8 -*-
#*************************************************************************
#***File Name: computation_class.py
#***Author: Zhonghai Zhao
#***Mail: zhaozhonghi@126.com 
#***Created Time: 2018年07月18日 星期三 13时05分55秒
#*************************************************************************
class computation_class(object):
    '''
    This class contains some function used in computation.
    '''
    # initialization
    def __init__(self):
        pass
    #*********************************************************************#
    def slope_limiter(self, a, b, limiter='minmod'):
        '''
        Slope limiters in CFD.
        Parameters:
            a              - left slope. 
            b              - right slope. 
            limiter        - limiters, minmod, vanLeer, etc.
        Rerurns:
            slope          - limited slope.
        Raises:
            KeyError.
        '''
        import numpy as np
        slope = 0.

        # select limiters
        if (limiter == 'minmod'):
            slope = 0.5*(np.sign(a) + np.sign(b)) * np.min([np.abs(a), np.abs(b)])
        else:
            pass

        return slope

    #*********************************************************************#
    def first_derivate(self, array, dx=1.,  order=4, limiter='minmod'):
        '''
        This function is used to calculate numerical first derivate of array based on order.
        Parameters:
            array          - array to be derivated.
            dx             - x step.
            order          - derivation order.
            limiter        - slope limiters, minmod, vanLeer, etc..
        Rerurns:
            array_d        - first numerical derivation of array.
        Raises:
            KeyError.
        '''
        import numpy as np
        n = len(array)
        array_d = np.zeros(n, np.float)

        # loop
        if (order == 4):
            for i in range(2, n-2):
                array_d[i] = ((array[i+1] - array[i-1])*8 - (array[i+2] - array[i-2]))/12./dx
        elif (order == 2):
            for i in range(3, n-3):
                if (limiter == 'none'):
                    array_d[i] = (array[i+1] - array[i-1])/2./dx
                else:
                    pl =  (3*array[i] - 4*array[i-1] + array[i-2])/2.
                    pr = -(3*array[i] - 4*array[i+1] + array[i+2])/2.
                    array_d[i] = cfd_class.slope_limiter(self, a=pl, b=pr, limiter=limiter)
                    array_d[i] = array_d[i]/dx
        elif (order == 3):
            for i in range(2, n-2):
                if (limiter == 'none'):
                    pass
                else:
                    pl = (11*array[i] -18*array[i-1] + 9*array[i-2] - 2*array[i-3])/6.
                    pr = (2*array[i+1] + 3*array[i] - 6*array[i-1] + array[i-2])/6.
                    array_d[i] = cfd_class.slope_limiter(self, a=pl, b=pr, limiter=limiter)
                    array_d[i] = array_d[i]/dx
        elif (order == 1):
            for i in range(2, n-2):
                if (limiter == 'none'):
                    array_d[i] = (array[i] - array[i-1])/dx
                else:
                    array_d[i] = cfd_class.slope_limiter(self, a=(array[i]-array[i-1]), b=(array[i+1]-array[i]), limiter=limiter)
                    array_d[i] = array_d[i]/dx
        else:
            pass

        return array_d

    #*********************************************************************#
    def derivate(self, array=[], delta=1., direction='x', order=1):
        '''
        This function is used to calculate an array's derivation.
        Parameters:
            array          - array to be derivated.
            delta          - dx.
            direction      - derivation direction.
            order          - derivation order.
        Rerurns:
            array_d        - derivation..
        Raises:
            KeyError.
        '''
        import numpy as np

        shape = array.shape
        m = shape[0]
        n = shape[1]
        array_d = np.zeros(shape, np.float)

        #select order
        if (order == 1):
            # select direction
            if (direction == 'x'):
                # compute inner box
                for i in range(0, m):
                    for j in range(1, n-1):
                        array_d[i, j] = 0.5*(array[i, j+1] - array[i, j-1])/delta
                    # compute edge
                    # j = 0
                    array_d[i, 0] = - 0.5*(3*array[i, 0] - 4*array[i, 1] + array[i, 2])/delta
                    # j = n-1
                    array_d[i, n-1] = 0.5*(3*array[i, n-1] - 4*array[i, n-2] + array[i, n-3])/delta
            elif (direction == 'y'):
                for j in range(0, n):
                    for i in range(1, m-1):
                        array_d[i, j] = 0.5*(array[i+1, j] - array[i-1, j])/delta
                    # compute edge
                    # i = 0
                    array_d[0, j] = - 0.5*(3*array[0, j] - 4*array[1, j] + array[2, j])/delta
                    # i = m-1
                    array_d[m-1, j] = 0.5*(3*array[m-1, j] - 4*array[m-2, j] + array[m-3, j])/delta
            else:
                pass
        elif (order == 2):
            pass
        else:
            print("No match derivation order!")

        return array_d
    #*********************************************************************#
    def weno_5(self, array, returns='omega'):
        '''
        Five order WENO data reconstruction.
        Parameters:
            array          - array to be reconstructed.
        Rerurns:
            array_r        - array after reconstructed.
            omega          - stencil coefficients.
            returns        - return.
        Raises:
            KeyError.
        '''
        import numpy as np
        n = len(array)
        array_r = np.zeros(n, np.float)
        # coefficients of ENO
        a = [[1./3., -7./6., 11./6.], \
             [-1./6., 5./6., 1./3. ], \
             [1./3., 5./6., -1./6. ]]
        # coefficients of WENO
        c = [0.1, 0.6, 0.3]

        # array to restore coefficients
        epsilon = 1.0e-6
        q = [0, 0, 0]
        s = [0, 0, 0]
        alpha = [0, 0, 0]
        omega = np.zeros([n, 3], np.float)
        # loop of reconstruction
        for i in range(2, n-2):
            # stencil 0
            q[0] = a[0][0]*array[i-2] + a[0][1]*array[i-1] + a[0][2]*array[i]
            s[0] = (13./12.)*(array[i-2] - 2*array[i-1] + array[i])**2 + (1./4.)*(array[i-2] - 4*array[i-1] + 3*array[i])**2
            alpha[0] = c[0]/(epsilon + s[0])**2
            # stencil 1
            q[1] = a[1][0]*array[i-1] + a[1][1]*array[i] + a[1][2]*array[i+1]
            s[1] = (13./12.)*(array[i-1] - 2*array[i] + array[i+1])**2 + (1./4.)*(array[i-1] - array[i+1])**2
            alpha[1] = c[1]/(epsilon + s[1])**2
            # stencil 2
            q[2] = a[2][0]*array[i] + a[2][1]*array[i+1] + a[2][2]*array[i+2]
            s[2] = (13./12.)*(array[i] - 2*array[i+1] + array[i+2])**2 + (1./4.)*(3*array[i] - 4*array[i+1] + array[i+2])**2
            alpha[2] = c[2]/(epsilon + s[2])**2

            # WENO coefficient
            sum_a = sum(alpha)
            omega[i][0] = alpha[0]/sum_a
            omega[i][1] = alpha[1]/sum_a
            omega[i][2] = alpha[2]/sum_a
            
            # reconstruction
            array_r[i] = omega[i][0]*q[0] + omega[i][1]*q[1] + omega[i][2]*q[2]
        if (returns == 'array_r'):
            return array_r
        else:
            return omega
