import numpy as np

from scipy.linalg import inv, expm, block_diag
from multiprocessing import Pool

from .force import Force
from .response import Response



class StateSpaceModel():
    def set_covs(self, cov_model, cov_force, cov_measurement):
        self.set_model_cov(cov_model)
        self.set_force_cov(cov_force)
        self.set_measurement_cov(cov_measurement)
        self.cov_augmented = block_diag(self.cov_model, self.cov_force)

    def set_model_cov(self, cov_model):
        self.cov_model = np.atleast_2d(cov_model)
        if self.cov_model.shape[0] == 1:
            self.cov_model = self.cov_model*np.eye(self.state_size-self.system.force_dim)
        try:
            self.cov_augmented = block_diag(self.cov_model, self.cov_force)
        except:
            pass

    def set_force_cov(self, cov_force):
        self.cov_force = np.atleast_2d(cov_force)
        if self.cov_force.shape[0] == 1:
            self.cov_force = self.cov_force*np.eye(self.system.force_dim)
        try:
            self.cov_augmented = block_diag(self.cov_model, self.cov_force)
        except:
            pass

    def set_measurement_cov(self, cov_measurement):
        self.cov_measurement = np.atleast_2d(cov_measurement).copy()
        if self.cov_measurement.shape[0] == 1:
            self.cov_measurement = self.cov_measurement*np.eye(self.measurement_size)

    def gen_result(self, state):
        response_est = Response(self.system.dim, self.time.length, self.time)
        force_est = Force.assign_by_value(self.time.value, state[-self.system.force_dim:,:])
        response_est.disp = state[0:self.system.dim,:]
        response_est.aux = state[self.system.dim:self.system.dim+self.system.element_dim,:]
        response_est.vel = state[self.system.dim+self.system.element_dim:2*self.system.dim+self.system.element_dim,:]
        Ac31 = np.atleast_2d(-inv(self.system.mass)@self.system.comp_mat.T@np.diag(self.system.alpha*self.system.element_stiff)@self.system.comp_mat)
        Ac32 = np.atleast_2d(-inv(self.system.mass)@self.system.comp_mat.T@np.diag((1-self.system.alpha)*self.system.element_stiff))
        Ac33 = np.atleast_2d(-inv(self.system.mass)@self.system.damping)
        response_est.acc = Ac31@response_est.disp + Ac32@response_est.aux + Ac33@response_est.vel+ inv(self.system.mass)@self.system.force_influence@force_est.value
        return response_est, force_est

    def kfrts(self, measurement, initial_state_cov, initial_state = None):
        response_kf, force_kf = self.kalmanFilter(measurement, initial_state_cov, initial_state)
        response_rts, force_rts = self.rtsSmoother()
        return response_kf, force_kf, response_rts, force_rts

    def _kfrts_error(self, measurement, initial_state_cov, force_cov, initial_state = None):
        measurement = np.atleast_2d(measurement)
        self.set_force_cov(force_cov)
        response_kf, force_kf, response_rts, force_rts = self.kfrts(measurement, initial_state_cov, initial_state)

        smoothing_kf = np.sum(np.linalg.norm(force_kf.value, axis=0)**2)
        smoothing_rts = np.sum(np.linalg.norm(force_rts.value, axis=0)**2)
        if self.mtype == 'acc':
            error_kf = np.sum(np.linalg.norm(measurement[0:self.system.dim,:]-response_kf.acc)**2)
            error_rts = np.sum(np.linalg.norm(measurement[0:self.system.dim,:]-response_rts.acc)**2)
        elif self.mtype == '3dof':
            error_kf = np.sum(np.linalg.norm(measurement[0:self.system.dim,:]-response_kf.acc)**2) + np.sum(np.linalg.norm(measurement[0,:]-response_kf.disp[0,:])**2)
            error_rts = np.sum(np.linalg.norm(measurement[0:self.system.dim,:]-response_rts.acc)**2) + np.sum(np.linalg.norm(measurement[0,:]-response_rts.disp[0,:])**2)
        return error_kf,smoothing_kf,error_rts,smoothing_rts
    
    def draw_Lcurve(self, measurement, initial_state_cov, force_cov_list, initial_state = None, num_cores = 1):
        pool = Pool(num_cores)
        results = pool.starmap(self._kfrts_error, [(measurement, initial_state_cov, 10**force_cov, initial_state) for force_cov in force_cov_list])
        pool.close()
        pool.join()
        results = np.array(results)
        return results


class HysteresisSSM(StateSpaceModel):
    def __init__(self, system, time, mtype='acc'):
        self.system = system
        self.time = time
        self.mtype = mtype

        self.state_size = self.system.dim*2+self.system.element_dim+self.system.force_dim
        self.measurement_size = self.system.dim
        if mtype == '3dof':
            self.measurement_size = self.system.dim+1
        
        self.zeros = np.zeros((system.dim,system.dim))
        self.zeros_LG = np.zeros((system.element_dim, system.dim))
        self.zeros_LL = np.zeros((system.element_dim, system.element_dim))
        self.eyes = np.eye(system.dim)
        self.eyes_LL = np.eye(system.element_dim)
        self.zeros_FG = np.zeros((system.force_dim, system.dim))
        self.zeros_FL = np.zeros((system.force_dim, system.element_dim))
        self.zeros_FF = np.zeros((system.force_dim, system.force_dim))

        Ac31 = np.atleast_2d(-inv(system.mass)@system.comp_mat.T@np.diag(system.alpha*system.element_stiff)@system.comp_mat)
        Ac32 = np.atleast_2d(-inv(system.mass)@system.comp_mat.T@np.diag((1-system.alpha)*system.element_stiff))
        Ac33 = np.atleast_2d(-inv(system.mass)@system.damping)

        self.Ac = np.block([[self.zeros, self.zeros_LG.T, self.eyes],
                            [self.zeros_LG, self.zeros_LL, self.system.comp_mat],
                            [Ac31, Ac32, Ac33]])
        self.Bc = np.block([[self.zeros_FG.T],
                            [self.zeros_FL.T],
                            [inv(system.mass)@ system.force_influence]])

        self.Aac = np.block([[self.Ac, self.Bc],[self.zeros_FG, self.zeros_FL, self.zeros_FG, self.zeros_FF]])
        self.Fac = np.zeros(self.state_size)

        self.Cd = self._construct_Cc()
        if mtype == 'acc':
            self.Dd = inv(system.mass) @ system.force_influence
        elif mtype == '3dof':
            last_disp = np.zeros((1,self.system.force_dim))
            self.Dd = np.vstack((inv(system.mass) @ system.force_influence, last_disp))
        else:   
            raise ValueError("mtype is not defined")

        self.Aa = expm(self.Aac*time.interval)
        self.Fa = np.zeros(self.state_size)
        self.Ga = np.hstack((self.Cd, self.Dd))
        self.Aa_list = np.zeros((self.state_size,self.state_size,self.time.length))
        self.Aac_list = np.zeros((self.state_size,self.state_size,self.time.length))
        self.Fa_list = np.zeros((self.state_size,self.time.length))
        self.Aa_list[:,:,0] = self.Aa
        self.Fa_list[:,0] = self.Fa
        
    def _update_SSM(self,state):
        element_vel = np.atleast_1d(self.system.comp_mat@state[self.system.dim+self.system.element_dim:2*self.system.dim+self.system.element_dim])
        aux = np.atleast_1d(state[self.system.dim:self.system.dim+self.system.element_dim])

        if self.system.model == 'Bilinear':
            Ac23 = []
            for ii in range(self.system.element_dim):
                if aux[ii] > self.system.fy[ii]/self.system.element_stiff[ii] and element_vel[ii]>0:
                    Ac23.append(0)
                elif aux[ii] < -self.system.fy[ii]/self.system.element_stiff[ii] and element_vel[ii]<0:
                    Ac23.append(0)
                else:
                    Ac23.append(1)
        elif self.system.model == 'BoucWen':
            Ac22 = -self.system.n*element_vel*abs(aux)**(self.system.n-1)*(self.system.gamma*np.sign(aux)+self.system.beta*np.sign(element_vel))
            Ac23 = self.system.A - abs(aux)**self.system.n*(self.system.gamma+self.system.beta*np.sign(aux*element_vel))
            self.Ac[self.system.dim:self.system.dim+self.system.element_dim,self.system.dim:self.system.dim+self.system.element_dim] = np.diag(Ac22)

            self.Fac = np.zeros(self.state_size)
            self.Fac[self.system.dim:self.system.dim+self.system.element_dim] = self.system.n*element_vel*abs(aux)**(self.system.n-1)*(self.system.gamma*np.sign(aux)+self.system.beta*np.sign(element_vel))*aux
        else:
            print('Hysteresis model is not defined')
        
        Ac23 = np.diag(Ac23) @ self.system.comp_mat
        self.Ac[self.system.dim:self.system.dim+self.system.element_dim,self.system.dim+self.system.element_dim:self.system.dim*2+self.system.element_dim] = Ac23
        self.Aac = np.block([[self.Ac, self.Bc],[self.zeros_FG, self.zeros_FL, self.zeros_FG, self.zeros_FF]])
        M = np.hstack((self.Aac, self.Fac.reshape(-1,1)))
        M = np.vstack((M, np.zeros((1,self.state_size+1))))
        eM = expm(M*self.time.interval)
        Aa = eM[0:self.state_size,0:self.state_size]
        Fa = eM[0:self.state_size,-1]
        return Aa, Fa, self.Aac
    
    def _time_update(self, state, cov):
        if self.system.model == 'Bilinear':
            state_tmp = self.Aa @ state
        else:
            state_tmp = self.Aa @ state + self.Fa
        cov_tmp = self.Aa @ cov @ (self.Aa.T) + self.cov_augmented
        return state_tmp, cov_tmp

    def _measurement_update(self, state_tmp, cov_tmp, y):
        kalman_gain = cov_tmp@(self.Ga.T)@inv(self.Ga@cov_tmp@(self.Ga.T)+self.cov_measurement)

        state = state_tmp + kalman_gain@(y-self.Ga@state_tmp)
        cov = (np.eye(self.state_size)-kalman_gain@self.Ga)@cov_tmp
        return state, cov

    def _partial_kf(self, state, state_cov, measurement, length):
        kf = np.zeros((self.state_size,length))
        kf_tmp = np.zeros((self.state_size,length))
        kf_cov = np.zeros((self.state_size, self.state_size, length))
        kf_cov_tmp = np.zeros((self.state_size, self.state_size, length))
        Aa_list = np.zeros((self.state_size,self.state_size,length))
        Fa_list = np.zeros((self.state_size,length))
        Aac_list = np.zeros((self.state_size,self.state_size,length))

        for index in range(1,length):
            self.Aa, self.Fa, self.Aac = self._update_SSM(state)
            Aa_list[:,:,index] = self.Aa
            Fa_list[:,index] = self.Fa
            Aac_list[:,:,index] = self.Aac
            state_tmp, kf_cov_tmp[:,:,index] = self._time_update(state, state_cov)
            kf_tmp[:,index] = state_tmp
            kf[:,index], kf_cov[:,:,index] = self._measurement_update(state_tmp, kf_cov_tmp[:,:,index], measurement[:,index])
            state, state_cov = kf[:,index], kf_cov[:,:,index]
        
        return kf, kf_cov, kf_tmp, kf_cov_tmp, Aa_list, Fa_list, Aac_list
    
    def kalmanFilter(self, measurement, initial_state_cov, initial_state = None):
        if initial_state is None:
            self.initial_state = np.zeros(self.state_size)
        else:
            self.initial_state = initial_state
        
        self.initial_state_cov = initial_state_cov

        self.kf, self.kf_cov, self.kf_tmp, self.kf_cov_tmp, self.Aa_list, self.Fa_list, self.Aac_list = self._partial_kf(self.initial_state, self.initial_state_cov, measurement, measurement.shape[1])

        response_est, force_est = self.gen_result(self.kf)

        return response_est, force_est

    def _partial_rts(self, kf, kf_cov, kf_cov_tmp, Aa_list, Fa_list, length):
        rts = np.zeros((self.state_size, length))
        rts_cov = np.zeros((self.state_size, self.state_size, length))

        rts[:,-1] = kf[:,-1]
        rts_cov[:,:,-1] = kf_cov[:,:,-1]
        for index in range(length-2, 0, -1):
            self.Aa = Aa_list[:,:,index+1]
            self.Fa = Fa_list[:,index+1]

            C = kf_cov[:,:,index]@self.Aa.T@inv(kf_cov_tmp[:,:,index+1])
            
            rts[:,index] = kf[:,index] + C@(rts[:,index+1]-self.Aa@kf[:,index]-self.Fa)
            rts_cov[:,:,index] = kf_cov[:,:,index] + C@(rts_cov[:,:,index+1]-kf_cov_tmp[:,:,index+1])@C.T
        
        return rts, rts_cov
    
    def rtsSmoother(self):
        self.rts, self.rts_cov = self._partial_rts(self.kf, self.kf_cov, self.kf_cov_tmp, self.Aa_list, self.Fa_list, self.time.length)

        response_est, force_est = self.gen_result(self.rts)

        return response_est, force_est

    def _construct_Cc(self):
        if self.mtype == 'acc':
            Cc1 = np.atleast_2d(-inv(self.system.mass)@self.system.comp_mat.T@np.diag(self.system.alpha*self.system.element_stiff)@self.system.comp_mat)
            Cc2 = np.atleast_2d(-inv(self.system.mass)@self.system.comp_mat.T@np.diag((1-self.system.alpha)*self.system.element_stiff))
            Cc3 = np.atleast_2d(-inv(self.system.mass)@self.system.damping)
            Cc = np.hstack((Cc1, Cc2, Cc3))
        elif self.mtype == '3dof':
            Cc1 = np.atleast_2d(-inv(self.system.mass)@self.system.comp_mat.T@np.diag(self.system.alpha*self.system.element_stiff)@self.system.comp_mat)
            Cc2 = np.atleast_2d(-inv(self.system.mass)@self.system.comp_mat.T@np.diag((1-self.system.alpha)*self.system.element_stiff))
            Cc3 = np.atleast_2d(-inv(self.system.mass)@self.system.damping)
            Cc = np.hstack((Cc1, Cc2, Cc3))
            Cc = np.vstack((Cc, np.zeros((1,self.system.dim*2+self.system.element_dim))))
            Cc[-1,0] = 1
        return Cc


class HysteresisDMSSM5(HysteresisSSM):
    def __init__(self, system, time, mtype='acc'):
        super().__init__(system, time, mtype=mtype)
        self.Ga = np.block([[self.Ga], [self.zeros_LG, self.eyes_LL,  self.zeros_LG, self.zeros_FL.T]])
        self.Ga = np.block([[self.Ga], [self.zeros_FG, self.zeros_FL, self.zeros_FG, np.eye(self.system.force_dim)]])
        self.measurement_size = self.system.dim+self.system.element_dim + self.system.force_dim
