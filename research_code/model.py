import numpy as np
from .parameter import BoucWenParameter
from .system import BoucWenSystem


class NSBB():
    def __init__(self):
        self.dim = 9
        self.mass = [34.5,33.85,33.85,33.85,33.85,33.85,33.85,33.85,36.6]
        self.mod_damping = [0]*self.dim
        self.bw_index = [0,1,10,18,25,31,36,40,43]
        self.stiff = [
            16506.9527976924, 
            38654.9298643029, 
            -9471.65127166421, 
            1625.79673225329, 
            -272.153425541571, 
            48.3336199564893,
            -15.0666880979758, 
            2.61893130662487, 
            -7.97988686471261, 
            -5676.74818401395, 
            40843.0407796389,
            -10209.3591874521, 
            1577.79031358035, 
            -276.177020638834, 
            49.7906083901859,
            -11.0203891457645, 
            0.00539080467104286, 
            931.536405270496, 
            36099.4067535433, 
            -7897.83189138013,
            1264.31209247754, 
            -210.238899524985, 
            43.3743336557548,
            -8.44222322859639,
            -170.682661332262, 
            30981.51211265, 
            -6945.0570289642, 
            1006.24585735839, 
            -194.346220093359,
            25.301473089978, 
            30.4094846568511, 
            26879.2163957677, 
            -5323.38942694299, 
            961.741052210119,
            -139.195894814186, 
            -7.81571030749706, 
            23092.2691476373, 
            -5939.84080931259, 
            805.942954312645,
            -0.392434915720514, 
            20736.7100153771, 
            -4238.27594537016, 
            -0.945672172509148, 
            11217.2721327549, 
            -3.57127312190914
        ]

        self.delta_y = [0.145661, 0.073379, 0.088417, 0.119957, 0.079751, 0.137673, 0.111261, 0.106056, 0.140737]
        self.rho = [0.229556, 0.307698, 0.276846, 0.393603, 0.438462, 0.416562, 0.764021, 0.505961, 0.301157]
        self.nu_bw = [1.899878, 1.986417, 1.957681, 1.655571, 2.229184, 2.040811, 2.276190, 2.299513, 1.812283]
        self.alpha_bw = [0.590509, 0.800973, 0.687800, 0.817615, 0.780905, 0.630119, 0.760197, 0.781489, 0.673541]

        self.beta_bw = [self.rho[i] / (self.delta_y[i] ** self.nu_bw[i]) for i in range(9)]
        self.gamma_bw = [(1 - self.rho[i]) / (self.delta_y[i] ** self.nu_bw[i]) for i in range(9)]

        self.alpha = np.ones(45)
        self.n = np.ones(45)
        self.beta = np.zeros(45)
        self.gamma = np.zeros(45)
        self.A = np.zeros(45)
        for ii in range(9):
            self.A[self.bw_index[ii]] = 1.0
            self.alpha[self.bw_index[ii]] = self.alpha_bw[ii]
            self.n[self.bw_index[ii]] = self.nu_bw[ii]
            self.beta[self.bw_index[ii]] = self.beta_bw[ii]
            self.gamma[self.bw_index[ii]] = self.gamma_bw[ii]

        self.parameter = BoucWenParameter(A=self.A, alpha=self.alpha, n=self.n, beta=self.beta, gamma=self.gamma)

        self.comp_mat = np.zeros((45,9))

        index = 0
        for ii in range(9):
            for jj in range(9-ii):
                if jj == 0:
                    self.comp_mat[index,ii] = 1
                else:
                    self.comp_mat[index,ii] = -1
                    self.comp_mat[index,ii+jj] = 1
                index += 1

        self.system = BoucWenSystem(
            mass = self.mass,
            damping = self.mod_damping,
            stiff = self.stiff, 
            parameter = self.parameter,
            comp_mat = self.comp_mat,
            force_influence=-np.array(self.mass).reshape(-1,1)
            )
        a1 = 2*0.02/(self.system.mod_freq[-1]+self.system.mod_freq[-3])
        a0 = 2*0.02*self.system.mod_freq[-3]*self.system.mod_freq[-1]/(self.system.mod_freq[-1]+self.system.mod_freq[-3])
        damping = a0*self.system.mass + a1*self.system.stiff
        self.system = BoucWenSystem(
            mass = self.mass,
            damping = damping,
            stiff = self.stiff,
            parameter = self.parameter,
            comp_mat = self.comp_mat,
            force_influence=-np.array(self.mass).reshape(-1,1)
            )