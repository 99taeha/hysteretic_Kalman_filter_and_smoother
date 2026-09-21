import numpy as np

from scipy.linalg import eig, inv


class System():
    def __init__(self, mass, damping, stiff, comp_mat, force_influence=None):
        mass = np.atleast_2d(mass)  
        damping = np.atleast_2d(damping)
        self.comp_mat = np.atleast_2d(comp_mat)
        self.element_stiff = np.atleast_1d(stiff)
        self.stiff = np.atleast_2d(self.comp_mat.T @ np.diag(np.atleast_1d(stiff)) @ self.comp_mat)
        if mass.shape[0] == 1:
            self.mass = np.diag(mass[0])
        else:
            self.mass = mass

        eig_value, mod_mat = eig(self.stiff, self.mass)
        self.mod_mass = np.diag(mod_mat.T @ self.mass @ mod_mat)
        if not np.all(np.isreal(eig_value)):
            print('Complex modal stiffness!')
        self.mod_freq = np.real(np.sqrt(eig_value))
        self.mod_mat = mod_mat
        self.mod_mat_inv = inv(mod_mat)

        if damping.shape[0] == 1:
            self.mod_damping = damping[0]
            self.damping = self.mod_mat_inv.T @ np.diag(2*self.mod_damping*self.mod_freq*self.mod_mass) @ self.mod_mat_inv
        else:
            self.damping = damping
            self.mod_damping = np.diag(mod_mat.T @ damping @ mod_mat)/self.mod_mass/self.mod_freq/2
        self.damped_freq = self.mod_freq*np.sqrt(1-self.mod_damping**2)

        self.dim = self.mass.shape[0]
        if force_influence is None:
            self.force_influence = np.eye(self.dim)
        else:
            self.force_influence = np.atleast_2d(force_influence)
            if self.force_influence.shape[0] == 1:
                self.force_influence = self.force_influence.T
            
        self.element_dim = self.element_stiff.shape[0]
        self.force_dim = self.force_influence.shape[1]


class BilinearSystem(System):
    def __init__(self, mass, damping, stiff, parameter, comp_mat, force_influence=None):
        super().__init__(mass, damping, stiff, comp_mat, force_influence=force_influence)
        self.alpha = parameter.alpha
        self.fy = parameter.fy
        self.model = 'Bilinear'


class BoucWenSystem(System):
    def __init__(self, mass, damping, stiff, parameter, comp_mat, force_influence=None):
        super().__init__(mass, damping, stiff, comp_mat, force_influence=force_influence)
        self.alpha = parameter.alpha
        self.A = parameter.A
        self.beta = parameter.beta
        self.gamma = parameter.gamma
        self.n = parameter.n
        self.model = 'BoucWen'
