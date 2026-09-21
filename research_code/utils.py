import pickle, os
import numpy as np
from pathlib import Path

from scipy.linalg import inv

from .force import Force
from .time import Time
from .response import Response

project_root = Path(__file__).resolve().parents[1]

def slice_data(time, force, response, interval):
    index = int(interval/time.interval)
    if index == interval/time.interval:
        time_sliced = Time(time.start, time.end, interval)
        response_sliced = Response(response.dim, response.length//index+1)
        response_sliced.disp = response.disp[:,::index]
        response_sliced.vel = response.vel[:,::index]
        response_sliced.acc = response.acc[:,::index]
        try:
            response_sliced.aux = response.aux[:,::index]
            response_sliced.element_disp = response.element_disp[:,::index]
            response_sliced.resisting_force = response.resisting_force[:,::index]
        except:
            pass
        response_sliced.interval = interval
        response_sliced.time = time_sliced.value
        force_sliced = Force.assign_by_value(time_sliced.value, force.value[:,::index])
        return time_sliced, force_sliced, response_sliced
    else:
        os.error('The interval should be a multiple of the original interval')


def compute_response(
            system,
            force,
            time,
            ctype,
            output_name:str = 'dummy',
            ):
    output = Response(system.dim, time.length, time)
    force_value = system.force_influence @ force.value
    if ctype == 'Newmark':
        GAMMA = 1/2
        BETA = 1/4

        output.element_disp = np.zeros((system.element_dim,time.length))
        output.resisting_force = np.zeros((system.element_dim,time.length))
        output.aux = np.zeros((system.element_dim,time.length))

        A1 = 1/BETA/time.interval**2*system.mass + GAMMA/BETA/time.interval*system.damping
        A2 = 1/BETA/time.interval*system.mass + (GAMMA/BETA-1)*system.damping
        A3 = (1/2/BETA-1)*system.mass + system.damping*time.interval*(GAMMA/2/BETA-1)

        tangent_stiff_hat = system.stiff + A1
        tangent_stiff_hat_inv = inv(tangent_stiff_hat)

        MAX_ITER = 1000
        for ii in range(1,time.length):
            output.disp[:,ii] = output.disp[:,ii-1]
            output.element_disp[:,ii] = output.element_disp[:,ii-1]
            output.resisting_force[:,ii] = output.resisting_force[:,ii-1]
            output.aux[:,ii] = output.aux[:,ii-1]
            force_hat = force_value[:,ii] + A1@output.disp[:,ii-1] + A2@output.vel[:,ii-1] + A3@output.acc[:,ii-1]
            for jj in range(MAX_ITER):
                residual_force = force_hat - system.comp_mat.T @ output.resisting_force[:,ii] - A1 @ output.disp[:,ii]
                delta_disp = tangent_stiff_hat_inv @ residual_force
                output.disp[:,ii] += delta_disp
                output.element_disp[:,ii] = system.comp_mat @ output.disp[:,ii]
                delta_element_disp = output.element_disp[:,ii] - output.element_disp[:,ii-1]

                if np.all(residual_force**2 < 1e-20) or np.all(residual_force**2/np.maximum(np.min(np.abs(force_hat)),1e-10)**2 < 1e-20):
                    break

                if system.model == 'Bilinear':
                    resisting_force_tmp = output.resisting_force[:,ii-1] + system.element_stiff*delta_element_disp
                    sign = np.sign(resisting_force_tmp-system.alpha*system.element_stiff*output.element_disp[:,ii])
                    value = np.minimum(np.abs(resisting_force_tmp-system.alpha*system.element_stiff*output.element_disp[:,ii]), (1-system.alpha)*system.fy)
                    output.resisting_force[:,ii] = sign*value + system.alpha*system.element_stiff*output.element_disp[:,ii]

                    yield_check = np.abs(resisting_force_tmp-system.alpha*system.element_stiff*output.element_disp[:,ii]) < (1-system.alpha)*system.fy
                    output.aux[:,ii] = output.aux[:,ii-1] + yield_check*delta_element_disp

                elif system.model == 'BoucWen':
                    delta_aux = delta_element_disp*(system.A-abs(output.aux[:,ii])**system.n*(system.gamma+system.beta*np.sign(output.aux[:,ii]*delta_element_disp)))
                    output.aux[:,ii] = output.aux[:,ii-1] + delta_aux
                    output.resisting_force[:,ii] = system.alpha*system.element_stiff*output.element_disp[:,ii]+(1-system.alpha)*system.element_stiff*output.aux[:,ii]

            if jj == MAX_ITER -1:
                print('Newton-Raphson did not converge')                    

            output.vel[:,ii] = GAMMA/BETA/time.interval*(output.disp[:,ii]-output.disp[:,ii-1])+(1-GAMMA/BETA)*output.vel[:,ii-1]+time.interval*(1-GAMMA/2/BETA)*output.acc[:,ii-1]
            output.acc[:,ii] = 1/BETA/time.interval**2*(output.disp[:,ii]-output.disp[:,ii-1])-1/BETA/time.interval*output.vel[:,ii-1]-(1/2/BETA-1)*output.acc[:,ii-1]
                
    with open(os.path.join(project_root,'data/'+output_name+'.pkl'), 'wb') as f:
        pickle.dump(force, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(system, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(time, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

    return output


def add_noise(value, noise_level):
    noise = np.random.normal(0, noise_level, value.shape)
    output = value + noise
    return output
