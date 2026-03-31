import numpy as np
import scipy.io as sio
import os
import multiprocessing as mp
from datetime import datetime

def run_simulation(realizations_chunk, lambda0UAV, R_UAV, vSigma, tVec, dt):
    HandoverTime = np.zeros(realizations_chunk)
    NumUAV_Initial = lambda0UAV * np.pi * R_UAV**2
    for i in range(realizations_chunk):
        NumUAV = np.random.poisson(NumUAV_Initial)
        if NumUAV == 0: continue
        # Rayleigh distribution for speeds
        v = np.random.rayleigh(vSigma, (NumUAV, 1))
        DisplacedTheta = np.random.uniform(0, 2 * np.pi, (NumUAV, 1))
        PosUAV_Range = np.sqrt(np.random.uniform(0, 1, (NumUAV, 1))) * R_UAV
        PosUAV_Theta = np.random.uniform(0, 2 * np.pi, (NumUAV, 1))
        PosUAV = np.hstack((PosUAV_Range * np.cos(PosUAV_Theta), PosUAV_Range * np.sin(PosUAV_Theta)))
        IndMin = np.argmin(PosUAV_Range)
        for idx, t in enumerate(tVec):
            vd = v * t * np.hstack((np.cos(DisplacedTheta), np.sin(DisplacedTheta)))
            DisplacedPosUAV = PosUAV + vd
            if np.argmin(np.linalg.norm(DisplacedPosUAV, axis=1)) != IndMin:
                HandoverTime[i] = t / dt
                break
    return HandoverTime

if __name__ == '__main__':
    vMean = 45 / 3.6
    vSigma = vMean / np.sqrt(np.pi / 2)
    Realizations = int(1e5) 
    tVec = np.arange(1, 301, 1)
    num_cores = mp.cpu_count()
    with mp.Pool(num_cores) as pool:
        results = pool.starmap(run_simulation, [(Realizations // num_cores, 1e-6, 1e5, vSigma, tVec, 1)] * num_cores)
    Handover_Prob = np.zeros(300)
    HandoverTime_All = np.concatenate(results)
    for i in range(300):
        Handover_Prob[i] = np.sum((HandoverTime_All <= (i + 1)) & (HandoverTime_All > 0)) / Realizations
    root_dir = os.path.dirname(os.getcwd()) # Gets the root folder
    output_folder = os.path.join(root_dir, 'Data', 'Mobile DBS')
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    sio.savemat(os.path.join(output_folder, 'Handover_RayleighVelocityDBS_Simulation.mat'), 
                {'Handover_RayleighDBS_Simulation': Handover_Prob})
    print(f"Rayleigh Simulation Saved: {datetime.now()}")