import numpy as np
import scipy.io as sio
import os
import multiprocessing as mp
from datetime import datetime

def run_simulation(realizations_chunk, lambda0UAV, R_UAV, v_const, tVec, dt):
    HandoverTime = np.zeros(realizations_chunk)
    NumUAV_Initial = lambda0UAV * np.pi * R_UAV**2
    for i in range(realizations_chunk):
        NumUAV = np.random.poisson(NumUAV_Initial)
        if NumUAV == 0: continue
        # Constant velocity for all drones
        v = np.full((NumUAV, 1), v_const)
        DisplacedTheta = np.random.uniform(0, 2 * np.pi, (NumUAV, 1))
        PosUAV_Range = np.sqrt(np.random.uniform(0, 1, (NumUAV, 1))) * R_UAV
        PosUAV_Theta = np.random.uniform(0, 2 * np.pi, (NumUAV, 1))
        PosUAV = np.hstack((PosUAV_Range * np.cos(PosUAV_Theta), PosUAV_Range * np.sin(PosUAV_Theta)))
        IndMin = np.argmin(PosUAV_Range)
        for idx, t in enumerate(tVec):
            vd = v * t * np.hstack((np.cos(DisplacedTheta), np.sin(DisplacedTheta)))
            DisplacedPosUAV = PosUAV + vd
            NewRange = np.linalg.norm(DisplacedPosUAV, axis=1)
            if np.argmin(NewRange) != IndMin:
                HandoverTime[i] = t / dt
                break
    return HandoverTime

if __name__ == '__main__':
    v_const = 45 / 3.6
    lambda0UAV, R_UAV, dt, T = 1e-6, 1e5, 1, 300
    tVec = np.arange(dt, T + dt, dt)
    Realizations = int(1e6)
    num_cores = mp.cpu_count()
    chunk_size = Realizations // num_cores
    with mp.Pool(num_cores) as pool:
        results = pool.starmap(run_simulation, [(chunk_size, 1e-6, 1e5, v_const, tVec, 1)] * num_cores)
    Handover_Prob = np.zeros(len(tVec))
    HandoverTime_All = np.concatenate(results)
    for i in range(len(tVec)):
        Handover_Prob[i] = np.sum((HandoverTime_All <= (i + 1)) & (HandoverTime_All > 0)) / Realizations
    root_dir = os.path.dirname(os.getcwd()) # Gets the root folder
    output_folder = os.path.join(root_dir, 'Data', 'Mobile DBS')
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    sio.savemat(os.path.join(output_folder, 'Handover_ConstantVelocityDBS_Simulation.mat'), 
                {'Handover_ConstantDBS_Simulation': Handover_Prob})
    print(f"Constant Simulation Saved: {datetime.now()}")