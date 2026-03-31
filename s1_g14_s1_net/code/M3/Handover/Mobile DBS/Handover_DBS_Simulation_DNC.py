import numpy as np
import scipy.io as sio
import os
import multiprocessing as mp
from datetime import datetime

def run_simulation(realizations_chunk, lambda0UAV, R_UAV, vStart, vEnd, tVec, dt):
    """
    Core simulation logic: Distributes drones and checks for the nearest 
    neighbor changes (handovers) over time.
    """
    HandoverTime = np.zeros(realizations_chunk)
    NumUAV_Initial = lambda0UAV * np.pi * R_UAV**2
    
    for i in range(realizations_chunk):
        # 1. Generate drones using Poisson distribution
        NumUAV = np.random.poisson(NumUAV_Initial)
        if NumUAV == 0:
            HandoverTime[i] = 0 
            continue
            
        # 2. Assign random Uniform speeds and directions
        v = np.random.uniform(vStart, vEnd, (NumUAV, 1))
        DisplacedTheta = np.random.uniform(0, 2 * np.pi, (NumUAV, 1))
        
        # 3. Initial Positions (Uniformly distributed in a circle)
        PosUAV_Range = np.sqrt(np.random.uniform(0, 1, (NumUAV, 1))) * R_UAV
        PosUAV_Theta = np.random.uniform(0, 2 * np.pi, (NumUAV, 1))
        PosUAV = np.hstack((PosUAV_Range * np.cos(PosUAV_Theta), 
                           PosUAV_Range * np.sin(PosUAV_Theta)))
        
        # 4. Find the initial serving DBS index
        IndMin = np.argmin(PosUAV_Range)
        
        # 5. Step through time to find the first handover event
        found_ho = False
        for idx, t in enumerate(tVec):
            # Update positions: x_new = x_old + v*t*cos(theta)
            vd = v * t * np.hstack((np.cos(DisplacedTheta), np.sin(DisplacedTheta)))
            DisplacedPosUAV = PosUAV + vd
            
            # Calculate distance to origin for all drones
            NewRange = np.linalg.norm(DisplacedPosUAV, axis=1)
            IndMin1 = np.argmin(NewRange)
            
            # If the closest drone ID changes, a handover occurred
            if IndMin1 != IndMin:
                HandoverTime[i] = t / dt
                found_ho = True
                break
                
    return HandoverTime

if __name__ == '__main__':
    print(f"Simulation Start Time: {datetime.now()}")
    
    # --- Configuration Parameters ---
    vMean_kmh = 45
    vDiff_kmh = 80  # This determines the filename (e.g., Uniform80)
    
    lambda0UAV = 1e-6
    R_UAV = 1e5
    dt, T = 1, 300
    tVec = np.arange(dt, T + dt, dt)
    Realizations = int(1e6) # 1 Million for paper-grade results
    
    # --- Unit Conversions ---
    vMean = vMean_kmh / 3.6
    vDiff = vDiff_kmh / 3.6
    vStart, vEnd = vMean - vDiff/2, vMean + vDiff/2
    
    # --- Parallel Execution ---
    num_cores = mp.cpu_count()
    chunk_size = Realizations // num_cores
    
    print(f"Running {Realizations} realizations on {num_cores} CPU cores...")
    with mp.Pool(num_cores) as pool:
        results = pool.starmap(run_simulation, 
                               [(chunk_size, lambda0UAV, R_UAV, vStart, vEnd, tVec, dt)] * num_cores)
    
    # --- Data Processing ---
    HandoverTime_All = np.concatenate(results)
    Handover_Prob_Vector = np.zeros(len(tVec))
    
    for i in range(len(tVec)):
        # Calculate P(Handover <= t)
        Handover_Prob_Vector[i] = np.sum((HandoverTime_All <= (i + 1)) & (HandoverTime_All > 0)) / Realizations

    # --- Saving to Data Folder ---
    # Construct the path: ./Data/Mobile DBS/
    output_folder = os.path.join('Data', 'Mobile DBS')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # Generate filename and internal variable name dynamically
    file_name = f'Handover_Uniform{vDiff_kmh}VelocityDBS_Simulation.mat'
    var_name = f'Handover_Uniform{vDiff_kmh}DBS_Simulation'
    file_path = os.path.join(output_folder, file_name)

    # Save dictionary with f-string key to ensure variable name is correct in MATLAB
    sio.savemat(file_path, {var_name: Handover_Prob_Vector})
    
    print(f"Data successfully saved to: {file_path}")
    print(f"Simulation End Time: {datetime.now()}")