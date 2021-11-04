import cv2
import numpy as np
import multiprocessing as mp
import time
from calibrate_video import FisheyeCalibrator
import gyrolog
from stabilizer import MultiStabilizer, fast_gyro_cost_func, better_gyro_cost_func, optical_flow, estimate_gyro_offset, ParallelSync


from tqdm import tqdm
from scipy.spatial.transform import Rotation
from scipy import signal, interpolate
import matplotlib.pyplot as plt


if __name__ == "__main__":

    infile_path = r"S:\Cloud\git\FPV\videos\GH011145.MP4"
    log_guess, log_type, variant = gyrolog.guess_log_type_from_video(infile_path)
    if not log_guess:
        print("Can't guess log")
        exit()

    stab = MultiStabilizer(infile_path, r"S:\Cloud\git\FPV\GoPro_Hero6_2160p_43.json", log_guess, gyro_lpf_cutoff = -1, logtype=log_type, logvariant=variant)
    stab.sync_points = stab.get_recommended_syncpoints(30)
    start = time.time()
    ps = ParallelSync(stab, 30)
    ps.begin_sync_parallel()
    print(f"time needed for parallel auto sync: {time.time() - start:.2f} s")