import cv2
import numpy as np
import multiprocessing as mp
import time
from calibrate_video import FisheyeCalibrator
import gyrolog
from stabilizer import MultiStabilizer, fast_gyro_cost_func, better_gyro_cost_func, optical_flow, estimate_gyro_offset, Sync


from tqdm import tqdm
from scipy.spatial.transform import Rotation
from scipy import signal, interpolate
import matplotlib.pyplot as plt


if __name__ == "__main__":

    infile_path = r"D:\Cloud\git\FPV\videos\GH011145.MP4"
    infile_path = r"D:\Cloud\git\gyroflow\IMU_calibration_GP6.MP4"
    lens_preset = r"D:\Cloud\git\FPV\GoPro_Hero6_2160p_43.json"
    # infile_path = r"D:\Cloud\git\FPV\videos\GH9.MP4"
    # lens_preset = r"D:\Cloud\git\gyroflow\camera_presets\GoPro\GoPro_HERO9_Black_Wide_5K_16by9_Wide.json"
    log_guess, log_type, variant = gyrolog.guess_log_type_from_video(infile_path)
    if not log_guess:
        print("Can't guess log")
        exit()
    stab = MultiStabilizer(infile_path, lens_preset, log_guess, gyro_lpf_cutoff = -1, logtype=log_type, logvariant=variant)
    stab.sync_points = stab.get_recommended_syncpoints(30)
    start = time.time()
    sync = Sync(stab, 30)
    sync.add_sync_points([7 * 30])
    # sync.begin_sync_parallel()
    sync.sort_sync_points()
    sync.read_sync_points()
    sync.cost()
    print(f"time needed for parallel auto sync: {time.time() - start:.2f} s")
