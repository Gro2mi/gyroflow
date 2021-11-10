# script to get the orientation matrix for a new log type and/or variant.
# set the variant orientation matrix to [0] (identity) in gyrolog.py
# record a video with 90° rotation on a single axis and back for all 3 axis (roll, pitch, yaw)
# copy and paste the orientation matrix in gyrolog.py and run script again.
# if axis of gyro and optical flow don't match record a new video and try again.

import cv2
import numpy as np
import time
from tqdm import tqdm
from calibrate_video import FisheyeCalibrator
import scipy
from scipy.spatial.transform import Rotation


import pandas as pd
import os
import gyrolog
import matplotlib.pyplot as plt
from datetime import datetime
import stabilizer


def check_videofile(videofile):
    t_start = datetime.now()
    cap = cv2.VideoCapture(videofile)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad_frames = []
    frame_readable = []
    for i in tqdm(range(num_frames), f"Checking each frame of {videofile}"):
        ret, prev = cap.read()
        if not ret:
            bad_frames.append(i)
            frame_readable.append(False)
        else:
            frame_readable.append(True)
    cap.release()
    print(f"{len(bad_frames)} of {num_frames} ({(len(bad_frames)/num_frames) * 100:.2f}%) not readable)")
    fig, ax = plt.subplots()
    ax.plot(frame_readable)
    plt.show()
    print(f"Time to check videofile: {datetime.now() - t_start}")


def df_savgol(df, prefix, sampling_rate):
    divider = 3
    window_length = round(sampling_rate / divider)
    if window_length % 2 == 0:
        if window_length - fps / divider > 0:
            window_length -= 1
        else:
            window_length += 1
    for ax in ['x', 'y', 'z']:
        df[f"{prefix}_{ax}_savgol"] = scipy.signal.savgol_filter(df[f"{prefix}_{ax}"], window_length=window_length, polyorder=1)
    return df


def df_interpol(df, new_time, prefix, window_length):
    for ax in ['x', 'y', 'z']:
        df[f"{prefix}_{ax}_savgol"] = scipy.signal.savgol_filter(df[f"{prefix}_{ax}"], window_length=window_length, polyorder=1)
    return df


def plot(df_gyro, df_optical_flow):
    fig, axes = plt.subplots(3, 2)
    axes[0, 0].plot(df_gyro.time, df_gyro.omega_x, alpha=.3)
    axes[0, 0].plot(df_gyro.time, df_gyro.omega_x_savgol)
    axes[1, 0].plot(df_gyro.time, df_gyro.omega_y, alpha=.3)
    axes[1, 0].plot(df_gyro.time, df_gyro.omega_y_savgol)
    axes[2, 0].plot(df_gyro.time, df_gyro.omega_z, alpha=.3)
    axes[2, 0].plot(df_gyro.time, df_gyro.omega_z_savgol)
    axes[0, 0].set(title="Gyro", ylabel="omega_x [rad/s]")
    axes[1, 0].set(ylabel="omega_y [rad/s]")
    axes[2, 0].set(ylabel="omega_z [rad/s]", xlabel="time [s]")
    axes[0, 1].plot(df_optical_flow.time, df_optical_flow.omega_x, alpha=.3)
    axes[0, 1].plot(df_optical_flow.time, df_optical_flow.omega_x_savgol)
    axes[1, 1].plot(df_optical_flow.time, df_optical_flow.omega_y, alpha=.3)
    axes[1, 1].plot(df_optical_flow.time, df_optical_flow.omega_y_savgol)
    axes[2, 1].plot(df_optical_flow.time, df_optical_flow.omega_z, alpha=.3)
    axes[2, 1].plot(df_optical_flow.time, df_optical_flow.omega_z_savgol)
    axes[0, 1].set(title="Optical Flow")
    axes[2, 1].set(xlabel="time [s]")
    plt.show()


def guess_orientation_matrix(df_gyro):
    error_matrix = np.ones((3, 3))
    error_matrix_negative = np.ones((3, 3))
    rotation_matrix = np.zeros((3, 3), dtype='int8')
    axes = ['x', 'y', 'z']
    for ii in range(3):
        for kk in range(3):
            rms = sum((df_gyro[f"omega_{axes[ii]}_savgol"] - df_gyro[f'omega_{axes[kk]}_savgol_interpol_of'])**2)/len(df_gyro)
            error_matrix[ii][kk] = rms
            rms = sum((df_gyro[f"omega_{axes[ii]}_savgol"] + df_gyro[f'omega_{axes[kk]}_savgol_interpol_of'])**2)/len(df_gyro)
            error_matrix_negative[ii][kk] = rms

    for ii in range(3):
        if min(error_matrix[ii]) < min(error_matrix_negative[ii]):
            rotation_matrix[np.argmin(error_matrix[ii])][ii] = 1
        else:
            rotation_matrix[np.argmin(error_matrix_negative[ii])][ii] = -1

    print("Guessed orientation matrix. Copy & paste this to gyrolog.py -> your log type -> your log variant in self.variants")
    print(f'"your_variant": {[-1, rotation_matrix.tolist()]},')
    print("Then run the script again. Gyro and optical flow plots should be on the same axis now. otherwise record a new video.")


def get_optical_flow(video_file, lens_preset, transform_file):
    # check_videofile(video_file)
    if not os.path.isfile(transform_file):
        undistort = FisheyeCalibrator()
        undistort.load_calibration_json(lens_preset, True)
        df_optical_flow = stabilizer.optical_flow(video_file, undistort, analyze_length=-1)
        df_optical_flow.to_csv(transform_file)
    else:
        df_optical_flow = pd.read_csv(transform_file, index_col=0)
    return df_optical_flow


def get_gyro(video_file):
    log_guess, log_type, variant = gyrolog.guess_log_type_from_video(video_file)
    print(variant)
    log_reader = gyrolog.get_log_reader_by_name(log_type)
    log_reader.set_pre_filter(50)
    success = log_reader.extract_log(video_file)
    if success:
        gyro = log_reader.standard_gyro
    else:
        print("Failed to read gyro!")
    gyro_rate = log_reader.gyro_sample_rate
    df_gyro = pd.DataFrame()
    df_gyro['time'] = gyro[:, 0]
    df_gyro['omega_x'] = gyro[:, 1]
    df_gyro['omega_y'] = gyro[:, 2]
    df_gyro['omega_z'] = gyro[:, 3]
    return df_gyro, gyro_rate


if __name__ == "__main__":
    video_file = r"D:\Cloud\git\gyroflow\OneR_1inch_gyro_samples\OneR_1inch_gyro_samples\Orentation_display_BACK\PRO_VID_20211102_143001_00_051.mp4"
    # video_file = r"D:\Cloud\git\gyroflow\OneR_1inch_gyro_samples\OneR_1inch_gyro_samples\Orientation_display_FRONT\PRO_VID_20211102_143237_10_053.mp4"
    lens_preset = r"D:\Cloud\git\gyroflow\OneR_1inch_gyro_samples\Insta360_OneR_1inch_PRO_4K_30fps_16by9.json"
    video_file = r"D:\Cloud\git\gyroflow\IMU_calibration_GP6.MP4"
    lens_preset = r"D:\Cloud\git\gyroflow\camera_presets\GoPro\GoPro_Hero6_2160p_43.json"
    transform_file = video_file + ".transform.csv.old"
    df_optical_flow = get_optical_flow(video_file, lens_preset, transform_file)
    df_gyro, gyro_rate = get_gyro(video_file)


    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # filter for bad OF points
    df_optical_flow = df_optical_flow[df_optical_flow.omega_x.abs() < 5]
    # do some more magic filtering
    df_gyro = df_savgol(df_gyro, 'omega', sampling_rate=gyro_rate)
    df_optical_flow = df_savgol(df_optical_flow, 'omega', sampling_rate=fps)
    for ax in ['x', 'y', 'z']:
        interpol = scipy.interpolate.interp1d(df_optical_flow.time, df_optical_flow[f"omega_{ax}_savgol"],
                                              assume_sorted=True, kind='linear', fill_value="extrapolate")
        df_gyro[f'omega_{ax}_savgol_interpol_of'] = interpol(df_gyro.time)
    plot(df_gyro, df_optical_flow)
    guess_orientation_matrix(df_gyro)
