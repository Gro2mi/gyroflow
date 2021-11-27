# install opencv-contrib-python
## start
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import stabilizer
import time
import gyrolog
import scipy
import os

stylesheet = 'S:\Cloud\git\mjr_dark.mplstyle'
if os.path.isfile(stylesheet):
    plt.style.use(stylesheet)

def get_max_correlation(original, match):
    z = scipy.signal.fftconvolve(original, match[::-1])
    lags = np.arange(z.size) - (match.size - 1)
    return lags[np.argmax(np.abs(z))]

video = r'S:\Cloud\git\FPV\videos\GH011144.MP4'
gyro = r'S:\Cloud\git\FPV\videos\GH011144.MP4'
cam_preset = 'S:\Cloud\git\gyroflow\camera_presets\GoPro\GoPro_Hero6_2160p_43.json'
gyro_file = "gyro_movement.csv"
video_file = "video_movement.csv"

video = r'S:\Cloud\git\FPV\videos\external_gyro_-15offset\P1077044.MP4'
gyro = r'S:\Cloud\git\FPV\videos\external_gyro_-15offset\LOG00001.BFL.csv'
cam_preset = r'S:\Cloud\git\FPV\videos\external_gyro_-15offset\Gh5+8mm-108025fps.json'
gyro_file = "gyro_movement2.csv"
video_file = "video_movement2.csv"
#
video = r'S:\Cloud\git\FPV\videos\GH011142.MP4'
gyro = r'S:\Cloud\git\FPV\videos\GH011142.MP4'
cam_preset = 'S:\Cloud\git\gyroflow\camera_presets\GoPro\GoPro_Hero6_2160p_43.json'
gyro_file = "gyro_movement3.csv"
video_file = "video_movement3.csv"

# video = r'S:\Cloud\git\FPV\videos\GH011161.MP4'
# gyro = r'S:\Cloud\git\FPV\videos\GH011161.MP4'
# cam_preset = 'S:\Cloud\git\gyroflow\camera_presets\GoPro\GoPro_Hero6_2160p_43.json'
# gyro_file = "gyro_movement4.csv"
# video_file = "video_movement4.csv"
#
# video = r'S:\Cloud\git\FPV\videos\GH011162.MP4'
# gyro = r'S:\Cloud\git\FPV\videos\GH011162.MP4'
# cam_preset = 'S:\Cloud\git\gyroflow\camera_presets\GoPro\GoPro_Hero6_2160p_43.json'
# gyro_file = "gyro_movement5.csv"
# video_file = "video_movement5.csv"

video = r"S:\Cloud\git\FPV\videos\tarsier\LOOP0496.MP4"
gyro = r"S:\Cloud\git\FPV\videos\tarsier\00000120.bin.csv"
gyro_file = "gyro_movement6.csv"
video_file = "video_movement6.csv"

pd.set_option('display.max_columns', 10)
pd.set_option("expand_frame_repr", True)
cap = cv2.VideoCapture(video)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# initializing subtractor
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

moving = []
frames = []
df = pd.DataFrame(columns=["time", "movement"])
fps = 30
t = time.time()
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = round(cap.get(cv2.CAP_PROP_FPS))

## Aquiring data; must only run once
guessed_log, logtype, logvariant = gyrolog.guess_log_type_from_log(gyro)
stab = stabilizer.MultiStabilizer(video, cam_preset, gyro, logtype=logtype, logvariant=logvariant)
df_gyro = stab.gyro_analysis(debug_plots=True)
df_gyro.to_csv(gyro_file)

for i in tqdm(range(frame_count), desc=f"Analyzing frames", colour="blue"):
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (100, 75))
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # save frametime and relative area of movement of the frame
        df.loc[len(df)] = [cap.get(cv2.CAP_PROP_POS_MSEC)/1000, len(np.where(fgmask > 0)[0])/fgmask.size]

# save data for testing and comparing
td = time.time() - t
print(f"Time for video movement detection: {td}")
df.to_csv(video_file)

## load and analyze video data
cap.release()
df = pd.read_csv(video_file)
df = df[df.time.diff() > 0]
df["moving"] = df.movement.rolling(2*fps, min_periods=1, center=True).apply(lambda x: np.sum(x < 0.001) < min(len(x), (0.8 * 2*fps)))
# print(df[df.moving.diff() !=0])

## plotting video movement
fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
ax.plot(df.time, df.movement, label="Movement")
ax.plot(df.time, df.moving, label="Moving")
ax.set(title="Video Movement Analysis", xlabel="time [s]", ylabel="relative area of detected movment [-]")
plt.legend()
plt.show()

## gyro analysis
df_gyro = pd.read_csv(gyro_file)
gyro_offset = df_gyro.time.iloc[0]
df_gyro.time = df_gyro.time - gyro_offset

sampling_rate = len(df_gyro) / (df_gyro.time.iloc[-1] - df_gyro.time.iloc[0])
window_size = round(sampling_rate) * 2
df_gyro['moving'] = df_gyro.total.rolling(window_size).median() > 0.02
# print(df_gyro[df_gyro.moving.diff() != 0])

## offset calculation
f_movement = scipy.interpolate.interp1d(df.time, df.movement, fill_value="extrapolate")
f_moving = scipy.interpolate.interp1d(df.time, df.moving, fill_value="extrapolate")
video_movement_interpolated = f_movement(df_gyro.time)
video_moving_interpolated = f_moving(df_gyro.time)

idx = get_max_correlation(df_gyro.total / df_gyro.total.mean(), video_movement_interpolated / video_movement_interpolated.mean())
print(f"Estimated offset: {df_gyro.time.iloc[idx]} s")

## plot
fig, ax = plt.subplots()
# ax.plot(df_gyro.time, df_gyro.total)
ax.plot(df_gyro.time, df_gyro.total.rolling(window_size).median(), label="gyro movement")
ax.plot(df_gyro.time, df_gyro.moving, label="gyro moving")
ax.plot(df_gyro.time + df_gyro.time.iloc[idx], video_movement_interpolated, label="video movement")
ax.plot(df_gyro.time + df_gyro.time.iloc[idx], video_moving_interpolated, label="video moving")
ax.set(xlabel="time [s]", ylabel="movement", title=f"offset estimation video to gyro: {df_gyro.time.iloc[idx]:.2f} s")
plt.legend()
plt.tight_layout()
plt.show()
print(f"Total time: {time.time() - t}")
