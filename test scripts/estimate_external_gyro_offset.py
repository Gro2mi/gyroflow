import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import stabilizer
import time
from scipy.signal import savgol_filter
import gyrolog

video = r'D:\Cloud\git\FPV\videos\GH011144.MP4'
gyro = r'D:\Cloud\git\FPV\videos\GH011144.MP4'
cam_preset = 'camera_presets/GoPro/GoPro_Hero6_2160p_43.json'

# video = r'D:\Cloud\git\FPV\videos\external_gyro_-15offset\P1077044.MP4'
# gyro = r'D:\Cloud\git\FPV\videos\external_gyro_-15offset\LOG00001.BFL.csv'
# cam_preset = r'D:\Cloud\git\FPV\videos\external_gyro_-15offset\Gh5+8mm-108025fps.json'
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
for i in tqdm(range(frame_count), desc=f"Analyzing frames", colour="blue"):
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (100, 75))
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # save frametime and relative area of movement of the frame
        df.loc[len(df)] = [cap.get(cv2.CAP_PROP_POS_MSEC)/1000, len(np.where(fgmask > 0)[0])/fgmask.size]

# save data for testing and comparing
df.to_csv("video_movement2.csv")

cap.release()
df = pd.read_csv("video_movement.csv")

df["movement_rolling_mean"] = df.movement.rolling(fps, min_periods=1).mean()

df["movement_rolling_std"] = df.movement.rolling(fps, min_periods=1).std()

df["savgol"] = savgol_filter(df.movement, window_length =31, polyorder = 1)
df["savgol_diff"] = savgol_filter(df.movement.diff(), window_length =151, polyorder = 1)
df["savgol_diff"] = df.movement.diff().abs().rolling(100).sum()

df["mmm"] = df.movement.rolling(2*fps, min_periods=1).apply(lambda x: np.sum(x < 0.001) < (0.8 * 2*fps))

df["moving"] = df.savgol > .1

# print(df.movement.mean()/4)
print(df[df.moving.diff() == 1])
fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
ax.plot(df.time[:-fps], df.movement[:-fps], label="Movement Detection")
# ax.plot(df.time, df.movement_rolling_mean, label="Movement Detection Rolling 15")
ax.plot(df.time[:-fps], df.mmm[:-fps], label="Movement")
# ax.plot(df.time, df.movement_rolling_mean)
# # ax.plot(df.time[:-50], df.savgol_std[:-50])
# ax.plot(df.time[:-50], df.savgol_diff[:-50])
ax.set(title="Video Movement Analysis", xlabel="time [s]", ylabel="relative area of detected movment [-]")
plt.legend()
plt.show()
td = time.time() - t
print(df.describe())
print(f"Time: {td}")

# comparison to gyro data
guessed_log, logtype, logvariant = gyrolog.guess_log_type_from_video(gyro)
stab = stabilizer.MultiStabilizer(video, cam_preset, gyro, logtype=logtype, logvariant=logvariant)
stab.gyro_analysis(debug_plots=True)
# stab.full_auto_sync_parallel()
