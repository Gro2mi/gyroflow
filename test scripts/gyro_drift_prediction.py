import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('S:\Cloud\git\mjr_dark.mplstyle')
df = pd.read_csv("gyro_drift_offset.csv")
df = df.drop_duplicates(subset=['videopath'])
df = df[df.cam_preset == "camera_presets/GoPro/GoPro_Hero6_2160p_43.json"][:-4]

coeff = np.polyfit(df.sync_drift, df.calculated_drift, 1)
yn = np.polyval(coeff, df.sync_drift)
y1 = np.polyval([0, coeff[1]], df.sync_drift)
print(coeff)
fig, ax_drift_estimation = plt.subplots()
ax_drift_estimation.plot(df.sync_drift, yn, color='#348ABD')
ax_drift_estimation.scatter(df.sync_drift, df.calculated_drift)
# ax_drift_estimation.scatter(df.sync_drift[-4:], df.calculated_drift[-4:], color="green")
ax_drift_estimation.set(title="gyro drift estimation with gyro sample rate",
                        xlabel="gyro drift",
                        ylabel="calculated drift",
                        )
ax_drift_estimation.text(x=.05, y=.5, s=f"Cam: GoPro Hero 6\n"
                       f"Sample size: {len(df)}\n"
                       f"Slope: {coeff[0]:.4f}\n"
                       f"Offset: {coeff[1]:.4}\n"
                       f"Error: {((df.calculated_drift - df.sync_drift)**2).sum()}\n"
                       f"std corrected: {((df.calculated_drift - yn)).std()}", transform=ax_drift_estimation.transAxes)
plt.grid(True)
plt.show()

fig, ax_offset_estimation = plt.subplots()
ax_offset_estimation.scatter(df.sync_drift, df.sync_offset)
# ax_offset_estimation.scatter(df.sync_drift[-4:], df.sync_offset[-4:], color="green")
ax_offset_estimation.axhline(y=df.sync_offset.mean(), linestyle='-', color='#348ABD')
ax_offset_estimation.set(title=f"gyro offset hero 6 (mean: {df.sync_offset.mean():.4f})",
                         ylabel="offset [s]",
                         xlabel="gyro drift")
plt.grid(True)
plt.show()