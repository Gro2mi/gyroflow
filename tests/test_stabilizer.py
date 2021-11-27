import pytest
import sys
import os
import subprocess



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gyrolog
from stabilizer import MultiStabilizer, Sync


def create_stab(infile_path="tests/test_files/videos/gopro6_test.mp4", cam_preset=r"camera_presets\GoPro\GoPro_Hero6_2160p_43.json"):
    log_guess, log_type, variant = gyrolog.guess_log_type_from_video(infile_path)
    if not log_guess:
        print("Can't guess log")
        exit()
    return MultiStabilizer(infile_path, cam_preset, log_guess, gyro_lpf_cutoff = -1, logtype=log_type, logvariant=variant)


def test_export():
    stab = create_stab(infile_path = "tests/test_files/videos/gopro6_test.mp4")
    # stab.rough_sync_search_interval = 0
    stab.full_auto_sync(debug_plots=False)

    export_settings = [
        # outfile, vcodec, vprofile, filesize
        ("render_test_libx264.mp4", "libx264", "high"),
        ("render_test_prores_ks.mov", "prores_ks", "auto")
    ]

    try:
        subprocess.check_output('nvidia-smi')
        export_settings.append(("render_test_h264_nvenv.mp4", "h264_nvenv", "high"))
    except Exception:  # this command not being found can raise quite a few different errors depending on the configuration
        print('No Nvidia GPU in system!')

    for settings in export_settings:
        print(f"\n\nTest export: settings: {settings}")
        outpath = os.path.join("tests", "test_files", "export_result", settings[0])
        stab.renderfile(starttime=2, stoptime=3, outpath=outpath, audio=True, vcodec=settings[1], vprofile=settings[2])
        print(os.stat(outpath).st_size)
        assert os.stat(outpath).st_size > 10000

def test_recommended_sync_points():
    stab = create_stab()
    print(stab.get_recommended_syncpoints(30))


def test_add_sync_points():
    stab = create_stab()
    sync = Sync(stab, 60)
    with pytest.raises(TypeError):
        sync.add_sync_points(4)
    sync.add_sync_points([4])
    assert len(sync.sync_points) == 1
    sync.add_sync_points([1, 7])
    assert len(sync.sync_points) == 3
    assert sync.sync_points[0].start_frame == 1
    assert sync.sync_points[1].start_frame == 4
    assert sync.sync_points[2].start_frame == 7



if __name__ == "__main__":
    # test_export()
    # test_recommended_sync_points()
    test_add_sync_points()