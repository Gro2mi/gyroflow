import argparse
import gyroflow
import gyrolog
import logging
import sys
import stabilizer

# python .\gyroflow_cli.py "D:\git\FPV\videos\GH011173.MP4" "camera_presets\GoPro\GoPro_Hero6_2160p_43.json"

logging.basicConfig(filename='gyroflow_cli.log', level=logging.INFO)

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info("\nStarting Gyroflow CLI\n")

# Instantiate the parser
parser = argparse.ArgumentParser(prog="Gyroflow", description='Gyroflow - Gyro based video stabilizer')
parser.add_argument('video', type=str, help='Video input path')
parser.add_argument('lens', type=str, help='Lens calibration file path')
parser.add_argument('-l', '--log')
parser.add_argument('-fov', '--field_of_view', type=float, default=1.4)
parser.add_argument('-lpf', '--low_pass_filter', type=float, default=-1)
parser.add_argument('-mfe', '--max_fitting_error', type=float, default=0.02)
parser.add_argument('-s', '--start', type=float, default=0)
parser.add_argument('-e', '--end', type=float, default=5)
parser.add_argument('-o', '--outfile', type=str)

if __name__=='__main__':
    # Append additional arguments to sys.argv
    # TODO remove line that is for dev purposes only
    sys.argv = sys.argv + [r"D:\git\FPV\videos\GH011162.MP4", r"camera_presets\GoPro\GoPro_Hero6_2160p_43.json", '-s', '15', '-e', '20']
    # sys.argv = sys.argv + [r"GH011172.MP4", r"camera_presets\GoPro\GoPro_Hero6_2160p_43.json"]
    args = parser.parse_args()
    if args.outfile is None:
        '.'.join(args.video.split('.')[:-1]) + '_gyroflow.' + args.video.split('.')[-1]

    logging.info(f"{'Argument':<10s}:    Value")
    for arg in vars(args):
        logging.info(f"{arg:<20s}:    {getattr(args, arg)}")


    log_guess, log_type, variant = gyrolog.guess_log_type_from_video(args.video)
    logging.info(f"Auto log detection {log_guess}: ")
    logging.info(f"    Log type: {log_type}")
    logging.info(f"    Log variant: {variant}")

    if args.log is None:
        gyroflow_data_path = stabilizer.find_gyroflow_data_file(args.video)
        if gyroflow_data_path:
            logging.info("Using .gyroflow file")
            stab = stabilizer.Stabilizer(args.video, gyroflow_file=gyroflow_data_path)
            # stab.renderfile(starttime=args.start, stoptime=args.end)

        elif log_type == "GoPro GPMF metadata":
            stab = stabilizer.GPMFStabilizer(args.video, args.lens, args.video, hero=variant.replace('hero', ''), fov_scale=args.field_of_view, gyro_lpf_cutoff=args.low_pass_filter)

            stab.set_smoothing_algo()
            stab.update_smoothing()
            if not stab.full_auto_sync(args.max_fitting_error, num_frames_analyze=10, debug_plots=False):
                logging.error("Couldn't auto sync")
                sys.exit()
            logging.info(f"Save stabilized video as '{args.outfile}'")
            stab.renderfile(starttime=args.start, stoptime=args.end)
