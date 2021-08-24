import argparse
import gyroflow
import gyrolog
import logging
import sys

logging.basicConfig(filename='gyroflow_cli.log', level=logging.DEBUG)

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info("\nStarting Gyroflow CLI\n")
# Instantiate the parser
parser = argparse.ArgumentParser(prog="Gyroflow", description='Gyroflow - Gyro based video stabilizer')
parser.add_argument('video', type=str, help='Video input path')
parser.add_argument('-l', '--log')

args = parser.parse_args()

logging.info(f"{'Argument':<10s}:    Value")
for arg in vars(args):
    logging.info(f"{arg:<10s}:    {getattr(args, arg)}")

args.video = r"D:\git\FPV\videos\GH011171.MP4"

log_guess, log_type, variant = gyrolog.guess_log_type_from_video(args.video)
logging.info(f"Auto log detection {log_guess}: ")
logging.info(f"    Log type: {log_type}")
logging.info(f"    Log variant: {variant}")

if log_type == "GoPro GPMF metadata":
    print("do gopro stuff")
             # , log_type, variant)