# Modules
import argparse
import json
from config import ROOT
from src.experiments.runner import run

CONFIG_FILE = ROOT/"config.json"
def load_config():
    defaults = {
        "loops": 1,
        "algo": "All",
        "show": True,
        "savedData": str(ROOT/"data"),
        "savedPlot": str(ROOT/"data"/"plots"),
        "onplots": False,
        "noPlots": False,
        "csvName": "global",
        "combinePlots": False
    }
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            try:
                return {**defaults, **json.load(f)}
            except json.JSONDecodeError:
                return defaults
    return defaults
def show_image(option):
    if option:
        with open("ascii.txt","r") as f:
            ct = f.read()
        print(ct)
def save_config(data):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=True)
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid argument.')
def main():

    current_config = load_config()

    parser = argparse.ArgumentParser(description="CipherEntropy CLI")

    parser.add_argument("-n", help="Run with n loops", type=int,default=1, metavar="Loops")
    parser.add_argument("-al", metavar="Algorithm", help = "Run with specific algorithm", type=str,  choices=["DES", "RC4", "AES","All"],default="All")
    parser.add_argument("-i", metavar="yes,y,1,no,n,0,true,false", help="Run with ASCII art prompt", type=str_to_bool,default=current_config["show"])
    parser.add_argument("-sd", metavar="Directory", help="Change directory of data",type=str,default=current_config["savedData"])
    parser.add_argument("-sp", metavar="Directory", help="Change directory of plots",type=str, default=current_config["savedPlot"])
    parser.add_argument("-rc", metavar="True/False", help="Reset default configuration on config.json", type=bool, default=False)
    parser.add_argument("-op", help="Only plots default data", action="store_true")
    parser.add_argument("-np", help="Skip plot generation", action="store_true")
    parser.add_argument("-oc", metavar="Name", help="Base name for the output CSV files (e.g. 'aes' -> aes.csv / aes_keys.csv)", type=str, default=current_config["csvName"])
    parser.add_argument("-cp", help="Combine every data/csv/*.csv file for cross-algorithm plots instead of just this job's own CSV", action="store_true")
    parser.add_argument("-r", help="Run program", action="store_true")
    args = parser.parse_args()
    if not args.rc:
        final_loops = args.n if args.n else current_config["loops"]
        final_algo =  args.al if args.al else current_config["algo"]
        final_prompt = args.i
        final_datap =  args.sd if args.sd else current_config["savedData"]
        final_plotp =  args.sp if args.sp else current_config["savedPlot"]
        final_csvname = args.oc if args.oc else current_config["csvName"]
        runtime_config = {
            "loops":final_loops,
            "algo":final_algo,
            "show":final_prompt,
            "savedData":final_datap,
            "savedPlot":final_plotp,
            "onplots":args.op,
            "noPlots":args.np,
            "csvName":final_csvname,
            "combinePlots":args.cp
        }
        save_config(runtime_config)
        current_config = runtime_config
    else:
        defaults = {
            "loops": 1,
            "algo": "All",
            "show": True,
            "savedData": str(ROOT/"data"),
            "savedPlot": str(ROOT/"data"/"plots"),
            "onplots": False,
            "noPlots": False,
            "csvName": "global",
            "combinePlots": False
        }
        save_config(defaults)
        current_config = defaults
    show_image(current_config["show"])
    if args.r:
        print("Reading config.json file...")
        run(current_config)
    else:
        print("Please add -r to run program")