import os
import glob

parent_dir = "results"
patterns = ["nni_*", "iba_*"]

for pattern in patterns:
    dir_paths = glob.glob(os.path.join(parent_dir, pattern))

    for dir_path in dir_paths:
        subdir = [subdir for subdir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, subdir))][0]
        
        log_file = os.path.join(dir_path, subdir, "log.txt")
        if os.path.exists(log_file):
            with open(log_file, "r") as file:
                lines = file.readlines()
                
                name = dir_path.__str__().split("\\")[1]
                type, PR = name.split('_')
                type = type.upper()
                IoU = lines[-33].split(' ')[-2]
                PA = lines[-30].split(' ')[-2]
                ASR = lines[-2].split(' ')[-1][:-2]
                
                print(f'{type} & {int(float(PR) * 100)} & {IoU} & {PA} & {ASR} \\\\')