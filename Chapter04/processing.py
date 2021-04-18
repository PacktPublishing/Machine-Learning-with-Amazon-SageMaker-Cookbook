import pandas as pd
import argparse
import os
import subprocess
import importlib

from sys import executable


def install_and_load(target):
    sequence = [executable, "-m", "pip", "install", target]
    subprocess.call(sequence)
    
    print(f'[+] Successfully installed {target}')
    
    return importlib.import_module(target)
    
    
def uninstall(target):
    sequence = [executable, "-m", "pip", "uninstall", "-y", target]
    subprocess.call(sequence)
    
    print(f'[+] Successfully uninstalled {target}')

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-argument', type=int, default=1)
    arguments, _ = parser.parse_known_args()
    
    return arguments

def load_input(input_target):
    df = pd.read_csv(input_target)
    
    return df

def save_output(output_target):
    with open(output_target, 'w') as writer:
        writer.write("sample output\n")


def main():
    args = process_args()
    print(args)
    
    plt = install_and_load('matplotlib')
    print(plt)
    uninstall('matplotlib')
    
    sample_input = load_input("/opt/ml/processing/input/dataset.processing.csv")
    print(sample_input)
    
    save_output("/opt/ml/processing/output/output.csv")
    print('[+] DONE')
    
    
if __name__ == "__main__":
    main()