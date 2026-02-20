#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
from pathlib import Path

def get_multi_data_dir(process_sh_path):
    """Extract multi_data_dir variable value from process.sh file"""
    with open(process_sh_path, 'r') as f:
        lines = f.readlines()
    
    data_name = None
    base_data = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('data_name='):
            data_name = line.split('=')[1].strip('"\'')
        elif line.startswith('base_data='):
            base_data = line.split('=')[1].strip('"\'')
    
    if data_name and base_data:
        return os.path.join(base_data, data_name)
    else:
        raise ValueError("Cannot extract multi_data_dir from process.sh")

def main():
    parser = argparse.ArgumentParser(description='Move multi-robot mapping results to target folder')
    parser.add_argument('--source', '-s', type=str, default='/home/cyw/Desktop/temp/multi_agent_mapping_egypt',
                      help='Source folder path containing robot_1, robot_2, robot_3 directories')
    parser.add_argument('--dest', '-d', type=str, help='Destination directory path')
    parser.add_argument('--process_sh', '-p', type=str, 
                      default='/home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/src/FAST_LIO_SLAM/SC-PGO/shell/Egent_project/process.sh',
                      help='Path to process.sh file')
    
    args = parser.parse_args()
    
    # Get target directory
    if args.dest:
        dest_dir = args.dest
    else:
        try:
            dest_dir = get_multi_data_dir(args.process_sh)
            print(f"Target directory: {dest_dir}")
        except Exception as e:
            print(f"Error: {e}")
            dest_dir = input("Please enter target directory path: ")
    
    # Ensure target directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Process each robot's data (robot_1, robot_2, robot_3)
    source_dir = Path(args.source)
    for i in range(1, 4):  # robot_1, robot_2, robot_3
        robot_dir = source_dir / f"robot_{i}"
        robot_name = f"robot{i}"
        
        if not robot_dir.exists():
            continue
        
        # Process Scans directory
        scans_dir = robot_dir / f"{robot_name}_Scans"
        if scans_dir.exists():
            dest_scans_dir = Path(dest_dir) / f"{robot_name}_Scans"
            if dest_scans_dir.exists():
                shutil.rmtree(dest_scans_dir)
            shutil.copytree(scans_dir, dest_scans_dir)
        
        # Process pose txt file
        txt_file = robot_dir / f"{robot_name}.txt"
        if txt_file.exists():
            dest_txt_file = Path(dest_dir) / f"{robot_name}.txt"
            shutil.copy2(txt_file, dest_txt_file)
    
    print(f"Results moved to: {dest_dir}")

if __name__ == "__main__":
    main()