# Copyright (c) [2024] Haopeng Geng, The University of Tokyo
## Plot the duration distribution of the segments
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", help='Path to wav.scp file')
    parser.add_argument("--db_root", help='Root directory of the database')
    
    args = parser.parse_args()
    # get all wav files from scp file
    try:
        # read the wav.scp file and get the file path
        with open(args.wav_scp, 'r') as f:
            lines = f.readlines()
            files = [line.split()[1] for line in lines]
    except:
        try:
            # go through the db_root and get all the wav files
            db_root = Path(args.db_root)
            files = list(db_root.glob('**/*.wav'))
        except:
            raise ValueError("Please provide either the wav.scp file or the db_root")
        
    # get duration of each wav file
    durations = []
    for file in files:
        import librosa
        y, sr = librosa.load(file, sr=None)
        durations.append(librosa.get_duration(y=y, sr=sr))
    durations = np.array(durations)

    # get the outliers, show their file index
    long_outliers = np.where(durations > 20)
    # short_outliers = np.where(durations < 0.5)
    
    # select valid durations files and copy them to a new dir
    valid_files = [file for i, file in enumerate(files) if i not in long_outliers[0] and i not in long_outliers[0]]
    # print the valid files
    print("The valid files are:")
    for file in valid_files:
        print(file)
    # mkdir for the valid files base on db_root 
    valid_dir = Path(args.db_root).parent / Path(Path(args.db_root).stem + "_valid")
    valid_dir.mkdir(exist_ok=True)
    for file in valid_files:
        file_name = Path(file).name
        Path(valid_dir / file_name).symlink_to(file)
        
    
    # show file name
    print("The outliers are:")
    for i in long_outliers[0]:
        print(files[i])
    
    # plot the violin plot
    plt.violinplot(durations, showmedians=True)
    
    plt.xlabel('Duration (s)')
    plt.ylabel('Count')
    plt.title('Duration Distribution')
    plt.show()
    
    # save the plot base on the db_root or the wav_scp
    # make id first
    if args.wav_scp:
        id = Path(args.wav_scp).stem
    else:
        id = Path(args.db_root).stem
    plt.savefig(f"{id}_duration_distribution.png")
        
if __name__ == "__main__":
    main()