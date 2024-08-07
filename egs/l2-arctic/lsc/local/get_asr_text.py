# get audio_dir and text file with <id> <text> format
# select the valid id from the text file
# and make a new text file with <id> <text> format base on the valid ids

import os
import sys
import pdb
audio_dir = sys.argv[1]
text_file = sys.argv[2]
audio_dir_style = sys.argv[3] # audio or scp

valid_ids = []
# case 1
# get valid audio ids base on the audio files
if audio_dir_style == "audio":
    for audio_file in os.listdir(audio_dir):
        audio_id = audio_file.split(".")[0]
        valid_ids.append(audio_id)

# case 2
# get valid audio ids base on the wav.scp file in audio_dir
elif audio_dir_style == "scp":
    with open(os.path.join(audio_dir, "wav.scp"), "r") as f:
        for line in f:
            audio_id, audio_path = line.strip().split(" ")
            valid_ids.append(audio_id)
else:
    raise ValueError("audio_dir_style should be audio or scp")        
# get valid text <id> <text> base on the text file, save into dict
transcripts = {}
with open(text_file, "r") as f:
    for line in f:
        # line remove the leading and trailing whitespaces, and "(" and ")"
        line = line.strip().replace("(", "").replace(")", "")
        audio_id, text = line.split(" ", 1)
        audio_id = audio_id.replace(" ", "")
        if audio_id in valid_ids:
            transcripts[audio_id] = text
            
transcripts = dict(sorted(transcripts.items()))
# save the valid text in <audio_dir>/text
with open(os.path.join(audio_dir, "text"), "w") as f:
    for audio_id, text in transcripts.items():
        f.write(f"( {audio_id} {text} )\n")
# pdb.set_trace()