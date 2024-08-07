# copy common wavs into a new directory base on comm_all_id list

import os
import sys
import shutil

if len(sys.argv) != 4:
    print("Usage: %s <src_dir> <dst_dir> <comm_all_id>" % sys.argv[0])
    sys.exit(1)
    
src_dir = sys.argv[1]
dst_dir = sys.argv[2]
comm_all_id = sys.argv[3]

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
# read comm_all_id
with open(comm_all_id, 'r') as f:
    comm_all_id_list = f.readlines()
comm_all_id_list = [x.strip() for x in comm_all_id_list]

# walk through src_dir and copy common wavs to dst_dir
for root, dirs, files in os.walk(src_dir):
    
    for file in files:
        if file.endswith(".wav"):
            file_id = file
            if file_id in comm_all_id_list:
                shutil.copy(os.path.join(root, file), dst_dir)
                print("copy %s to %s" % (os.path.join(root, file), dst_dir))
print("Done")