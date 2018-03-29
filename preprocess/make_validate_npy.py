import numpy as np
import glob
import ntpath
import pandas
from config_training import config

val_num = 9

trainlist = []
vallist = []

subset_dir = config['luna_raw']
print (subset_dir)

subject_no_dict = {}
for subject_no in range(0, 10):
    src_dir = subset_dir + "subset" + str(subject_no) + "/"
    for src_path in glob.glob(src_dir + "*.mhd"):
        patient_id = ntpath.basename(src_path).replace(".mhd", "")
        subject_no_dict[patient_id] = subject_no

abbrevs = np.array(pandas.read_csv(config['luna_abbr'],header=None))
namelist = list(abbrevs[:,1])
ids = abbrevs[:,0]

print (len(subject_no_dict))

for key, value in subject_no_dict.items():
    id = ids[namelist.index(key)]
    id = '0' * (3 - len(str(id))) + str(id)
    if value != val_num:
        trainlist.append(id)
    else:
        vallist.append(id)

print (len(trainlist))
print (len(vallist))
print (trainlist)
print (vallist)

np.save('train_luna_' + str(val_num) + '.npy',np.array(trainlist))
np.save('val' + str(val_num) + '.npy',np.array(vallist))