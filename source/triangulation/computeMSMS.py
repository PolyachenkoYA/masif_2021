import os
from subprocess import Popen, PIPE

from input_output.read_msms import read_msms
from triangulation.xyzrn import output_pdb_as_xyzrn
from default_config.global_vars import msms_bin
from default_config.masif_opts import masif_opts
#import random
import sys

# Pablo Gainza LPDI EPFL 2017-2019
# Calls MSMS and returns the vertices.
# Special atoms are atoms with a reduced radius.
def computeMSMS(pdb_file,  have_xyzrn=True):
    #randnum = random.randint(1,10000000)
    file_base = masif_opts['tmp_dir']+"/msms_"+os.path.basename(pdb_file)
    out_xyzrn = file_base+".xyzrn"
    #print('msms file' + file_base)

    if((not os.path.isfile(file_base + '.vert')) or (not os.path.isfile(file_base + '.face')) or (not os.path.isfile(file_base + '.area'))):
        if have_xyzrn:
            output_pdb_as_xyzrn(pdb_file, out_xyzrn)
        else:
            print("Error - pdb2xyzrn is deprecated.")
            sys.exit(1)
        # Now run MSMS on xyzrn file
        #FNULL = open(os.devnull, 'w')
        args = [msms_bin, "-density", "3.0", "-hdensity", "3.0", "-probe",\
                        "1.5", "-if",out_xyzrn,"-of",file_base, "-af", file_base + '.area']
        #print(args)
        p2 = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()

    vertices, faces, normals, names = read_msms(file_base)
    areas = {}
    ses_file = open(file_base + '.area')
    next(ses_file) # ignore header line
    for line in ses_file:
        fields = line.split()
        areas[fields[3]] = fields[1]
    return vertices, faces, normals, names, areas

