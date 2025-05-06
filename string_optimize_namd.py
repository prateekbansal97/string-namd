"""
Sequential string-method with swarms for GABAB receptor activation
CVs: backbone RMSD to active, intra-lobe distance, inter-subunit distance
"""

import os
import numpy as np, json, pathlib, copy
import glob, subprocess
from multiprocessing import Pool
from math import atan2, pi, sqrt
import mdtraj as md
from tqdm import tqdm
import string

# ---------------------------------------------------------------------
# ---- USER-TUNEABLE PARAMETERS ---------------------------------------
NUM_IMAGES       = 50
NUM_SWARMS       = 20 
NUM_SWARM_STEPS  = 20          # tiny; just to probe local gradient
NUM_ITER         = 300
#TIME_STEP_FS     = 1.0
#TEMP_K           = 310.0
SMOOTH_LAMBDA    = 0.10
#K_BIAS           = 0.2 * unit.kilocalories_per_mole / unit.angstrom**2
K_RAMP_STAGES    = 20 
EQUIL_STEPS      = 20000
PSF_FILE         = 'Initial_Path2/step5_input_namd.psf' 
OUTDIR           = pathlib.Path("string_out")
NAMD_EXE     = "namd3"     # give full path or module name if needed
MAX_WORKERS  = 20          # how many NAMD processes to run *at once*
PARAMSET         = ['../Initial_Path2/toppar_particular/par_all36_lipid.prm',
'../Initial_Path2/toppar_particular/par_all36m_prot.prm',
'../Initial_Path2/toppar_particular/toppar_all36_carb_glycolipid.str',
'../Initial_Path2/toppar_particular/toppar_all36_lipid_cholesterol.str',
'../Initial_Path2/toppar_particular/toppar_all36_lipid_sphingo.str',
'../Initial_Path2/toppar_particular/toppar_water_ions.str',
'../toppar_c36_jul20/par_all36_na.prm',
'../toppar_c36_jul20/par_all36_carb.prm',
'../toppar_c36_jul20/par_all36_cgenff.prm',
'../toppar_c36_jul20/par_all36_lipid.prm']
PARAMSET = [os.path.abspath(j) for j in PARAMSET]
PARAMSET = [f'parameters {j}\n' for j in PARAMSET]
PARAMSET = ''.join(PARAMSET)
OUTDIR.mkdir(exist_ok=True)



def make_output_dirs(it, img, is_swarm, equi_stage=None):
    if is_swarm:
        for swarm in range(1, NUM_SWARMS+1):
            suboutput = f'string_out/iter{it}/img{img}/sw{swarm}'
            if not os.path.exists(suboutput):
                os.system(f'mkdir -p {suboutput}')
            #pathlib.Path(suboutput).mkdir(exist_ok=True)
    else:
        for img in range(1, NUM_IMAGES+1):
            for equi_stage in range(1, K_RAMP_STAGES+1): 
                suboutput = f'string_out/iter{it}/img{img}/equi{equi_stage}'
                if not os.path.exists(suboutput):
                    os.system(f'mkdir -p {suboutput}')
            #pathlib.Path(suboutput).mkdir(exist_ok=True)
        
def make_conf(it, img, is_swarm):
    with open('6VJM_namd_string.template', 'r') as fhandle:
        conf = string.Template(fhandle.read())
    make_output_dirs(it, img, is_swarm)
    psf_file = os.path.abspath('../Initial_Path2/step5_input_namd.psf')
    coord_file = os.path.abspath('../Initial_Path2/6VJM_APO_HMass_NPT_formatted.pdb')
    if it == 1:
        bincoord_fname = os.path.abspath(f'../Initial_Path2/6VJM_{(img - 1):02d}.coor')
        xsc_fname = os.path.abspath(f'../Initial_Path2/6VJM_{(img - 1):02d}.xsc')
    else:
        bincoord_fname = os.path.abspath(f'string_out/iter{it-1}/img{img}/equi{K_RAMP_STAGES}/6VJM_out_iter{it-1}_img{img}_equi20_out.coor')
        xsc_fname = os.path.abspath(f'string_out/iter{it-1}/img{img}/equi{K_RAMP_STAGES}/6VJM_out_iter{it-1}_img{img}_equi20_out.xsc')

    if is_swarm:
        nsteps = NUM_SWARM_STEPS
        for swarm in range(1, NUM_SWARMS+1):
            conf_fname = os.path.abspath(f'string_out/iter{it}/img{img}/sw{swarm}/6VJM_conf_iter{it}_img{img}_sw{swarm}.conf.tcl')
            colvars_fname = os.path.abspath(f'string_out/iter{it}/img{img}/sw{swarm}/6VJM_colvars_iter{it}_img{img}_sw{swarm}.colvars.tcl')
            output_fname = os.path.abspath(f'string_out/iter{it}/img{img}/sw{swarm}/6VJM_out_iter{it}_img{img}_sw{swarm}_out')
            conff = conf.safe_substitute(colvars_fname = colvars_fname,
                    output_fname = output_fname, 
                    nsteps = nsteps,
                    bincoord_fname = bincoord_fname, 
                    vel_line = 'temperature 310',
                    xsc_fname = xsc_fname, 
                    psf_file = psf_file,
                   coord_file = coord_file, parameters = PARAMSET)
            with open(conf_fname, 'w') as fhandle:
                fhandle.write(conff)
    else:
        nsteps = EQUIL_STEPS//K_RAMP_STAGES 
        for equi_stage in range(1, K_RAMP_STAGES+1):
            #binvel_fname = os.path.abspath(f'string_out/iter{it}/img{img}/equi{equi_stage}/6VJM_out_iter{it}_img{img}_equi{equi_stage - 1}_out.vel')
            if equi_stage == 1:
                vel_line = f'temperature 310'
            else:
                binvel_fname = os.path.abspath(f'string_out/iter{it}/img{img}/equi{equi_stage - 1}/6VJM_out_iter{it}_img{img}_equi{equi_stage - 1}_out.vel')
                bincoord_fname = os.path.abspath(f'string_out/iter{it}/img{img}/equi{equi_stage - 1}/6VJM_out_iter{it}_img{img}_equi{equi_stage - 1}_out.coor')
                xsc_fname = os.path.abspath(f'string_out/iter{it}/img{img}/equi{equi_stage - 1}/6VJM_out_iter{it}_img{img}_equi{equi_stage - 1}_out.xsc')
                vel_line = f'binVelocities {binvel_fname}'
            conf_fname = os.path.abspath(f'string_out/iter{it}/img{img}/equi{equi_stage}/6VJM_conf_iter{it}_img{img}_equi{equi_stage}.conf.tcl')
            colvars_fname = os.path.abspath(f'string_out/iter{it}/img{img}/equi{equi_stage}/6VJM_colvars_iter{it}_img{img}_equi{equi_stage}.colvars.tcl')
            output_fname = os.path.abspath(f'string_out/iter{it}/img{img}/equi{equi_stage}/6VJM_out_iter{it}_img{img}_equi{equi_stage}_out')
            conff = conf.safe_substitute(colvars_fname = colvars_fname,
                   output_fname = output_fname,
                   nsteps = nsteps,
                   bincoord_fname = bincoord_fname,
                   vel_line = vel_line,
                   xsc_fname = xsc_fname,
                   psf_file = psf_file,
                   coord_file = coord_file, parameters = PARAMSET)
    
            with open(conf_fname, 'w') as fhandle:
                fhandle.write(conff)

def make_colvar(it, img, is_swarm, new_centers=None):
    with open('6VJM_colvars_string.template', 'r') as fhandle:
        colvar = string.Template(fhandle.read())
     
    reffile = os.path.abspath('../Initial_Path2/6UOA_APO_HMass_NPT_PROT.pdb')
    colvar_names = ['CA_RMSD', 'CV1_distance', 'CV2_distance'] + [f'res_distance{i}' for i in range(1, 29)]
    k_names = [f'bias_{j}_K' for j in colvar_names]
    center_names = [f'bias_{j}_CENTER' for j in colvar_names]
    k_values_full = np.array([1000, 40, 40] + [5]*28)

    if is_swarm:
        trajfreq = 10
        for swarm in range(1, NUM_SWARMS+1):

            colvars_fname = os.path.abspath(f'string_out/iter{it}/img{img}/sw{swarm}/6VJM_colvars_iter{it}_img{img}_sw{swarm}.colvars.tcl')
            k_values = {j:0.0 for j in k_names} #Remove bias
            center_values = {j:0.0 for j in center_names}
            mapping = {}
            mapping.update(k_values)        # k_* entries
            mapping.update(center_values)   # center* entries
            mapping['reffile'] = reffile
            mapping['trajfreq'] = trajfreq
            colvarf = colvar.safe_substitute(mapping)
            with open(colvars_fname, 'w') as fhandle:
                fhandle.write(colvarf)
    else:
        trajfreq = 1000
        if new_centers is None:
            raise ValueError("NOT SUPPLIED NEW CENTERS")
        for equi_stage in range(1, K_RAMP_STAGES+1):
            k_values_stage = k_values_full*(equi_stage)/K_RAMP_STAGES
            colvars_fname = os.path.abspath(f'string_out/iter{it}/img{img}/equi{equi_stage}/6VJM_colvars_iter{it}_img{img}_equi{equi_stage}.colvars.tcl')
            #print("IMG - 1", img-1)
            center_values= {j:k for j,k in zip(center_names, new_centers[img - 1])}
            k_values = {j:k for j,k in zip(k_names,k_values_stage)} #Ramp up bias according to stage
            mapping = {}
            mapping.update(k_values)        # k_* entries
            mapping.update(center_values)   # center* entries
            mapping['reffile'] = reffile
            mapping['trajfreq'] = trajfreq
            colvarf = colvar.safe_substitute(mapping)
            with open(colvars_fname, 'w') as fhandle:
                fhandle.write(colvarf)

def get_new_centers_from_first_step():
    for img in range(1, NUM_IMAGES+1):
        CV_path = os.path.abspath(f'string_out/iter1/img{img}/sw{swarm}/6VJM_out_iter1_img{img}_sw1_out.colvars.traj')
        with open(CV_path, 'r') as f:
            lines = f.readlines()[1].strip().split()
        CV_val_swarm = np.array([float(j) for j in lines[1:]])
        CV_vals[swarm - 1] = CV_val_swarm
    return CV_vals

def read_CVs(it, img, done=False):
    if not done:
        CV_vals = np.zeros((NUM_SWARMS, 31))
        for swarm in range(1, NUM_SWARMS+1):
            CV_path = os.path.abspath(f'string_out/iter{it}/img{img}/sw{swarm}/6VJM_out_iter{it}_img{img}_sw{swarm}_out.colvars.traj')
            with open(CV_path, 'r') as f:
                lines = f.readlines()[-1].strip().split()
            CV_val_swarm = np.array([float(j) for j in lines[1:]])
            CV_vals[swarm - 1] = CV_val_swarm
        mean_CV_vals = np.mean(CV_vals, axis = 0)
        with open(f'./string_out/iter{it}/img{img}/6VJM_out_iter{it}_img{img}_mean.CV.csv', 'w') as f:
            f.write(','.join([str(j) for j in mean_CV_vals.tolist()]))
    else:
        mean_CV_vals = np.genfromtxt(f'./string_out/iter{it}/img{img}/6VJM_out_iter{it}_img{img}_mean.CV.csv', delimiter=',')
    return mean_CV_vals

def _run_namd(conf):
    """Worker: run one namd3 job and save output to .log next to .conf."""
    log = conf.replace(".conf.tcl", ".log")
    with open(log, "w") as fp:
        subprocess.run([NAMD_EXE, conf],
                       stdout=fp, stderr=subprocess.STDOUT, check=True)

def do_simulation(it, img, is_swarm):
    """
    Launch all .conf.tcl files for the current iteration.
    If is_swarm==True → use img*/sw*/ pattern,
    else               → img*/eq*/ pattern.
    """
    phase = "swarms" if is_swarm else "equilibrations"

    it_dir  = f"string_out/iter{it}"           
    img_dir = f"img{img}"
    
    if is_swarm:
        pattern = f"{img_dir}/sw*/6VJM_conf*.conf.tcl"
        confs = sorted(glob.glob(os.path.join(it_dir, pattern)))
        print(f"Launching {len(confs)} {phase} "
          f"(≤{MAX_WORKERS} concurrent namd3 processes).")

        with Pool(processes=MAX_WORKERS) as pool:
            pool.map(_run_namd, confs)
        print(f"… finished {phase}")
    else:    
        confs = [os.path.abspath(f'./string_out/iter{it}/img{img}/equi{equi_stage}/6VJM_conf_iter{it}_img{img}_equi{equi_stage}.conf.tcl') for equi_stage in range(1, K_RAMP_STAGES+1) for img in range(1, NUM_IMAGES+1)]
        for stage in (pbar4 := tqdm(range(1, K_RAMP_STAGES+1), leave=False)):
            pbar4.set_description(f'RUNNING STAGE {stage}')
            confs_final = confs[(stage-1)*NUM_IMAGES:stage*NUM_IMAGES]
            #print(confs_final)
            #print(f"Launching {len(confs_final)} {phase} "
            #f"(≤{MAX_WORKERS} concurrent namd3 processes).")
            with Pool(processes=MAX_WORKERS) as pool:
                pool.map(_run_namd, confs_final)
            print(f"… finished {phase} stage {stage}")

def smooth(arr):
    new = arr.copy()
    for i in range(1,len(arr)-1):
        new[i] = (1-SMOOTH_LAMBDA)*arr[i] + 0.5*SMOOTH_LAMBDA*(arr[i-1]+arr[i+1])
    return new

def reparam(centers):
    s=[0.]; 
    for i in range(1,len(centers)):
        s.append(s[-1] + np.linalg.norm(centers[i]-centers[i-1]))
    total=s[-1]
    equal=[centers[0]]
    for i in range(1,len(centers)-1):
        tgt=i*total/(len(centers)-1)
        for j in range(1,len(s)):
            if s[j]>=tgt: break
        lam=(tgt-s[j-1])/(s[j]-s[j-1])
        equal.append(centers[j-1]+lam*(centers[j]-centers[j-1]))
    equal.append(centers[-1])
    #pprint("Reparameterized")
    return np.array(equal)

# ---- MAIN LOOP ------------------------------------------------------
for it in (pbar := tqdm(range(1, NUM_ITER+1))):
    pbar.set_description(f'Epoch {it}')
    drifts = np.zeros((NUM_IMAGES, 31))
    #print(drifts.shape)
    for img in (pbar2 := tqdm(range(1, NUM_IMAGES+1), leave=False)):
        pbar2.set_description(f'Image {img}')
        is_swarm = True
        make_conf(it, img, is_swarm)
        make_colvar(it, img, is_swarm)
        if not os.path.exists(os.path.abspath(f'./string_out/iter{it}/img{img}/6VJM_out_iter{it}_img{img}_mean.CV.csv')):
            do_simulation(it, img, is_swarm)
            mean_vals = read_CVs(it, img) 
        else:
            mean_vals = read_CVs(it, img, done=True)
        #mean_vals = np.mean(vals,axis=0)
        drifts[img-1]=mean_vals #np.mean(vals,axis=0)
        #print(drifts)
    centers = reparam(smooth(drifts))
    np.savetxt(OUTDIR/f"centers_iter{it:03d}.dat", centers)
    #print(drifts, centers)
    # ---- equilibration with bias ramp ----
    is_swarm = False
    for img in range(1, NUM_IMAGES + 1):
        is_swarm = False
        make_conf(it, img, is_swarm)
        make_colvar(it, img, is_swarm, centers)
    #for img in (pbar4 := tqdm(range(1, NUM_IMAGES + 1))):    
    #    pbar4.set_description(f"equilibrating {img}")
    for img in range(1, NUM_IMAGES + 1):
        if os.path.exists(f'./string_out/iter{it}/img{img}/equi20/6VJM_out_iter{it}_img{img}_equi20_out.coor'):
            pass
        else:
            do_simulation(it, img, is_swarm)

    print(f"iteration {it+1}/{NUM_ITER} done")

print("String optimization finished.")
