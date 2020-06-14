from utils.setup_grid import setup_grid
from DP.DP import run_DP
from os import getcwd, makedirs
from os.path import join, exists
from utils.custom_functions import createFolder, append_summary_to_summaryFile
from QL.TQLearn_RUNNER import run_QL
from definition import ROOT_DIR
import argparse
from multiprocessing import Pool, Process
threshold = 1e-3
dir = ''


def get_dir_name(exp_num):
    global dir
    dir = join(ROOT_DIR, 'Experiments')
    # print("Exp Folder: ", dir)
    return join(dir, str(exp_num))


def create_new_dir():
    n=1
    exp_dir = get_dir_name(n)
    while exists(exp_dir):
        n += 1
        exp_dir = get_dir_name(n)
    makedirs(exp_dir)
    return exp_dir, n


def get_method_str(DP,QL):
    method = 'None'
    if DP != None and QL != None:
        method = 'Both'
    elif DP != None and QL == None:
        method = 'DP'
    elif DP == None and QL != None:
        method = ' QL'
    return method


def append_params_to_summary(exp_summary, input_params, output_params):
    for ip in input_params:
        exp_summary.append(str(ip))
    for op in output_params:
        exp_summary.append(str(op))
    return exp_summary


# output_path = create_new_dir()
output_file = '_'

def run_Experiment(DP = None, QL = None):
    """
    Runs experiment using DP, QL or both.
    Creates new directory automatically
    Save result summary to summary file
    :param DP: [prob_file(just name of file, not path), output_path]
    :param QL: [, .....]
    :return:
    """

    # Path information
    output_path, exp_num = create_new_dir()          #dirs Exp/1, Exp/2, ...
    DP_path = join(output_path,'DP')                 #dirs Exp/1/DP
    QL_path = join(output_path,'QL')                 #dirs Exp/1/QL
    print("************  Exp ", exp_num, "************ \n")

    # Exp_summary_data
    method = get_method_str(DP, QL)
    exp_summary = [str(exp_num), method]


    # Run DP
    if DP != None:
        print("In Runner: Executing DP !!")

        prob_file = DP[0]
        createFolder(DP_path)
        # output_params = [V_so, mean, variance, bad_count]
        output_params = run_DP(setup_grid_params, prob_file, output_file, DP_path, threshold = threshold)

        """CHANGE ARGUMENT if return order of setup_grid() is changed"""
        input_params = setup_grid_params[9].copy()
        input_params.append(prob_file)

        exp_summary = append_params_to_summary(exp_summary, input_params, output_params)
        append_summary_to_summaryFile('Experiments/Exp_summary.csv', exp_summary)
        print("In Runner: Executing DP Finished!!")

    # Run QL
    if QL != None:
        print("In Runner: Executing QL !!")

        QL_params = QL
        createFolder(QL_path)
        output_parameters_all_cases = run_QL(setup_grid_params, QL_params, QL_path, exp_num)
        # run_QL(setup_grid_params, QL_params, QL_path)

        print("In Runner: Executing QL Finished !!")


# Training_traj_size_list, ALPHA_list, esp0_list, QL_Iters, init_Q, with_guidance = QL_params



setup_grid_params = setup_grid(num_actions=16)
model_file = 'GPU_test_17_3D_60nT_a16'




# employ argparse to run code from commanline
parser = argparse.ArgumentParser(description='Take parameters as input args.')
# parser.add_argument('num_passes', type=int, help='number of passes for learning data from trajectories')
# parser.add_argument('QL_Iters', type=int, help='number of QL iters in regfiniement phase')
# parser.add_argument('eps0_list', metavar='eps0_list', type=float, nargs='+',help='eps0_list')
# parser.add_argument('eps_dec_method', type=int, help='1= dec to 0.05 eps0; 2= dec to 0.5eps0')
# parser.add_argument('N_inc', type=float, help='increment parameter for Nsa')
parser.add_argument('dt_size', type=int, help='training data size 0-5000')

args = parser.parse_args()


# Training_traj_size_list, ALPHA_list, esp0_list, QL_Iters, init_Q, with_guidance = QL_params
setup_grid_params = setup_grid(num_actions=16)
model_file = 'GPU_test_17_3D_60nT_a16'



# Paramerers for QL
#Traing data size
dt_size = args.dt_size
Training_traj_size_list = [dt_size]

# ALPHA_list = [0.05, 0.5, 1]
ALPHA_list = [0.05]

esp0_list = [0.1, 0.3]
# esp0_list = [0.33, 0.66, 1]
# esp0_list = args.eps0_list

num_passes_list = [50]
# num_passes_list = [20, 200]
# num_passes_list = args.num_passes

QL_Iters_multiplier_list = [1, 2] #for refinement
# QL_Iters_multiplier_list = [1, 10, 100] #for refinement
# QL_Iters_multiplier_list = args.QL_Iters #for refinement

init_Q = -1

with_guidance = True

method = 'reverse_order'
# mehtod = 'iid'

# eps_dec_method = args.eps_dec_method
eps_dec_method_list = [1]

# N_inc_list = [0.005, 0.05]
N_inc_list = [0.01, 0.05, 0.005]
# N_inc = args.N_inc

cart_prod = [ ([npl], e, n) for npl in num_passes_list for e in eps_dec_method_list for n in N_inc_list]

# QL_params = [Training_traj_size_list, ALPHA_list, esp0_list, QL_Iters_multiplier_list, init_Q, with_guidance, method, num_passes_list, eps_dec_method, N_inc]
# QL_params = [Training_traj_size_list, ALPHA_list, esp0_list, QL_Iters_multiplier_list, init_Q, with_guidance, method, num_passes_list]
QL_params = [Training_traj_size_list, ALPHA_list, esp0_list, QL_Iters_multiplier_list, init_Q, with_guidance, method]

QL_params_list_mp = []
for (npl_list, eps_dec_method, N_inc) in cart_prod:
    QL_params_full = QL_params.copy()
    QL_params_full. append(npl_list)
    QL_params_full. append(eps_dec_method)
    QL_params_full. append(N_inc)
    QL_params_list_mp.append([None, QL_params_full])
    print(QL_params_full)

print(QL_params_list_mp)

# WORKS but all 4 processes not running.may be folder creation issue. not sure.
# with Pool(len(QL_params_list_mp)) as p:
#     output_params_list = p.starmap(run_Experiment, QL_params_list_mp)

p = ['dummy']*len(cart_prod)
for i in range(len(cart_prod)):
    p[i] = Process(target=run_Experiment, args=QL_params_list_mp[i] )

for i in range(len(cart_prod)):
    p[i].start()
    for j in range(1000000):  #spend time between starts
        k=j

for i in range(len(cart_prod)):
    p[i].join()  



print("Launching experiment")


# run_Experiment(QL = QL_params)
# run_Experiment(DP = [model_file])
# run_Experiment(DP = [model_file], QL = QL_params)
