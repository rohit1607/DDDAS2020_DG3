from utils.setup_grid import setup_grid
from DP.DP import run_DP
from os import getcwd, makedirs
from os.path import join, exists
from utils.custom_functions import createFolder, append_summary_to_summaryFile
# from QL.TQLearn_RUNNER import run_QL
from definition import ROOT_DIR

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
        # run_QL(setup_grid_params, QL_params, QL_path)

        print("In Runner: Executing QL Finished !!")


# Training_traj_size_list, ALPHA_list, esp0_list, QL_Iters, init_Q, with_guidance = QL_params

setup_grid_params = setup_grid(num_actions=16)
model_file = 'GPU_test_5_3D_90nT_a16'

Training_traj_size_list = [5000]
ALPHA_list = [0.5]
esp0_list = [0.25]

QL_Iters = 100
init_Q = -1e6
with_guidance = True
method = 'reverse_order'
num_passes = 50
QL_params = [Training_traj_size_list, ALPHA_list, esp0_list, QL_Iters, init_Q, with_guidance, method, num_passes]

print("Launnching experiment")

# run_Experiment(QL = QL_params)
run_Experiment(DP = [model_file])
# run_Experiment(DP = [model_file], QL = QL_params)
