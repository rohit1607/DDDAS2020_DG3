import numpy as np
import matplotlib.pyplot as plt
import math
from definition import c_ni, ROOT_DIR
from utils.custom_functions import createFolder, picklePolicy, read_pickled_File, extract_velocity
from os.path import join
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx


def action_to_quiver(a):
    vt = a[0]
    theta = a[1]
    vtx = vt * math.cos(theta)
    vty = vt * math.sin(theta)
    return vtx, vty

def plot_exact_trajectory():
    return

def plot_exact_trajectory_set(g, policy, X, Y, vel_field_data, fpath,
                                                fname='Trajectories'):

    # time calculation and state trajectory
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    # set grid
    minor_xticks = np.arange(g.xs[0] - 0.5 * g.dj, g.xs[-1] + 2 * g.dj, g.dj)
    minor_yticks = np.arange(g.ys[0] - 0.5 * g.di, g.ys[-1] + 2 * g.di, g.di)

    major_xticks = np.arange(g.xs[0], g.xs[-1] + 2 * g.dj, 5 * g.dj)
    major_yticks = np.arange(g.ys[0], g.ys[-1] + 2 * g.di, 5 * g.di)

    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)

    ax.grid(which='major', color='#CCCCCC', linestyle='')
    ax.grid(which='minor', color='#CCCCCC', linestyle='--')
    st_point= g.start_state
    plt.scatter(g.xs[st_point[2]], g.ys[g.ni - 1 - st_point[1]], c='g')
    plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')

    plt.quiver(X, Y, vel_field_data[0][0,:,:], vel_field_data[1][0, :, :])

    nt, nrzns, nmodes = vel_field_data[4].shape #vel_field_data[4] is all_Yi

    bad_count =0
    t_list_all=[]
    t_list_reached=[]
    G_list=[]
    traj_list = []

    for rzn in range(nrzns):
        g.set_state(g.start_state)
        dont_plot =False
        # bad_flag = False
        # t = 0
        G = 0

        xtr = []
        ytr = []

        t, i, j = g.start_state

        a = policy[g.current_state()]
        xtr.append(g.x)
        ytr.append(g.y)

        # while (not g.is_terminal()) and g.if_within_actionable_time():
        while True:
            vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
            r = g.move_exact(a, vx, vy)
            G = G + r
            # t += 1
            (t, i, j) = g.current_state()

            xtr.append(g.x)
            ytr.append(g.y)

            # if edge state encountered, then increment badcount and Dont plot
            if g.if_edge_state((i,j)):
                bad_count += 1
                dont_plot=True
                break

            if (not g.is_terminal()) and  g.if_within_actionable_time():
                a = policy[g.current_state()]
            elif g.is_terminal():
                break
            else:
            #  i.e. not terminal and not in actinable time.
            # already checked if ternminal or not. If not terminal 
            # if time reaches nt ie not within actionable time, then increment badcount and Dont plot
                bad_count+=1
                bad_flag=True
                # dont_plot=True
                break


        if dont_plot==False:
            plt.plot(xtr, ytr)
            traj_list.append((xtr, ytr))
            t_list_all.append(t)
            if bad_flag==False:
                t_list_reached.append(t)
            G_list.append(G)
        
        #ADDED for trajactory comparison
        else:
            traj_list.append(None)
            t_list_all.append(None)
            G_list.append(None)

    if fname != None:
        plt.savefig(fname, dpi=300)
        print("*** pickling traj_list ***")
        picklePolicy(traj_list, join(fpath,fname))
        print("*** pickled ***")

    bad_count_tuple = (bad_count, str(bad_count * 100 / nrzns) + '%')
    return t_list_all, t_list_reached, G_list, bad_count_tuple


def plot_learned_policy(g, DP_params = None, QL_params = None,  showfig = False):
    """
    Plots learned policy
    :param g: grid object
    :param DP_params: [policy, filepath]
    :param QL_params: [policy, Q, init_Q, label_data, filepath]  - details mentioned below
    :param showfig: whether you want to see fig during execution
    :return:
    """
    """
    QL_params:
    :param Q: Leared Q against which policy is plotted. This is needed just for a check in the QL case. TO plot policy only at those states which have been updated
    :param policy: Learned policy.
    :param init_Q: initial value for Q. Just like Q, required only for the QL policy plot
    :param label_data: Labels to put on fig. Currently requiered only for QL
    """
    # TODO: check QL part for this DG3
    # full_file_path = ROOT_DIR
    if DP_params == None and QL_params == None:
        print("Nothing to plot! Enter either DP or QL params !")
        return

    # set grid
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(1, 1, 1)

    minor_xticks = np.arange(g.xs[0] - 0.5 * g.dj, g.xs[-1] + 2 * g.dj, g.dj)
    minor_yticks = np.arange(g.ys[0] - 0.5 * g.di, g.ys[-1] + 2 * g.di, g.di)

    major_xticks = np.arange(g.xs[0], g.xs[-1] + 2 * g.dj, 5 * g.dj)
    major_yticks = np.arange(g.ys[0], g.ys[-1] + 2 * g.di, 5 * g.di)

    ax1.set_xticks(minor_xticks, minor=True)
    ax1.set_yticks(minor_yticks, minor=True)
    ax1.set_xticks(major_xticks)
    ax1.set_yticks(major_yticks)

    ax1.grid(which='major', color='#CCCCCC', linestyle='')
    ax1.grid(which='minor', color='#CCCCCC', linestyle='--')
    xtr=[]
    ytr=[]
    ax_list=[]
    ay_list=[]


    if QL_params != None:
        policy, Q, init_Q, label_data, full_file_path = QL_params
        F, ALPHA, initq, QIters = label_data
        ax1.text(0.1, 9, 'F=(%s)'%F, fontsize=12)
        ax1.text(0.1, 8, 'ALPHA=(%s)'%ALPHA, fontsize=12)
        ax1.text(0.1, 7, 'initq=(%s)'%initq, fontsize=12)
        ax1.text(0.1, 6, 'QIters=(%s)'%QIters, fontsize=12)
        for s in Q.keys():
            # for a in Q[s].keys():
                # if s[0]==0 and a == policy[s]: # to print policy at time t = 0
            a = policy[s]
            if Q[s][a] != init_Q: # to plot policy of only updated states
                t, i, j = s
                xtr.append(g.xs[j])
                ytr.append(g.ys[g.ni - 1 - i])
                # print("test", s, a_policy)
                ax, ay = action_to_quiver(a)
                ax_list.append(ax)
                ay_list.append(ay)
                # print(i,j,g.xs[j], g.ys[g.ni - 1 - i], ax, ay)

        plt.quiver(xtr, ytr, ax_list, ay_list)
        ax1.scatter(g.xs[g.start_state[2]], g.ys[g.ni - 1 - g.start_state[1]], c='g')
        ax1.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')
        if showfig == True:
            plt.show()
        fig1.savefig(full_file_path + '_policy_plot.png', dpi=300)



    if DP_params != None:
        policy, full_file_path = DP_params
        policy_plot_folder = createFolder(join(full_file_path,'policy_plots'))

        for tt in range(g.nt-1):
            ax_list =[]
            ay_list = []
            xtr = []
            ytr = []

            for s in g.ac_state_space(time=tt):
                a = policy[s]
                t, i, j = s
                xtr.append(g.xs[j])
                ytr.append(g.ys[g.ni - 1 - i])
                # print("test", s, a_policy)
                ax, ay = action_to_quiver(a)
                ax_list.append(ax)
                ay_list.append(ay)

            plt.quiver(xtr, ytr, ax_list, ay_list)
            ax1.scatter(g.xs[g.start_state[2]], g.ys[g.ni - 1 - g.start_state[1] ], c='g')
            ax1.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')
            if showfig ==True:
                plt.show()
            fig1.savefig(full_file_path + '/policy_plots/policy_plot_t' + str(tt), dpi=300)
            plt.clf()
            fig1.clf()

    return