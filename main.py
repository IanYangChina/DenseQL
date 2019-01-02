import time
import copy
import numpy as np
import pandas as pd
import xlwt
import matplotlib.pyplot as plt
import DenseQL
import os
os.getcwd()


class Solution:
    __slots__ = ['nl', 'on', 'ev', 'rd', 'fd_sum',
                 'a_wait_j', 'a_wait_n', 't_wait', 'a_work_j', 'a_work_n', 't_work', 'wait_d_work',
                 'fig_t', 'info_to_excel']

    def __init__(self):
        # Nurse list
        self.nl = []
        # Occupied nurse number
        self.on = 0
        # Evaluation function value
        self.ev = -1000
        # Remaining demand number
        self.rd = []
        # Fulfilled demand number
        self.fd_sum = 0
        # Average waiting time of fulfilled job
        self.a_wait_j = 0
        # Average waiting time of occupied nurse
        self.a_wait_n = 0
        # Total waiting time
        self.t_wait = 0
        # Average service time of job
        self.a_work_j = 0
        # Average workload of occupied nurse
        self.a_work_n = 0
        # Total workload of occupied nurse
        self.t_work = 0
        # Percentage of waiting in workload
        self.wait_d_work = 0
        # Figure test
        self.fig_t = ''
        # Information to excel
        self.info_to_excel = []

    def calculate(self, q_learning_result, remaining_demand):
        self.t_work = float('%.2f' % (copy.deepcopy(q_learning_result[0])))
        self.t_wait = float('%.2f' % (copy.deepcopy(q_learning_result[1])))
        self.rd = DenseQL.count_demand_num(copy.deepcopy(remaining_demand))
        self.fd_sum = file_scale - sum(self.rd) - 1
        self.wait_d_work = float('%.2f' % (self.t_wait / self.t_work))
        # Calculate occupied nurses
        for o in range(len(self.nl)):
            if len(self.nl[o].r) > 2:
                self.on += 1
        if self.on != 0:
            self.a_wait_j = float('%.2f' % (self.t_wait / self.fd_sum))
            self.a_wait_n = float('%.2f' % (self.t_wait / self.on))
            self.a_work_j = float('%.2f' % (self.t_work / self.fd_sum))
            self.a_work_n = float('%.2f' % (self.t_work / self.on))
            # Evaluation function updated on 19/09/2018, depends on total fulfilled demands
            self.ev = self.fd_sum

    def get_solution_info(self):
        indexes = []
        skill_level = []
        workload = []
        waiting = []
        avg_waiting = []
        route_e = []
        route_j = []
        fulfilled_d = []
        # Stack solution information by columns
        for i in range(len(self.nl)):
            indexes.append(self.nl[i].l)
            skill_level.append(self.nl[i].s)
            workload.append(self.nl[i].tt)
            waiting.append(self.nl[i].twt)
            avg_waiting.append(self.nl[i].avg_w)
            e = []
            j = []
            for r in range(len(self.nl[i].r)):
                e.append(self.nl[i].r[r].e)
                j.append(self.nl[i].r[r].l)
            route_e.append(e)
            route_j.append(j)
            fulfilled_d.append(self.nl[i].sd)
        # return organized information as a list
        self.info_to_excel = \
            copy.deepcopy([indexes, skill_level, workload, waiting, avg_waiting, route_e, route_j, fulfilled_d])

    def save_solution_info(self, file_name):
        # save solution information into excel file
        exc = xlwt.Workbook()
        exc.add_sheet("Sheet1")
        exc.save(file_name)
        writer = pd.ExcelWriter(file_name, sheet_name='Sheet1')
        solution_df = pd.DataFrame({'Nurses': self.info_to_excel[0],
                                    'Skill': self.info_to_excel[1],
                                    'Workload': self.info_to_excel[2],
                                    'Waiting': self.info_to_excel[3],
                                    'Average waiting': self.info_to_excel[4],
                                    'Routes by elder': self.info_to_excel[5],
                                    'Routes by job': self.info_to_excel[6],
                                    'Fulfilled demands': self.info_to_excel[7]})
        solution_df.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.save()

    def create_fig_text(self, running_time):
        if running_time != 0:
            self.fig_t = "\n" \
                         "Running time: " + str(float('%.2f' % running_time)) + " mins \n" \
                         "Results of best solution found:\n" \
                         "Solution evaluation value = " + str(self.ev) + "\n" \
                         "Solution occupied nurses = " + str(self.on) + "\n" \
                         "Solution avg workload of nurse = " + str(self.a_work_n) + "\n" \
                         "Solution avg waiting time of job = " + str(self.a_wait_j) + "\n" \
                         "Solution avg waiting time of nurse = " + str(self.a_wait_n) + "\n" \
                         "Solution total workload = " + str(self.t_work) + "\n" \
                         "Solution total waiting time = " + str(self.t_wait) + "\n" \
                         "Solution waiting time proportion = " + str(self.wait_d_work) + "\n" \
                         "Solution remaining demands = " + str(self.rd)
        else:
            self.fig_t = "\n" \
                         "Results of testing the trained agent:\n" \
                         "Solution evaluation value = " + str(self.ev) + "\n" \
                         "Solution occupied nurses = " + str(self.on) + "\n" \
                         "Solution avg workload of nurse = " + str(self.a_work_n) + "\n" \
                         "Solution avg waiting time of job = " + str(self.a_wait_j) + "\n" \
                         "Solution avg waiting time of nurse = " + str(self.a_wait_n) + "\n" \
                         "Solution total workload = " + str(self.t_work) + "\n" \
                         "Solution total waiting time = " + str(self.t_wait) + "\n" \
                         "Solution waiting time proportion = " + str(self.wait_d_work) + "\n" \
                         "Solution remaining demands = " + str(self.rd)


def plot(figure_name, y_label, x_label, x, y, figure_text):
    y_lower = min(y) - 5
    y_upper = max(y) + 5
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xlim(0, x)
    plt.ylim(y_lower, y_upper)
    plt.plot(y)
    # Figure explanations
    plt.text(x-1, y_lower+1, figure_text, fontsize=9, va="baseline", ha="right")
    name = figure_name + '.png'
    plt.savefig(name, dpi=900, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    """------Instance representations------"""
    # Get elder data and distance matrix
    # File index = A,  B,  C,  D,   E
    # File scale = 61, 61, 56, 160, 304
    file_index = 'A'
    file_scale = 61
    e_jobs = []
    e_init_distance_matrix = DenseQL.get_data(file_index, file_scale, e_jobs)
    # Set up nurse resource: how many skill levels and how many nurses you would like to try
    # For example: 3 skill levels, and 2, 2, 3 nurses, respectively
    e_nurses_skill = [1, 1, 2, 2, 3, 3, 3]
    # Build initial preference matrix
    e_random_pre = np.loadtxt("./Instances/initial_preference_ABC.csv", delimiter=",", skiprows=0)
    e_preference_matrix = np.row_stack((np.zeros((1, len(e_nurses_skill))), e_random_pre))
    # Skill demand matching parameter matrix
    e_service_time_mean = np.array([[0, 0, 0, 0],
                                    [0, 25, -1, -1],
                                    [0, 20, 30, -1],
                                    [0, 18, 20, 20]])
    qn = DenseQL.QNet()
    # qn.initialize_variables()

    """Parameter settings"""
    # Q Learning
    # discount greedy
    para_ql = [[0.95, 0.5]]
    # Ant colony optimization
    # ant_number alpha beta rho initialPheromone
    para_aco = [[30, 1, 1, 0.1, 20]]
    aco_pheromone_matrix = np.ones((len(e_init_distance_matrix[0]), len(e_init_distance_matrix[0]))) * para_aco[0][0]
    # Chance Constrained Programming Model
    # confidence levels:        100%  95%   90%   80%   70%   60%   50%
    # inverse standard normal:  3.09  1.65, 1.29  0.85  0.52  0.26  0
    # acquaintance_increment waiting workload alpha beta walk_speed
    para_ccp = [[0.05, 10, 480, 1.29, 1.29, 60]]

    """---Solution Recording Variables---"""
    solution_final = Solution()

    """Figure data record"""
    axis_sub_evaluation_value_iter = []
    axis_avg_wait_nurse_iter = []
    axis_evaluation_value_iter = []
    axis_episode_reward_iter = []

    """---Start iteration---"""
    # Record the starting time of the training process
    start = time.time()
    iter = 0
    iter_max = 800
    print('Current Experimental Instance is ' + file_index)
    print('Instance Scale: ' + str(file_scale-1))
    print('Start training...')
    while iter < iter_max:
        iter += 1
        # Solution objective for current sub-solution
        sub_solution = Solution()
        # Changeable list of targets
        available_targets = copy.deepcopy(e_jobs)
        # Delete depot
        available_targets.remove(available_targets[0])
        # Changeable preference, pheromone matrix and nurse's skill set
        changeable_preference_matrix = copy.deepcopy(e_preference_matrix)
        changeable_pheromone_matrix = copy.deepcopy(aco_pheromone_matrix)
        changeable_nurses_skill = copy.deepcopy(e_nurses_skill)
        # Start Q Learning process
        ql_result = DenseQL.q_learning(sub_solution.nl, available_targets, changeable_nurses_skill, qn, para_ql,
                                       # ACO variables
                                       para_ccp, para_aco, changeable_pheromone_matrix,
                                       changeable_preference_matrix, e_init_distance_matrix, e_service_time_mean,
                                       e_jobs[0])
        print("Episode: " + str(iter) + ", total reward: " + str(ql_result[2]))
        # Calculate fulfilled and remaining demands
        sub_solution.calculate(ql_result, available_targets)
        # Update global solution according to evaluation value
        if solution_final.ev < sub_solution.ev:
            solution_final = copy.deepcopy(sub_solution)

        # Decaying greedy rate
        para_ql[0][1] *= 0.9985

        axis_avg_wait_nurse_iter.append(solution_final.a_wait_n)
        axis_evaluation_value_iter.append(solution_final.ev)
        axis_sub_evaluation_value_iter.append(sub_solution.ev)
        axis_episode_reward_iter.append(ql_result[2])

    # Record the ending time of the training process
    end = time.time()
    rt = (end - start) / 60
    # Create figure text for the solution
    solution_final.create_fig_text(rt)
    print(solution_final.fig_t)
    print("Final greedy rate: " + str(para_ql[0][1]))
    # Save detailed information into an excel file
    solution_final.get_solution_info()
    solution_final.save_solution_info("final solution.xls")
    # Plot the figures
    plot("iter - avg waiting of nurse", "avg waiting of nurse", "iteration",
         iter_max - 1, axis_avg_wait_nurse_iter, solution_final.fig_t)
    plot("iter - evaluation value", "evaluation value", "iteration",
         iter_max - 1, axis_evaluation_value_iter, solution_final.fig_t)
    plot("iter - sub evaluation value", "sub ev", "iteration",
         iter_max - 1, axis_sub_evaluation_value_iter, solution_final.fig_t)
    plot("iter - episode_reward", "episode_reward", "iteration",
         iter_max - 1, axis_episode_reward_iter, solution_final.fig_t)

