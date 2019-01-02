import pandas as pd
import numpy as np
import tensorflow as tf
import copy
import math
import random


# Basic classes and instance data input
class QNet:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 7])
        self.y1 = tf.layers.dense(self.x, units=14, activation=tf.nn.relu)
        self.y2 = tf.layers.dense(self.y1, units=28, activation=tf.nn.relu)
        self.y3 = tf.layers.dense(self.y2, units=14, activation=tf.nn.relu)
        self.y_pred = tf.layers.dense(self.y3, units=1, activation=tf.nn.relu)
        self.y_true = tf.placeholder(tf.float32)
        self.loss = tf.losses.mean_squared_error(labels=self.y_true, predictions=self.y_pred)
        self.optimizer = tf.train.AdamOptimizer(0.01)

    # Initialize the model in terms of variables and save it
    def initialize_variables(self):
        # Call the optimizer as an initialization, or it will not be saved in the checkpoint file
        train = self.optimizer.minimize(self.loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().save(sess, "./model.ckpt")

    # Train the model
    def train(self, inputs, outputs):
        train = self.optimizer.minimize(self.loss)

        with tf.Session() as sess:
            tf.train.Saver().restore(sess, "./model.ckpt")

            for i in range(50):
                _, loss_value = sess.run((train, self.loss), feed_dict={self.x: inputs,
                                                                        self.y_true: outputs})
            tf.train.Saver().save(sess, "./model.ckpt")

    # Predict the Q value of a pair of state and action based on the current model
    def predict(self, inputs):
        with tf.Session() as sess:
            tf.train.Saver().restore(sess, "./model.ckpt")
            return sess.run(self.y_pred, feed_dict={self.x: inputs})


class Nurses:
    # Use '__slots__' to save memory
    __slots__ = ['l', 's', 'tt', 'twt', 'avg_w', 'r', 'sd', 'aT']

    def __init__(self, label, skill):
        self.l = label
        self.s = skill
        # Total workload
        self.tt = 0
        # Total waiting time
        self.twt = 0
        # Average waiting time by job
        self.avg_w = 0
        # Visiting targets in line
        self.r = []
        # Fulfilled demand numbers by level
        self.sd = [0, 0, 0]
        # Arrival time at every target
        self.aT = []


class Jobs:
    # Use '__slots__' to save memory
    __slots__ = ['l', 'e', 'lv', 'c', 'twb', 'twe']

    def __init__(self, label, elder, level, coordinate_x, coordinate_y, coordinate_z,
                 time_window_begin, time_window_end):
        self.l = label
        self.e = elder
        self.lv = level
        self.c = [coordinate_x, coordinate_y, coordinate_z]
        self.twb = time_window_begin
        self.twe = time_window_end


def get_data(f_index, f_scale, jobs):
    # Setup statement
    file = './Instances/Elders_' + f_index + '.xlsx'

    # Temporal lists
    elders_index = []
    job_num = []
    job_level = []
    elder_location = []
    job_coordinate = np.zeros((f_scale, 3))
    time_window = np.zeros((f_scale, 2))

    # Read out column 'JobNum', 'Indexes', and 'JobLevel' from excel file
    excel = pd.read_excel(file, sheet_name='Sheet1')
    job_num.append(0)
    job_num += (list(copy.deepcopy(excel['JobNum'].values)))
    elders_index.append(0)
    elders_index += (list(copy.deepcopy(excel['Indexes'].values)))
    job_level.append(0)
    job_level += (list(copy.deepcopy(excel['JobLevel'].values)))

    # The first job is defined as the depot with coordinate (125, 125, 0)
    job_coordinate[0][0] = 125
    job_coordinate[0][1] = 125
    job_coordinate[0][2] = 0
    time_window[0][0] = 0.00
    time_window[0][1] = 480.00

    # Read out coordinates and time windows
    xyz = np.vstack((copy.deepcopy(excel['X'].values), copy.deepcopy(excel['Y'].values), copy.deepcopy(excel['Z'].values)))
    for i in range(len(xyz[0])):
        job_coordinate[i+1][0] = xyz[0][i]
        job_coordinate[i+1][1] = xyz[1][i]
        job_coordinate[i+1][2] = xyz[2][i]
    tw = np.vstack((copy.deepcopy(excel['TWB'].values), copy.deepcopy(excel['TWE'].values)))
    for i in range(len(tw[0])):
        time_window[i+1][0] = tw[0][i]
        time_window[i+1][1] = tw[1][i]

    # Read out locations labelled by elders for computing distance matrix
    lo = []
    for i in range(f_scale):
        lo.append([elders_index[i], job_coordinate[i][0], job_coordinate[i][1], job_coordinate[i][2]])
    for i in range(f_scale):
        if lo[i] in elder_location:
            continue
        else:
            elder_location.append(lo[i])

    # Build job classes and stack them into a list
    for fs in range(f_scale):
        jobs.append(
            Jobs(fs, elders_index[fs], job_level[fs], job_coordinate[fs][0], job_coordinate[fs][1],
                 job_coordinate[fs][1], time_window[fs][0], time_window[fs][1]))

    # Build distance matrix and return it
    num = len(elder_location)
    distance = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            hD = math.sqrt(pow((elder_location[i][1] - elder_location[j][1]), 2) + pow(
                (elder_location[i][2] - elder_location[j][2]), 2))
            if hD == 0:
                distance[i][j] = distance[j][i] = 9.6 * abs(elder_location[i][3] - elder_location[j][3])
            else:
                distance[i][j] = distance[j][i] = hD + elder_location[i][3] + elder_location[j][3]
    return distance


# Q Learning
def state_identification(target_set, skill_set):
    dn = count_demand_num(target_set)
    if len(skill_set) != 0:
        d1 = dn[0]
        d2 = dn[1]
        d3 = dn[2]
        if sum(dn) != 0:
            return [d1, d2, d3, str(skill_set).count('1'), str(skill_set).count('2'), str(skill_set).count('3')]
        else:
            return 1    # 1-no more demands
    else:
        return 2   # 2-no more nurses


def action_taking(state, q_net, skill_set, greedy):
    # Generate a constant randomly
    g = random.uniform(0, 1)
    if g < greedy:
        q_values = []
        # Act according to maximum q value
        for i in range(3):
            # Get Q values by feeding action and state pairs to the value Network
            _ = q_net.predict(np.array([np.append(state, [i+1])]))
            q_values.append(_[0])
        # Return chosen skill with the highest Q value
        s1 = np.argmax(q_values)
        if s1 + 1 in skill_set:
            skill_set.remove(s1 + 1)
            return s1 + 1
        else:
            q_values[s1] = -1
            s2 = np.argmax(q_values)
            if s2 + 1 in skill_set:
                skill_set.remove(s2 + 1)
                return s2 + 1
            else:
                q_values[s2] = -1
                s3 = np.argmax(q_values)
                skill_set.remove(s3 + 1)
                return s3 + 1
    else:
        # Act randomly as exploration
        skill = copy.deepcopy(skill_set)
        random.shuffle(skill)
        action = skill[0]
        skill_set.remove(action)
        return action


def count_demand_num(target_set):
    d = [0, 0, 0]
    for j in range(len(target_set)):
        if target_set[j].lv == 1:
            d[0] += 1
        if target_set[j].lv == 2:
            d[1] += 1
        if target_set[j].lv == 3:
            d[2] += 1
    return d


def q_learning(q_nurses_list, target_set, skill_set, q_net, q_learning_para,
               # ACO variables
               ccp_para, aco_para, changeable_pheromone_matrix,
               changeable_preference_matrix, e_init_distance_matrix, e_service_time_mean,
               depot):
    # Start a new QL episode
    discount = q_learning_para[0][0]
    greedy = q_learning_para[0][1]
    q_sub_total_workload = 0
    q_sub_total_waiting = 0
    q_sub_total_reward = 0
    for n in range(len(skill_set)):
        # Identify current state
        current_state = state_identification(target_set, skill_set)
        if current_state == 1 or current_state == 2:
            # reach absorbing state
            break
        # Take action according to e-greedy
        chosen_skill = action_taking(current_state, q_net, skill_set, greedy)

        # Create nurse object in a list sequentially
        q_nurses_list.append(Nurses(n, chosen_skill))
        current_demand_num = count_demand_num(target_set)

        # Collect targets according to skill demand match rule
        sd_matched_targets = []
        for aj in range(len(target_set)):
            if target_set[aj].lv <= q_nurses_list[n].s:
                sd_matched_targets.append(target_set[aj])

        # Build route by ACO algorithm
        aco(ccp_para, aco_para, changeable_pheromone_matrix,
            changeable_preference_matrix, e_init_distance_matrix, e_service_time_mean,
            q_nurses_list[n], sd_matched_targets, depot)

        # Remove fulfilled demands
        for o in range(len(q_nurses_list[n].r)):
            if q_nurses_list[n].r[o].l != 0:
                for b in range(len(target_set)):
                    if target_set[b].l == q_nurses_list[n].r[o].l:
                        target_set.remove(target_set[b])
                        break

        # Calculate fulfilled demands
        # Calculate reward: reward > 0 only if some demands are fulfilled
        # Reward function updated on 16/09/2018, waiting parameter was changed
        # Reward function updated on 19/09/2018, equals to the amount of fulfilled demands
        remaining_demands = count_demand_num(target_set)
        reward = sum([x-y for x,y in zip(current_demand_num, remaining_demands)])
        next_state = state_identification(target_set, skill_set)
        # If the next state is absorbing state, then set the next max q value to 0
        if next_state == 1 or 2:
            next_max_q = 0
        else:
            _ = []
            for i in range(3):
                # Get Q values by feeding action and state pairs to the value Network
                _.append(q_net.predict(np.array([np.append(next_state, [i+1])]))[0])
            next_max_q = np.max(_)
        # Pass state-action pair and target value to the network
        q_value = np.array([[reward + discount * next_max_q]])
        state_action_pair = np.array([np.append(current_state, chosen_skill)])
        q_net.train(state_action_pair, q_value)

        # Accumulate total workload
        q_sub_total_workload += q_nurses_list[n].tt
        q_sub_total_waiting += q_nurses_list[n].twt
        q_sub_total_reward += reward

    # Return total workload, waiting time and total reward for this episode
    return [q_sub_total_workload, q_sub_total_waiting, q_sub_total_reward]


# ACO realization
def collect_feasible_targets(visiting_list, distance_matrix, walk_speed, waiting_limit, current_job, current_time):
    ft = []
    for j in range(len(visiting_list)):
        distance = distance_matrix[current_job.e][visiting_list[j].e]
        travel = distance / walk_speed
        arrival = current_time + travel
        # Arrival time must be earlier than the upper bound
        # and later than the maximum waiting time + lower bound
        if arrival < visiting_list[j].twe:
            if (arrival + waiting_limit) >= visiting_list[j].twb:
                ft.append(visiting_list[j])
                continue
        else:
            continue
    return ft


def choose_target_deterministically(pr):
    p_axi = []
    for pta in range(len(pr)):
        p_axi.append(pr[pta][2])
    max_ind = np.argmax(p_axi)
    return max_ind


def choose_target_randomly(pr):
    p_coor = 0
    p_axi = [0]

    for pta in range(len(pr)):
        p_coor += pr[pta][2]
        p_axi.append(p_coor)
    # generate a random value
    ran_var = random.uniform(0, p_coor)
    search = (len(p_axi) // 2) - 1

    while True:
        if search <= 0:
            return 0
        if ran_var <= p_axi[search]:
            if ran_var > p_axi[search - 1]:
                return search - 1
            else:
                search -= 1
        else:
            try:
                _ = p_axi[search + 1]
            except:
                print("error: search = " + str(search))
                print(p_axi)
            else:
                if ran_var <= p_axi[search + 1]:
                    return search
                else:
                    search += 1


def calculate_transition_probability(feasible_targets, current_time, distance_matrix, current_job, walk_speed,
                                     sub_arrival_time, ant_path_table, visiting_list, pheromone_matrix,
                                     alpha_aco_p, beta_aco_p,
                                     depot):
    # Count feasible targets
    # =0: return depot as the next target
    # =1: return it as the next target
    # >2: return the target chosen according to probability transition function
    if (len(feasible_targets)) == 0:
        # No feasible targets, end routing
        current_time += (distance_matrix[current_job.e][depot.e]) / walk_speed
        sub_arrival_time.append(copy.deepcopy(current_time))  # record arrival time back to depot
        ant_path_table.append(depot)
        return depot
    elif len(feasible_targets) == 1:
        # Only one feasible target, choose it and update route
        ant_path_table.append(feasible_targets[0])
        current_time += (distance_matrix[current_job.e][feasible_targets[0].e]) / walk_speed
        # Remove chosen target from visiting list
        for v in range(len(visiting_list)):
            if visiting_list[v].l == feasible_targets[0].l:
                visiting_list.remove(visiting_list[v])
                return feasible_targets[0]
    else:
        # More than 1 feasible targets, calculate transition probabilities
        pr = []
        pD = 0
        for pdd in range(len(feasible_targets)):
            yitaD = distance_matrix[current_job.e][feasible_targets[pdd].e]
            pD += pow((pheromone_matrix[current_job.e][feasible_targets[pdd].e]), alpha_aco_p) \
                  * pow(yitaD, beta_aco_p)
        for pt in range(len(feasible_targets)):
            yitaU = distance_matrix[current_job.e][feasible_targets[pt].e]
            pU = pow((pheromone_matrix[current_job.e][feasible_targets[pt].e]), alpha_aco_p) \
                 * pow(yitaU, beta_aco_p)
            pT = pU / pD
            pr.append([current_job, feasible_targets[pt], pT])
            if math.isnan(pT):
                print(pT)
        # Choose target randomly and update route
        target_index = choose_target_deterministically(pr)
        ant_path_table.append(pr[target_index][1])
        current_time += (distance_matrix[current_job.e][pr[target_index][1].e]) / walk_speed
        # Remove chosen target from visiting list
        for v in range(len(visiting_list)):
            if visiting_list[v].l == pr[target_index][1].l:
                visiting_list.remove(visiting_list[v])
                break
        return pr[target_index][1]


def update_pheromone(best_path, worst_path, pheromone_matrix, rho_aco_p, distance_matrix):
    # update pheromone according to Best-Worst rule
    for bP in range(len(best_path)):
        if (bP + 1) == len(best_path):
            break
        else:
            pheromone_matrix[best_path[bP].e][best_path[bP + 1].e] \
                = (1 - rho_aco_p) * pheromone_matrix[best_path[bP].e][best_path[bP + 1].e] \
                  + distance_matrix[best_path[bP].e][best_path[bP + 1].e]
    for wP in range(len(worst_path)):
        if (wP + 1) == len(worst_path):
            break
        else:
            pheromone_matrix[worst_path[wP].e][worst_path[wP + 1].e] \
                = (1 - rho_aco_p) * pheromone_matrix[worst_path[wP].e][worst_path[wP + 1].e] \
                  - distance_matrix[worst_path[wP].e][worst_path[wP + 1].e]


def aco(ccp_para, aco_para, pheromone_matrix,
        preference_matrix, distance_matrix, service_time_mean,
        nurse, sd_matched_targets, depot):
    # Read parameters
    # ant_number alpha beta rho initialPheromone
    ant_num = aco_para[0][0]
    alpha_aco_p = aco_para[0][1]
    beta_aco_p = aco_para[0][2]
    rho_aco_p = aco_para[0][3]
    # acquaintance_increment waiting workload alpha beta walk_speed
    acquaintance_increment = ccp_para[0][0]
    waiting_limit = ccp_para[0][1]
    workload = ccp_para[0][2]
    alpha_model_p = ccp_para[0][3]
    beta_model_p = ccp_para[0][4]
    walk_speed = ccp_para[0][5]
    # Output variables
    arrival_time_trace = []
    shortest_time = 0
    waiting_time = 0
    best_path = []
    # Temporal variables
    worst_path = []
    ccp_best_objective = 0
    ccp_worst_objective = 0

    for ant in range(ant_num):
        # Initialization: depot, time, waiting time, workload
        current_job = depot
        current_time = 0
        current_waiting = 0
        current_workload = 0
        # Lists for recording sub-arrival time and path
        sub_arrival_time = []
        ant_path_table = [depot]
        # Initialize visiting list and preference matrix
        visiting_list = copy.deepcopy(sd_matched_targets)
        current_preference = copy.deepcopy(preference_matrix)

        # Build routes
        while current_workload <= workload:
            # Read out service time mean value and preference value
            st_mean = service_time_mean[nurse.s][current_job.lv]
            preference_factor = copy.deepcopy(current_preference[current_job.e][nurse.l])

            # Inspect waiting and record sub-arrival time
            if current_time < current_job.twb:
                current_waiting += (current_job.twb - current_time)
                current_time = copy.deepcopy(current_job.twb)
                sub_arrival_time.append(copy.deepcopy(current_time))
            else:
                sub_arrival_time.append(copy.deepcopy(current_time))

            # Compute arrival time as predicted workload when going back to depot at current position
            # Then check if overwork occurs
            current_workload = current_time + (preference_factor * st_mean)\
                               + beta_model_p + (distance_matrix[current_job.e][depot.e]) / walk_speed
            if current_workload >= workload:
                # Overwork predicted, stop routing
                # Set depot as the next target and record arrival time
                ant_path_table.append(depot)
                current_time = copy.deepcopy(current_workload)
                sub_arrival_time.append(copy.deepcopy(current_time))
                break
            else:
                # Continue routing
                # Add up service time
                current_time += (preference_factor * st_mean) + alpha_model_p

            # Search for targets satisfying the time window constraint
            feasible_targets = collect_feasible_targets(visiting_list, distance_matrix, walk_speed, waiting_limit,
                                                        current_job, current_time)
            # Count feasible targets, calculate transition probabilities and choose target
            chosen_target = calculate_transition_probability(feasible_targets, current_time, distance_matrix,
                                                             current_job, walk_speed, sub_arrival_time, ant_path_table,
                                                             visiting_list, pheromone_matrix, alpha_aco_p, beta_aco_p,
                                                             depot)
            if chosen_target.l == 0:
                # No feasible target, back to depot, stop routing
                break
            else:
                # Feasible target chosen, continue
                current_job = chosen_target
                # Revise preference
                current_preference[chosen_target.e][nurse.l] -= acquaintance_increment
                continue

        # Calculate fulfilled demands
        fulfilled_demand = copy.deepcopy(len(ant_path_table) - 2)
        # Record the best and worst solution according to the CCP objective
        if fulfilled_demand == 0:
            # no fulfilled demand
            if len(best_path) == 0:
                best_path = copy.deepcopy(ant_path_table)
                worst_path = copy.deepcopy(ant_path_table)
        else:
            # record current PPM objective: total waiting time
            ccp_objective = copy.deepcopy(current_waiting)
            if ant == 0:
                # first iteration, record best CCP objective, worst PPM objective, working time,
                # waiting time, best route, and worst route
                ccp_best_objective = copy.deepcopy(ccp_objective)
                ccp_worst_objective = copy.deepcopy(ccp_objective)
                shortest_time = copy.deepcopy(current_time)
                waiting_time = copy.deepcopy(current_waiting)
                best_path = copy.deepcopy(ant_path_table)
                worst_path = copy.deepcopy(ant_path_table)
                arrival_time_trace = copy.deepcopy(sub_arrival_time)
            else:  # not first iteration
                if ccp_best_objective > ccp_objective:  # find the best one
                    ccp_best_objective = copy.deepcopy(ccp_objective)
                    shortest_time = copy.deepcopy(current_time)
                    waiting_time = copy.deepcopy(current_waiting)
                    best_path = copy.deepcopy(ant_path_table)
                    arrival_time_trace = copy.deepcopy(sub_arrival_time)
                elif ccp_worst_objective < ccp_objective:  # find the worst one
                    ccp_worst_objective = copy.deepcopy(ccp_objective)
                    worst_path = copy.deepcopy(ant_path_table)
                else:
                    continue

        # update pheromone according to Best-Worst rule
        update_pheromone(best_path, worst_path, pheromone_matrix, rho_aco_p, distance_matrix)

    # update route
    nurse.tt = copy.deepcopy(shortest_time)
    nurse.aT = copy.deepcopy(arrival_time_trace)
    nurse.twt = copy.deepcopy(waiting_time)
    for o in range(len(best_path)):
        nurse.r.append(best_path[o])
        if best_path[o].lv == 1:
            nurse.sd[0] += 1
        elif best_path[o].lv == 2:
            nurse.sd[1] += 1
        elif best_path[o].lv == 3:
            nurse.sd[2] += 1
    if sum(nurse.sd) != 0:
        nurse.avg_w = float('%.2f' % (copy.deepcopy(nurse.twt / sum(nurse.sd))))