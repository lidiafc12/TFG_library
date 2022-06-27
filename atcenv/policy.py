"""
Policy module
"""
import math
from collections import deque
import gym
from typing import List
from atcenv.definitions import *
from atcenv.env import *
from atcenv import Environment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def position_angles(env, i: int, j: int) -> tuple:
    """ """
    """ Respect to flight: i """
    dx_i = env.flights[j].position.x - env.flights[i].position.x
    dy_i = env.flights[j].position.y - env.flights[i].position.y
    compass_i = math.atan2(dx_i, dy_i)
    compass_i = (compass_i + u.circle) % u.circle

    angle_i = compass_i - env.flights[i].track
    if angle_i > math.pi:
        angle_i = -(u.circle - angle_i)
    elif angle_i < -math.pi:
        angle_i = u.circle + angle_i

    """ Respect to flight: j """
    dx_j = env.flights[i].position.x - env.flights[j].position.x
    dy_j = env.flights[i].position.y - env.flights[j].position.y
    compass_j = math.atan2(dx_j, dy_j)
    compass_j = (compass_j + u.circle) % u.circle

    angle_j = compass_j - env.flights[j].track
    if angle_j > math.pi:
        angle_j = -(u.circle - angle_j)
    elif angle_j < -math.pi:
        angle_j = u.circle + angle_j

    return angle_i, angle_j
    ##########################################################


def t_cpa(env, i: int, j: int) -> float:
    """
    Time to get the closest point of approach of a flight i to j
    :return: time to the closest point of approach, done with straight formula
    """
    dx = env.flights[j].position.x - env.flights[i].position.x
    dy = env.flights[j].position.y - env.flights[i].position.y

    """ Computing relative velocity contemplating bearing """
    # vrx = (env.flights[j].airspeed * math.sin(env.flights[j].bearing)) - (env.flights[i].airspeed * math.sin(env.flights[i].bearing))
    # vry = (env.flights[j].airspeed * math.cos(env.flights[j].bearing)) - (env.flights[i].airspeed * math.cos(env.flights[i].bearing))

    """ Computing relative velocity contemplating track --> x """
    vrx = env.flights[j].components[0] - env.flights[i].components[0]
    vry = env.flights[j].components[1] - env.flights[i].components[1]

    if i == j:
        tcpa = 0
    else:
        tcpa = max(0, -(dx * vrx + dy * vry) / (vrx ** 2 + vry ** 2))

    return tcpa
    ##########################################################


def d_cpa(env, i: int, j: int) -> float:
    """
    Distance to get the closest point of approach of a flight i to j
    :return: distance to the closest point of approach
    """
    dx = env.flights[j].position.x - env.flights[i].position.x
    dy = env.flights[j].position.y - env.flights[i].position.y
    vrx = env.flights[j].components[0] - env.flights[i].components[0]
    vry = env.flights[j].components[1] - env.flights[i].components[1]
    if i == j:
        dcpa = 0
    else:
        tcpa = t_cpa(env, i, j)
        dcpa = math.sqrt((dx + vrx * tcpa) ** 2 + (dy + vry * tcpa) ** 2)

    return dcpa
    ##########################################################


def t_cpa_bearing(env, i: int, j: int) -> float:
    """
    Time to get the closest point of approach of a flight i to j following its bearing
    :return: time to the closest point of approach, done with straight formula
    """
    track_i = env.flights[i].track + env.flights[i].drift
    track_j = env.flights[j].track + env.flights[j].drift

    """ Computing relative velocity contemplating bearing """
    dx = env.flights[j].position.x - env.flights[i].position.x
    dy = env.flights[j].position.y - env.flights[i].position.y

    """ Computing relative velocity contemplating track --> x """
    vrx = env.flights[j].airspeed * math.sin(track_j) - env.flights[i].airspeed * math.sin(track_i)
    vry = env.flights[j].airspeed * math.cos(track_j) - env.flights[i].airspeed * math.cos(track_i)

    if i == j:
        tcpa_bearing = 0
    else:
        tcpa_bearing = max(0, -(dx * vrx + dy * vry) / (vrx ** 2 + vry ** 2))

    return tcpa_bearing
    ##########################################################


def d_cpa_bearing(env, i: int, j: int) -> float:
    """
    Distance to get the closest point of approach of a flight i to j following its bearing
    :return: distance to the closest point of approach
    """
    track_i = env.flights[i].track + env.flights[i].drift
    track_j = env.flights[j].track + env.flights[j].drift
    dx = env.flights[j].position.x - env.flights[i].position.x
    dy = env.flights[j].position.y - env.flights[i].position.y
    vrx = env.flights[j].airspeed * math.sin(track_j) - env.flights[i].airspeed * math.sin(track_i)
    vry = env.flights[j].airspeed * math.cos(track_j) - env.flights[i].airspeed * math.cos(track_i)

    if i == j:
        dcpa_bearing = 0
    else:
        tcpa = t_cpa_bearing(env, i, j)
        dcpa_bearing = math.sqrt((dx + vrx * tcpa) ** 2 + (dy + vry * tcpa) ** 2)

    return dcpa_bearing
    ##########################################################


def safe_turn_angle(env, i: int, j: int) -> float:
    angle_right = 0
    while angle_right < env.performance_limitation:
        track = env.flights[i].track + angle_right
        dx = env.flights[j].position.x - env.flights[i].position.x
        dy = env.flights[j].position.y - env.flights[i].position.y
        vrx = env.flights[j].components[0] - env.flights[i].airspeed * math.sin(track)
        vry = env.flights[j].components[1] - env.flights[i].airspeed * math.cos(track)
        tcpa = max(0, -(dx * vrx + dy * vry) / (vrx ** 2 + vry ** 2))
        dcpa = math.sqrt((dx + vrx * tcpa) ** 2 + (dy + vry * tcpa) ** 2)

        tol = env.alert_distance - dcpa
        if tol < 0 or tcpa == 0:
            break
        angle_right = angle_right + (1 * (u.circle / 360))

    angle_left = 0
    while angle_left > - env.performance_limitation:
        track = env.flights[i].track + angle_left
        dx = env.flights[j].position.x - env.flights[i].position.x
        dy = env.flights[j].position.y - env.flights[i].position.y
        vrx = env.flights[j].components[0] - env.flights[i].airspeed * math.sin(track)
        vry = env.flights[j].components[1] - env.flights[i].airspeed * math.cos(track)
        tcpa = max(0, -(dx * vrx + dy * vry) / (vrx ** 2 + vry ** 2))
        dcpa = math.sqrt((dx + vrx * tcpa) ** 2 + (dy + vry * tcpa) ** 2)

        tol = env.alert_distance - dcpa
        if tol < 0 or tcpa == 0:
            break
        angle_left = angle_left - (1 * (u.circle / 360))

    """ Choosing whether turn right or turn left """
    if angle_right != env.performance_limitation and angle_left != env.performance_limitation:
        # Computing right track and verify: 0 < angle < 2*phi
        track_right = env.flights[i].track + angle_right
        if track_right > u.circle:
            track_right = track_right - u.circle
        # Computing left track and verify: 0 < angle < 2*phi
        track_left = env.flights[i].track + angle_left
        if track_left < 0:
            track_left = track_left + u.circle
        # Computing...
        dif_right = abs(track_right - env.flights[j].track)
        dif_left = abs(track_left - env.flights[j].track)

        if dif_right > dif_left:
            angle_safe_turn = angle_right
        else:
            angle_safe_turn = angle_left
    else:
        angle_safe_turn = min(abs(angle_right), abs(angle_left))
        if angle_safe_turn == abs(angle_left):
            angle_safe_turn = angle_left
    # When a solution is not encountered
    if angle_right == env.performance_limitation and angle_left == env.performance_limitation:
        print('FLight ', i, 'has not encountered a safe turn angle')

    return angle_safe_turn
    ##########################################################


def safe_turn_MULTIPLE_angle(env, angle_safe_turn, list_i, i) -> float:

    """ The action of i will be the maximum between angles of the closest flight and others """
    previous_angle = angle_safe_turn
    n = 1
    while n < len(list_i):
        k = list_i[n]
        angle_safe_turn_multiple = safe_turn_angle(env, i, k)

        max_multiple_angle = max(abs(previous_angle), abs(angle_safe_turn_multiple))
        if max_multiple_angle == abs(previous_angle):
            max_multiple_angle = previous_angle
        if max_multiple_angle == abs(angle_safe_turn_multiple):
            max_multiple_angle = angle_safe_turn_multiple

        previous_angle = max_multiple_angle
        n += 1

    return max_multiple_angle
    ##########################################################


def alert_detection(env, previous_distances, previous_actions, current_distances, FirstStepConflict, InConflict) -> List:

    """ Creating a matrix with ALL flights and its closest conflicts ordered by time, and actualizing conflict parameters """
    FlightsInConflictWith = []
    for i in range(env.num_flights):
        num = []
        time = []
        for j in range(env.num_flights):
            if i not in env.done and j not in env.done and i != j:
                tcpa = t_cpa(env, i, j)
                dcpa = d_cpa(env, i, j)
                tcpa_bearing = t_cpa_bearing(env, i, j)
                dcpa_bearing = d_cpa_bearing(env, i, j)
                if (tcpa < env.alert_time and dcpa < env.alert_distance < previous_distances[i, j]) or (
                        current_distances[i, j] < env.alert_distance < previous_distances[i, j]) or (previous_actions[i] == env.performance_limitation) or (
                        current_distances[i, j] < env.alert_distance and current_distances[i, j] <= previous_distances[i, j]):
                    InConflict[i] = True
                    FirstStepConflict[i] = True
                    FirstStepConflict[j] = True
                    time.append(tcpa)
                    num.append(j)
                else:
                    if (tcpa_bearing < env.alert_time and dcpa_bearing < env.alert_distance) or current_distances[i, j] < env.alert_distance:
                        InConflict[i] = True
                        time.append(tcpa_bearing)
                        num.append(j)

        tcpa_sorted = [x for _, x in sorted(zip(time, num))]
        FlightsInConflictWith.append(tcpa_sorted)

    return FlightsInConflictWith
    ##########################################################


def SERA_rules_application(env, i, j, angle_i, angle_j, actions, angle_safe_turn) -> None:

    """ Looking at the conditions and, in consequence, applying the actions"""
    approach = u.circle / 24  # 15º
    converge = (110 / 360) * u.circle  # 110º

    ###############
    # APPROACHING #
    ###############
    # When two aircraft are approaching head-on or approximately so and there is danger of
    # collision, each shall alter its heading to the right.
    if abs(angle_i) < approach and abs(angle_j) < approach:
        actions[i] = angle_safe_turn / 2
        actions[j] = angle_safe_turn / 2

    ##############
    # CONVERGING #
    ##############
    # When two aircraft are converging at approximately the same level, the aircraft that has
    # the other on its right shall give way.
    elif approach <= abs(angle_i) <= converge or approach <= abs(angle_j) <= converge:

        if angle_i > 0:
            if angle_j > 0:
                if env.flights[i].airspeed > env.flights[j].airspeed:
                    actions[i] = angle_safe_turn
            else:
                actions[i] = angle_safe_turn

        elif angle_i <= 0:
            if angle_j <= 0:
                if env.flights[i].airspeed > env.flights[j].airspeed:
                    actions[i] = angle_safe_turn

    ##############
    # OVERTAKING #
    ##############
    # An aircraft that is being overtaken has the right-of-way and the overtaking aircraft
    # shall keep out of the way of the other aircraft by altering its heading to the right, and no subsequent change in the relative positions of
    # the two aircraft shall absolve the overtaking aircraft from this obligation until it is entirely past and clear
    # In all circumstances, the faster flight that is overtaking shall give way
    elif abs(angle_j) > converge or abs(angle_i) > converge:

        if env.flights[i].airspeed > env.flights[j].airspeed:
            actions[i] = angle_safe_turn

    return None
    ##########################################################


""" POLICY DEFINITION """


def policy_action(memory, env) -> List:
    actions = [0] * env.num_flights
    FirstStepConflict = [False] * env.num_flights
    InConflict = [False] * env.num_flights
    previous_distances, previous_actions = memory.pop()
    current_distances = env.distances_matrix()

    FlightsInConflictWith = alert_detection(env, previous_distances, previous_actions, current_distances, FirstStepConflict, InConflict)

    """ For each flight i, the conflict between i and its closest flight j is solved modifying the action i """
    for i in range(env.num_flights):
        if i not in env.done:

            list_i = FlightsInConflictWith[i]
            if not InConflict[i]:
                """ NO CONFLICT """
                actions[i] = env.flights[i].drift

            if InConflict[i]:
                """ CONFLICT """
                if not FirstStepConflict[i]:
                    """ Solving the conflict, in process """
                    actions[i] = 0

                if FirstStepConflict[i]:
                    """ First step in the conflict """

                    """ Solving the most important conflict: The closest conflict in terms of time (tcpa) """
                    j = list_i[0]
                    angle_safe_turn = safe_turn_angle(env, i, j)

                    if env.airspace.polygon.contains(env.flights[i].position) and env.airspace.polygon.contains(
                            env.flights[j].position):

                        """ Computing the angle formed by the position of the intruder flight in respect of it's own track """
                        angle_i, angle_j = position_angles(env, i, j)

                        """ Applying the SERA Rules """
                        SERA_rules_application(env, i, j, angle_i, angle_j, actions, angle_safe_turn)

                    if len(list_i) > 1:
                        """ If there is a multiple conflict, is solved calculating the maximum angle that solves all """

                        actions[i] = safe_turn_MULTIPLE_angle(env, angle_safe_turn, list_i, i)

    return actions


class PrioritizedReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1 / len(self.buffer) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return map(list, zip(*samples)), importance, sample_indices

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset
