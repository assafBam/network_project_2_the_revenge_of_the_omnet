NEW_DELY = False
OLD_PROJ_NEW_GRAPH = True

import math
import numpy as np
import networkx as nx
import gym
from gym import spaces

if NEW_DELY:
    import new_dely as dely
elif OLD_PROJ_NEW_GRAPH:
    import fermi_dely as dely
else:
    import dely


def rl_state(env):
    return env.env_T


def uniform_traffic():
    if NEW_DELY or OLD_PROJ_NEW_GRAPH:
        T = np.random.uniform(0.1, 1, [10, 10]) / 9
    else:
        T = np.random.uniform(0.1, 1, [14, 14]) / 13
    return T


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class drl_agent_env(gym.Env):

    def __init__(self):
        if NEW_DELY:
            self.kpi=dely.new_delay('Network_nsfnetbw.ned','graph.txt')
        else:
            self.kpi = dely.delay('Network_nsfnetbw.ned')

        if NEW_DELY or OLD_PROJ_NEW_GRAPH:
            self.ACTIVE_NODES = 10
        else:
            self.ACTIVE_NODES = 14

        topology = 'NetworkAll.matrix'
        self.graph = nx.Graph(np.loadtxt(topology, dtype=int))
        print(self.graph)

        ports = 'NetworkAll.ports'
        self.ports = np.loadtxt(ports, dtype=int)

        self.a_dim = self.graph.number_of_edges()

        self.s_dim = self.ACTIVE_NODES ** 2 - self.ACTIVE_NODES

        self.action_space = spaces.Box(low=np.zeros([21]),
                                       high=np.ones([21]), dtype=np.float32)

        self.observation_space = spaces.Box(low=np.zeros([182]),
                                            high=np.full([182], fill_value=2),
                                            dtype=np.float32)

        self.tgen = uniform_traffic  # traffic matrix generator, IMPORTANT: MUST BE SAME DISTRUBTION AS ROUTENET

        self.env_T = np.full([182], -1.0, dtype=float)
        self.env_W = np.full([self.a_dim], -1.0, dtype=float)  # weights
        self.env_R = np.full([self.ACTIVE_NODES] * 2, -1.0, dtype=int)  # routing

        self.counter = 0
        self.reset()

    def upd_env_T(self, matrix):
        A = np.asarray(matrix)
        self.old_env_T_rout = (A[~np.eye(len(A), dtype=bool)].reshape(len(A), -1)).flatten()
        if NEW_DELY:
            traffic_file = "traffic.txt"
            with open(traffic_file, "w") as tm_fd:
                for src, dsts in enumerate(matrix):
                    for dst, bw in enumerate(dsts):
                        avg_bw = int(np.round(bw * 10000 / (self.counter % 9 + 8)))  # NOTE: 625 was before that 10000 but now we try 10000/16 = 625
                        time_dist = 1
                        # on_time = 5
                        # off_time = 10
                        pkt_size_1 = 1000  # TODO: ?
                        prob_1 = 1
                        # pkt_size_2 = 1700
                        # prob_2 = 0.5
                        tos = 0
                        traffic_line = f"{src},{dst},{avg_bw},{time_dist},0,{pkt_size_1},{prob_1},{tos}"
                        tm_fd.write(traffic_line + "\n")
            self.env_T_rout = traffic_file

        else:
            self.env_T_rout = self.old_env_T_rout

    def upd_env_W(self, vector):
        self.env_W = np.asarray(softmax(vector))

    def upd_env_R(self):
        weights = {}

        for e, w in zip(self.graph.edges(), self.env_W):
            weights[e] = w

        nx.set_edge_attributes(self.graph, 'weight', weights)

        routing_nodes = np.full([self.ACTIVE_NODES] * 2, -1.0, dtype=int)
        routing_ports = np.full([self.ACTIVE_NODES] * 2, -1.0, dtype=int)

        APSP = nx.all_pairs_dijkstra_path(self.graph)  # calculate all pairs shortest paths based on agent weights


        if NEW_DELY:
            # creating routing file
            routing_file = 'routing.txt'
            with open(routing_file, "w") as r_fd:
                # lPaths = nx.shortest_path(G)
                for src in APSP:
                    for dst, path in APSP[src].items():
                        if src == dst:
                            continue
                        r_fd.write(f'{",".join(str(node) for node in path)}\n')
            self.env_R = routing_file  # TODO: check if this is working
        else:
            # all we need as APSP (to generate from it the routing file), can ignore all of this nonsense
            for s in range(self.ACTIVE_NODES):
                for d in range(self.ACTIVE_NODES):
                    if s != d:
                        next1 = APSP[s][d][1]
                        port = self.ports[s][next1]
                        routing_nodes[s][d] = next1
                        routing_ports[s][d] = port
                    else:
                        routing_nodes[s][d] = -1
                        routing_ports[s][d] = -1

            self.env_R = np.asarray(routing_ports)

    def reset(self, easy=False):
        self.counter = 0
        self.upd_env_W(np.full([self.a_dim], 0.5, dtype=float))
        self.upd_env_R()
        traffic_matrix = self.tgen() * 8  # 8 is our default intensity
        self.upd_env_T(traffic_matrix)

        return rl_state(self)

    def step(self, action):
        self.counter += 1

        self.upd_env_W(action)
        self.upd_env_R()
        delay = self.kpi.calc(self.env_T_rout, self.env_R)[0]
        reward = -np.mean(delay) * 0.015 / np.mean(self.old_env_T_rout)
        traffic_matrix = self.tgen() * (self.counter % 9 + 8)  # traffic with intensity 8-16
        self.upd_env_T(traffic_matrix)
        new_state = rl_state(self)
        done = False
        if self.counter >= 90:  # episode length
            done = True
        return new_state, reward, done, {}


def end(self):
    return
