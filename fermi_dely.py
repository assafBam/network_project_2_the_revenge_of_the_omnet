import tensorflow as tf
import torch

import upcdataset as up
import numpy as np
import RouteNet_Fermi as rt


class delay:
    def __init__(self, nedfile):
        (self.connections,
         self.n,
         self.edges,
         self.capacities) = up.ned2lists(nedfile)
        graph = tf.Graph()
        with graph.as_default():
            self.model = rt.model.RouteNet_Fermi()
            # self.model.build(input_shape=(10, 10))
            # self.features = {
            #     'traffic': tf.keras.Input(dtype=tf.float32),
            #     'packets': tf.keras.Input(dtype=tf.float32),
            #     'length': tf.keras.Input(dtype=tf.int32),  # TODO: maybe float?
            #     'model':
            # }
            # TODO: Check if TensorSpec the right one? Is shape=None the right one?
            # TODO: I do not think this is the way to do it. Need to delete later

            self.features = {
                'traffic': tf.TensorSpec(shape=None),
                'packets': tf.TensorSpec(shape=None),
                'length': tf.TensorSpec(shape=None, dtype=tf.int32),  # TODO: maybe float?
                'model': tf.TensorSpec(shape=None),
                'eq_lambda': tf.TensorSpec(shape=None),
                'avg_pkts_lambda': tf.TensorSpec(shape=None),
                'exp_max_factor': tf.TensorSpec(shape=None),
                'pkts_lambda_on': tf.TensorSpec(shape=None),
                'avg_t_off': tf.TensorSpec(shape=None),
                'avg_t_on': tf.TensorSpec(shape=None),
                'ar_a': tf.TensorSpec(shape=None),
                'sigma': tf.TensorSpec(shape=None),
                'capacity': tf.TensorSpec(shape=None),
                'policy': tf.TensorSpec(shape=None),
                'queue_size': tf.TensorSpec(shape=None),
                'priority': tf.TensorSpec(shape=None),
                'weight': tf.TensorSpec(shape=None),
                'queue_to_path': tf.TensorSpec(shape=None),
                'link_to_path': tf.TensorSpec(shape=None),
                'path_to_link': tf.TensorSpec(shape=None),
                'path_to_queue': tf.TensorSpec(shape=None),
                'queue_to_link': tf.TensorSpec(shape=None)
            }
            # self.model.build(input_shape=(10, 10))


            # TODO: I Don't think we need those lines
            # self.predictions = self.model.call(self.features)[..., 0]
            # self.predictions = tf.squeeze(self.predictions)

    def calc(self, traffic, routing):
        # TODO: make it work somehow
        feature = self.input_to_call(traffic, routing)

        # TODO: there is no need for a loop right?
        # self.model.build(input_shape=(10, 10), input=feature)
        result = self.model.call(feature)
        print("===== RESULT IS =====")
        print(result)
        return result

    def input_to_call(self, traffic, routing):
        '''
        This is the function of the previous project, need to change what the returned value feature has to be inline
        with the call function of RouteNet_Fermi


        traffic - 2d tensor. Represents the overall data size which was sent from every node to every node. (overall size = 90)
        packets - 2d tensor. Represents the number of packets sent from every node to every node.


        '''

        R = routing

        paths = []
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    paths.append([self.edges.index(tup) for tup in up.pairwise(up.genPath(R, i, j, self.connections))])
        link_indices, path_indices, sequ_indices = up.make_indices(paths)

        n_paths = self.n * (self.n - 1)
        n_links = max(max(paths)) + 1

        feature = {
            "traffic": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "packets": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "length": tf.TensorSpec(shape=None, dtype=tf.int32),
            "model": tf.TensorSpec(shape=None, dtype=tf.int32),
            "eq_lambda": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "avg_pkts_lambda": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "exp_max_factor": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "pkts_lambda_on": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "avg_t_off": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "avg_t_on": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "ar_a": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "sigma": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "delay": tf.TensorSpec(shape=None, dtype=tf.float32),
            "queue_size": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "policy": tf.TensorSpec(shape=None, dtype=tf.int32),
            "priority": tf.TensorSpec(shape=None, dtype=tf.int32),
            "weight": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "link_to_path": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
            "queue_to_path": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
            "queue_to_link": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
            "path_to_queue": tf.RaggedTensorSpec(shape=(None, None, 2), dtype=tf.int32, ragged_rank=1),
            "path_to_link": tf.RaggedTensorSpec(shape=(None, None, 2), dtype=tf.int32, ragged_rank=1)
        }

        """
        feature['traffic'] = [feature['traffic'].astype(np.float32)]
        feature['capacity'] = [i / 10 for i in feature['capacity']]
        feature['traffic'] = torch.FloatTensor([(i - 0.18) / 0.15 for i in feature['traffic']])  # TODO: probably need to make it matrix of 10x9 (or 9x10) instead of array of len 90
        """
        return feature
