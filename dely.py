import tensorflow as tf
import upcdataset as up
import numpy as np
import routenet as rt


hparams = tf.contrib.training.HParams(
    link_state_dim=32,
    path_state_dim=32,
    T=8,
    readout_units=8,
    learning_rate=0.001,
    batch_size=16,
    dropout_rate=0.5,
    l2=0.1,
    l2_2=0.01,
    learn_embedding=True,  # If false, only the readout is trained
    readout_layers=2,  # number of hidden layers in readout model
)


class delay:
    def __init__(self, nedfile):
        (self.connections,
         self.n,
         self.edges,
         self.capacities) = up.ned2lists(nedfile)
        graph = tf.Graph()
        with graph.as_default():
            self.model = rt.RouteNet(hparams, 2)
            self.model.build()
            self.features = {
                'traffic': tf.placeholder(dtype=tf.float32),
                'capacities': tf.placeholder(dtype=tf.float32),
                'links': tf.placeholder(dtype=tf.int32),
                'paths': tf.placeholder(dtype=tf.int32),
                'sequences': tf.placeholder(dtype=tf.int32),
                'n_links': tf.placeholder(dtype=tf.int32),
                'n_paths': tf.placeholder(dtype=tf.int32)
            }

            self.predictions = self.model.call(self.features)[..., 0]  # equal to [:,0]
            self.predictions = tf.squeeze(self.predictions)

            self.sess = tf.Session(graph=graph)
            self.saver = tf.compat.v1.train.Saver()
            # path to the checkpoint we want to restore
            self.saver.restore(self.sess, 'checkpoints_delay/nsfnetbw/model.ckpt-130000')

            # print("PREDICTION IS:")
            # print(self.predictions)
            # print("TYPE IS:")
            # print(self.predictions.shape)

    def calc(self, traffic, routing):

        feture = self.input_to_tfrecord(traffic, routing)
        hats = []
        for i in range(50):
            p = self.sess.run([self.predictions], feed_dict={self.features['traffic']: feture['traffic'],
                                                             self.features['capacities']: feture['capacities'],
                                                             self.features['links']: feture['links'],
                                                             self.features['paths']: feture['paths'],
                                                             self.features['sequences']: feture['sequences'],
                                                             self.features['n_links']: feture['n_links'],
                                                             self.features['n_paths']: feture['n_paths'],
                                                             })
            hats.append(p)
            # print("===== p is ======")
            # print(p)
            # print("===== shape of p is: =====")
            # print(p[0].shape)
        final_prediction = np.median(hats, axis=0)
        print("===== Final Prediction is =====")
        print(final_prediction)
        return final_prediction

    def input_to_tfrecord(self, traffic, routing):

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
            'capacities': self.capacities,
            'links': link_indices,
            'paths': path_indices,
            'sequences': sequ_indices,
            'n_links': n_links,
            'n_paths': n_paths,
            'traffic': traffic,
        }
        feature['traffic'] = feature['traffic'].astype(np.float32)
        feature['capacities'] = [i / 10 for i in feature['capacities']]
        feature['traffic'] = [(i - 0.18) / 0.15 for i in feature['traffic']]
        return feature

