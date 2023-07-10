import routenet_trainer as trainer
import networkx as nx
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')

from RouteNet_Fermi.datanetAPI import DatanetAPI  # This API may be different for different versions of the dataset
import RouteNet_Fermi.data_generator as data_generator
from RouteNet_Fermi.data_generator import input_fn

"""
def hypergraph_to_input_data(HG):
    n_q = 0
    n_p = 0
    n_l = 0
    mapping = {}
    for entity in list(HG.nodes()):
        if entity.startswith('q'):
            mapping[entity] = ('q_{}'.format(n_q))
            n_q += 1
        elif entity.startswith('p'):
            mapping[entity] = ('p_{}'.format(n_p))
            n_p += 1
        elif entity.startswith('l'):
            mapping[entity] = ('l_{}'.format(n_l))
            n_l += 1

    HG = nx.relabel_nodes(HG, mapping)

    link_to_path = []
    queue_to_path = []
    path_to_queue = []
    queue_to_link = []
    path_to_link = []

    for node in HG.nodes:
        in_nodes = [s for s, d in HG.in_edges(node)]
        if node.startswith('q_'):
            path = []
            for n in in_nodes:
                if n.startswith('p_'):
                    path_pos = []
                    for _, d in HG.out_edges(n):
                        if d.startswith('q_'):
                            path_pos.append(d)
                    path.append([int(n.replace('p_', '')), path_pos.index(node)])
            if len(path) == 0:
                print(in_nodes)
            path_to_queue.append(path)
        elif node.startswith('p_'):
            links = []
            queues = []
            for n in in_nodes:
                if n.startswith('l_'):
                    links.append(int(n.replace('l_', '')))
                elif n.startswith('q_'):
                    queues.append(int(n.replace('q_', '')))
            link_to_path.append(links)
            queue_to_path.append(queues)
        elif node.startswith('l_'):
            queues = []
            paths = []
            for n in in_nodes:
                if n.startswith('q_'):
                    queues.append(int(n.replace('q_', '')))
                elif n.startswith('p_'):
                    path_pos = []
                    for _, d in HG.out_edges(n):
                        if d.startswith('l_'):
                            path_pos.append(d)
                    paths.append([int(n.replace('p_', '')), path_pos.index(node)])
            path_to_link.append(paths)
            queue_to_link.append(queues)

    return {"traffic": np.expand_dims(list(nx.get_node_attributes(HG, 'traffic').values()), axis=1),
            "packets": np.expand_dims(list(nx.get_node_attributes(HG, 'packets').values()), axis=1),
            "length": list(nx.get_node_attributes(HG, 'length').values()),
            "model": list(nx.get_node_attributes(HG, 'model').values()),
            "eq_lambda": np.expand_dims(list(nx.get_node_attributes(HG, 'eq_lambda').values()), axis=1),
            "avg_pkts_lambda": np.expand_dims(list(nx.get_node_attributes(HG, 'avg_pkts_lambda').values()), axis=1),
            "exp_max_factor": np.expand_dims(list(nx.get_node_attributes(HG, 'exp_max_factor').values()), axis=1),
            "pkts_lambda_on": np.expand_dims(list(nx.get_node_attributes(HG, 'pkts_lambda_on').values()), axis=1),
            "avg_t_off": np.expand_dims(list(nx.get_node_attributes(HG, 'avg_t_off').values()), axis=1),
            "avg_t_on": np.expand_dims(list(nx.get_node_attributes(HG, 'avg_t_on').values()), axis=1),
            "ar_a": np.expand_dims(list(nx.get_node_attributes(HG, 'ar_a').values()), axis=1),
            "sigma": np.expand_dims(list(nx.get_node_attributes(HG, 'sigma').values()), axis=1),
            "capacity": np.expand_dims(list(nx.get_node_attributes(HG, 'capacity').values()), axis=1),
            "queue_size": np.expand_dims(list(nx.get_node_attributes(HG, 'queue_size').values()), axis=1),
            "policy": list(nx.get_node_attributes(HG, 'policy').values()),
            "priority": list(nx.get_node_attributes(HG, 'priority').values()),
            "weight": np.expand_dims(list(nx.get_node_attributes(HG, 'weight').values()), axis=1),
            "delay": list(nx.get_node_attributes(HG, 'delay').values()),
            "link_to_path": tf.ragged.constant(link_to_path),
            "queue_to_path": tf.ragged.constant(queue_to_path),
            "queue_to_link": tf.ragged.constant(queue_to_link),
            "path_to_queue": tf.ragged.constant(path_to_queue, ragged_rank=1),
            "path_to_link": tf.ragged.constant(path_to_link, ragged_rank=1)
            }
"""

def change_dtypes(input_dict):
    input_dict['traffic'] = input_dict['traffic'].astype('float32')
    input_dict['packets'] = input_dict['packets'].astype('float32')
    # input_dict['length'] = input_dict['length'].astype('int32')
    # input_dict['model'] = input_dict['model'].astype('int32')
    input_dict['eq_lambda'] = input_dict['eq_lambda'].astype('float32')
    input_dict['avg_pkts_lambda'] = input_dict['avg_pkts_lambda'].astype('float32')
    input_dict['exp_max_factor'] = input_dict['exp_max_factor'].astype('float32')
    input_dict['pkts_lambda_on'] = input_dict['pkts_lambda_on'].astype('float32')
    input_dict['avg_t_off'] = input_dict['avg_t_off'].astype('float32')
    input_dict['avg_t_on'] = input_dict['avg_t_on'].astype('float32')
    input_dict['ar_a'] = input_dict['ar_a'].astype('float32')
    input_dict['sigma'] = input_dict['sigma'].astype('float32')
    input_dict['capacity'] = input_dict['capacity'].astype('float32')
    # input_dict['delay'] = input_dict['delay'].astype('float32')
    input_dict['queue_size'] = input_dict['queue_size'].astype('float32')
    # input_dict['policy'] = input_dict['policy'].astype('int32')
    # input_dict['priority'] = input_dict['priority'].astype('int32')
    input_dict['weight'] = input_dict['weight'].astype('float32')
    # input_dict['link_to_path'] = input_dict['link_to_path'].astype('int32')
    # input_dict['queue_to_path'] = input_dict['queue_to_path'].astype('int32')
    # input_dict['queue_to_link'] = input_dict['queue_to_link'].astype('int32')
    # input_dict['path_to_queue'] = input_dict['path_to_queue'].astype('int32')
    # input_dict['path_to_link'] = input_dict['path_to_link'].astype('int32')

    return input_dict

def get_delay(loss_object):
    def delays(y_true, y_pred):
        # loss_value = 100 * abs((y_true - y_pred) / y_true)
        with open("delays.txt", "w"):
            print(y_pred)
            tf.print(y_pred, summarize=-1, output_stream="file://delays.txt")
        return loss_object(y_true, y_pred)
    return delays


def test_model(checkpoint, number_of_nodes=12):
    data_dir = "training/results/dataset2"
    ds_test = input_fn(data_dir, shuffle=False)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    model = trainer.RouteNet_Fermi()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    loss_object = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(loss=get_delay(loss_object),
                  optimizer=optimizer,
                  run_eagerly=False)
    model.load_weights(checkpoint)
    model.evaluate(ds_test)
    with open("delays.txt", "r") as file:
        lines = file.readlines()
    mean_delay = []
    for line in lines:
        list_of_delays = eval(line.replace(" ", ","))
        index = 0
        delay_matrix = []
        for src in range(number_of_nodes):
            delays_to_dsts = []
            for dst in range(number_of_nodes):
                if src == dst:
                    delays_to_dsts.append(0)
                else:
                    delays_to_dsts.append(list_of_delays[index])  # TODO: check this
                    index += 1
            delay_matrix.append(delays_to_dsts)
            # print(src, str(delays_to_dsts))
            # delays_with_load.append(delays_to_dsts)
        mean_delay.append(np.array(delay_matrix))

    result = sum(mean_delay) / len(mean_delay)
    print(result)




    new_model = trainer.RouteNet_Fermi()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    loss_object = tf.keras.losses.MeanAbsolutePercentageError()

    new_model.compile(loss=get_delay(loss_object),
                  optimizer=optimizer,
                  run_eagerly=False)
    # new_model.load_weights(checkpoint)
    new_model.evaluate(ds_test)
    with open("delays.txt", "r") as file:
        lines = file.readlines()
    mean_delay = []
    for line in lines:
        list_of_delays = eval(line.replace(" ", ","))
        index = 0
        delay_matrix = []
        for src in range(number_of_nodes):
            delays_to_dsts = []
            for dst in range(number_of_nodes):
                if src == dst:
                    delays_to_dsts.append(0)
                else:
                    delays_to_dsts.append(list_of_delays[index])  # TODO: check this
                    index += 1
            delay_matrix.append(delays_to_dsts)
            # print(src, str(delays_to_dsts))
            # delays_with_load.append(delays_to_dsts)
        mean_delay.append(np.array(delay_matrix))

    new_result = sum(mean_delay) / len(mean_delay)
    print(new_result)



    # graph = nx.DiGraph(nx.read_gml("training/graphs/graph_pre_made.txt"))
    # # inputs = hypergraph_to_input_data(graph)
    # gen = data_generator.generator(data_dir=data_dir, shuffle=False, seed=1234, training=False)
    # inputs = change_dtypes(next(gen)[0])
    # # inputs = data_generator.input_fn(data_dir, shuffle=False)
    # print(inputs)
    #
    # output1 = model.call(inputs)
    # print("The output is:")
    # print("\t", list(range(number_of_nodes)))
    # index = 0
    # delays_without_load = []
    # for src in range(number_of_nodes):
    #     delays_to_dsts = []
    #     for dst in range(number_of_nodes):
    #         if src == dst:
    #             delays_to_dsts.append(None)
    #         else:
    #             delays_to_dsts.append(output1[index].numpy()[0])  # TODO: check this
    #             index += 1
    #     print(src, str(delays_to_dsts))
    #     delays_without_load.append(delays_to_dsts)
    #
    # model2 = trainer.RouteNet_Fermi()
    # model2.compile(loss=get_delay(loss_object),
    #               optimizer=optimizer,
    #               run_eagerly=False)
    # # model2.load_weights(checkpoint)
    # output2 = model2.call(inputs)
    #
    # index = 0
    # delays_with_load = []
    # for src in range(number_of_nodes):
    #     delays_to_dsts = []
    #     for dst in range(number_of_nodes):
    #         if src == dst:
    #             delays_to_dsts.append(None)
    #         else:
    #             delays_to_dsts.append(output2[index].numpy()[0])  # TODO: check this
    #             index += 1
    #     print(src, str(delays_to_dsts))
    #     delays_with_load.append(delays_to_dsts)
    #
    # sum = 0
    # for i in range(len(delays_with_load)):
    #     for j in range(len(delays_with_load)):
    #        if delays_with_load[i][j] and delays_with_load[i][j] > delays_without_load[i][j]:
    #            sum += 1
    #        else:
    #            print("src is", i, "dst is", j)
    #
    # print("sum is", sum)
    # # print(np.sum(output2 > output1))
    # # print(output)


if __name__ == '__main__':
    # test_model('checkpoints/modelCheckpoints_10000samples_good_result/20-0.60')
    test_model("modelCheckpoints/20-0.32")
