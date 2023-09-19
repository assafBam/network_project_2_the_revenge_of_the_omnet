import routenet_trainer as trainer
import networkx as nx
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow as tf
tf.get_logger().setLevel('INFO')

from RouteNet_Fermi.datanetAPI import DatanetAPI  # This API may be different for different versions of the dataset
import RouteNet_Fermi.data_generator as data_generator
from RouteNet_Fermi.data_generator import input_fn
import datanetAPI
import matplotlib.pyplot as plt

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


def test_model(checkpoint, number_of_nodes=12, dataset_name="dataset2"):
    # data_dir = "training/results/dataset2"
    # ds_test = input_fn(data_dir, shuffle=False)
    # ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    # model = trainer.RouteNet_Fermi()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    #
    # loss_object = tf.keras.losses.MeanAbsolutePercentageError()
    #
    # model.compile(loss=get_delay(loss_object),
    #               optimizer=optimizer,
    #               run_eagerly=False)
    # model.load_weights(checkpoint)
    # model.evaluate(ds_test)
    # with open("delays.txt", "r") as file:
    #     lines = file.readlines()
    # mean_delay = []
    # for line in lines:
    #     list_of_delays = eval(line.replace(" ", ","))
    #     index = 0
    #     delay_matrix = []
    #     for src in range(number_of_nodes):
    #         delays_to_dsts = []
    #         for dst in range(number_of_nodes):
    #             if src == dst:
    #                 delays_to_dsts.append(0)
    #             else:
    #                 delays_to_dsts.append(list_of_delays[index])  # TODO: check this
    #                 index += 1
    #         delay_matrix.append(delays_to_dsts)
    #         # print(src, str(delays_to_dsts))
    #         # delays_with_load.append(delays_to_dsts)
    #     mean_delay.append(np.array(delay_matrix))
    #
    # result = sum(mean_delay) / len(mean_delay)
    # print(result)
    #
    #
    #
    #
    # new_model = trainer.RouteNet_Fermi()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    #
    # loss_object = tf.keras.losses.MeanAbsolutePercentageError()
    #
    # new_model.compile(loss=get_delay(loss_object),
    #               optimizer=optimizer,
    #               run_eagerly=False)
    # # new_model.load_weights(checkpoint)
    # new_model.evaluate(ds_test)
    # with open("delays.txt", "r") as file:
    #     lines = file.readlines()
    # mean_delay = []
    # for line in lines:
    #     list_of_delays = eval(line.replace(" ", ","))
    #     index = 0
    #     delay_matrix = []
    #     for src in range(number_of_nodes):
    #         delays_to_dsts = []
    #         for dst in range(number_of_nodes):
    #             if src == dst:
    #                 delays_to_dsts.append(0)
    #             else:
    #                 delays_to_dsts.append(list_of_delays[index])  # TODO: check this
    #                 index += 1
    #         delay_matrix.append(delays_to_dsts)
    #         # print(src, str(delays_to_dsts))
    #         # delays_with_load.append(delays_to_dsts)
    #     mean_delay.append(np.array(delay_matrix))
    #
    # new_result = sum(mean_delay) / len(mean_delay)
    # print(new_result)


    data_dir = "training/results/" + dataset_name  # TODO: change dataset back to 2
    ds_test = input_fn(data_dir, shuffle=False)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    model = trainer.RouteNet_Fermi()
    model.load_weights(checkpoint)
    graph = nx.DiGraph(nx.read_gml("training/graphs/graph_pre_made.txt"))
    # inputs = hypergraph_to_input_data(graph)
    gen = data_generator.generator(data_dir=data_dir, shuffle=False, seed=1234, training=False)
    inputs = change_dtypes(next(gen)[0])
    # inputs = data_generator.input_fn(data_dir, shuffle=False)
    print(inputs)

    # for t in range(3):
    #     print("Try number", t)
    #     print(model.call(inputs))
    #     print("=================")

    output1 = model.call(inputs)
    print("The output is:")
    print("\t", list(range(number_of_nodes)))
    index = 0
    delays_without_load = []
    for src in range(number_of_nodes):
        delays_to_dsts = []
        for dst in range(number_of_nodes):
            if src == dst:
                delays_to_dsts.append(0)
                # continue
            else:
                delays_to_dsts.append(output1[index].numpy()[0])  # TODO: check this
                index += 1
        print(src, str(delays_to_dsts))
        delays_without_load.append(delays_to_dsts)
    final_result = np.median(delays_without_load, axis=0)
    return delays_without_load
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

def get_mean_delay_from_omnet(dataset_name, avg_bw):
    data_folder_name = "training"
    src_path = f"{data_folder_name}/results/" + dataset_name + "/"
    # max_avg_lambda_range = [10, 1000]
    max_avg_lambda_range = avg_bw
    net_size_lst = [12]
    reader = datanetAPI.DatanetAPI(src_path,max_avg_lambda_range, net_size_lst)

    samples_lst = []
    for sample in reader:
        samples_lst.append(sample)
    print("Number of selected samples: ", len(samples_lst))

    delays_lst = []
    packets_sent_list = []
    packets_lost_list = []
    # pkts_drop_dict = {}
    # for i in range(net_size_lst[0]):
    #     pkts_drop_dict[i] = {}
    #     for j in range(net_size_lst[0]):
    #         pkts_drop_dict[i][j] = []

    # print(pkts_drop_list)
    for s in samples_lst:
        performance_matrix = s.get_performance_matrix()
        delays_lst_per_sample = []
        for i in range(s.get_network_size()):
            for j in range(s.get_network_size()):
                if (i == j):
                    # continue
                    delays_lst_per_sample.append(0)
                    # pkts_drop_dict[i][j].append(0)
                else:
                    # Append to the list the average delay of the path i,j.
                    delays_lst_per_sample.append(performance_matrix[i, j]["AggInfo"]["AvgDelay"])
                    # print(performance_matrix[i, j]["AggInfo"]["PktsDrop"])

                    # pkts_drop_dict[i][j].append(performance_matrix[i, j]["AggInfo"]["PktsDrop"])
        delays_lst.append([np.array(delays_lst_per_sample)])
        packets_sent_list.append(s.get_global_packets())
        packets_lost_list.append(s.get_global_losses())
        # print(delays_lst_per_sample)

    final_result_delays = np.median(delays_lst, axis=0)
    final_result_packets_sent = np.median(packets_sent_list, axis=0)
    final_result_packets_lost = np.median(packets_lost_list, axis=0)
    # final_result_pkts_drop = np.median(pkts_drop_list, axis=0)
    # print("===== Final result is =====")
    # print(final_result)
    return final_result_delays, final_result_packets_sent, final_result_packets_lost

def test_overload_omnet(list_of_datasets, num_of_samples):
    """

    Parameters
    ----------
    list_of_datasets - gets the list in a format of "number_avgBW", where number is the value of the average BW

    Returns
    -------

    """
    hi_priority_delays = []  # flow is 1 -> 2
    regular_priority_delays = []  # flow is 5 -> 2
    regular_priority_non_affected_delays = []  # flow is 2 -> 5
    small_flow = []  # flow is 9 -> 11
    list_of_bw = []
    packet_loss_percentage = []
    high_pri_index = 12 * 1 + 2
    reg_pri_index = 12 * 5 + 2
    reg_pri_non_affected_index = 12 * 2 + 5
    small_index = 12 * 9 + 11
    for dataset_name in list_of_datasets:
        print("Current bw:", dataset_name.split("_")[0])
        bw_value = int(dataset_name.split("_")[0])
        omnet_delay, sent, lost = get_mean_delay_from_omnet(dataset_name + "_" + str(num_of_samples) + "_samples", [bw_value])
        # hi_pri_current = []
        # regular_pri_current = []
        # for current_sample in omnet_delay:
        #     print("Current omnet results for sample", current_sample)
        #     print(current_sample)
        #     hi_pri_current.append(current_sample[high_pri_index])
        #     regular_pri_current.append(current_sample[reg_pri_index])
        print(omnet_delay[0])
        hi_priority_delays.append(omnet_delay[0][high_pri_index])
        regular_priority_delays.append(omnet_delay[0][reg_pri_index])
        regular_priority_non_affected_delays.append(omnet_delay[0][reg_pri_non_affected_index])
        small_flow.append(omnet_delay[0][small_index])
        list_of_bw.append(bw_value)
        packet_loss_percentage.append((lost / sent) * 100)

    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    # ax2.set_ylim(ax1.get_ylim())


    ax1.plot(list_of_bw, hi_priority_delays, 'b-')
    ax1.set_xlabel('avg bw values')
    ax1.set_ylabel('high priority delay', color='b')
    ax1.tick_params(axis='y', colors='b')

    # ax2.plot(list_of_bw, regular_priority_delays, 'r-')
    # ax2.set_ylabel('regular priority delay affected by high priority', color='r')
    # ax2.tick_params(axis='y', colors='r')
    #
    ax3.plot(list_of_bw, regular_priority_non_affected_delays, 'g-')
    ax3.set_ylabel('regular flow non affected by high priority', color='g')
    ax3.tick_params(axis='y', colors='g')

    # plt.title("avg delay as a function of bw")
    # plt.show()

    fig, ax4 = plt.subplots()
    ax4.plot(list_of_bw, small_flow, 'g-')
    ax4.set_xlabel('avg bw values')
    ax4.set_ylabel('small flow delay', color='g')
    ax4.tick_params(axis='y', colors='g')

    plt.title("avg delay as a function of bw")
    plt.show()

    print("HIGH:")
    print(hi_priority_delays)
    print("REGULAR AFFECTED:")
    print(regular_priority_delays)
    print("REGULAR NON AFFECTED:")
    print(regular_priority_non_affected_delays)
    print("SMALL")
    print(small_flow)


def test_singular_flow_omnet(src, dst, num_of_samples):
    list_of_datasets = ["500_avgBW", "1000_avgBW", "2000_avgBW", "2500_avgBW", "5000_avgBW", "7500_avgBW",
                        "10000_avgBW", "12500_avgBW", "15000_avgBW", "17500_avgBW", "20000_avgBW", "25000_avgBW",
                        "30000_avgBW", "32500_avgBW", "35000_avgBW", "37500_avgBW", "40000_avgBW", "50000_avgBW",
                          "75000_avgBW", "100000_avgBW", "1000000_avgBW", "3000000_avgBW", "5000000_avgBW", "7000000_avgBW"]
    delays = []
    list_of_bw = []
    # pkts_drop_perc = []
    packet_loss_percentage = []

    for dataset_name in list_of_datasets:
        print("Current bw:", dataset_name.split("_")[0])
        bw_value = int(dataset_name.split("_")[0])
        omnet_delay, sent, lost = get_mean_delay_from_omnet(dataset_name + "_" + str(num_of_samples) + "_samples", [bw_value])
        print(omnet_delay[0])
        # print(pkts_drop)
        delays.append(omnet_delay[0][src * 12 + dst])
        list_of_bw.append(bw_value)
        # pkts_drop_perc.append(np.mean(pkts_drop[src][dst]))
        packet_loss_percentage.append((lost / sent) * 100)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(list_of_bw, delays, 'b-')
    ax1.set_xlabel('avg bw values')
    ax1.set_ylabel('delay from src ' + str(src) + " to dst " + str(dst), color='b')
    ax1.tick_params(axis='y', colors='b')

    ax2.plot(list_of_bw, packet_loss_percentage, 'g-')
    ax2.set_ylabel('packet loss percentage', color='g')
    ax2.tick_params(axis='y', colors='g')

    plt.title("avg delay as a function of bw")
    plt.show()

    print("list of delays:")
    print(delays)
    print("loss percentage:")
    print(packet_loss_percentage)


if __name__ == '__main__':
    # test_model('checkpoints/modelCheckpoints_10000samples_good_result/20-0.60')


    # routenet_delay = test_model("overloadCheckpoints_100_epochs/100-12.02", dataset_name="10000_avgBW")
    # omnet_delay = get_mean_delay_from_omnet("10000_avgBW", [10000])
    #
    # print("=================")
    # print("Routenet Delay:")
    # print(np.array(routenet_delay))
    # print("=================")
    # print("Omnet Delay:")
    # print(omnet_delay)
    #
    # diff_list = []
    # index = 0
    # for delays_list in routenet_delay:
    #     for delay in delays_list:
    #         diff_list.append(delay - omnet_delay[0][index])
    #         index += 1
    #
    # print("=================")
    # print("Diff list is:")
    # print(diff_list)
    # print("=================")
    # print("AVG diff is:")
    # print(np.average(np.abs(diff_list)))
    # routenet_delay = np.array(routenet_delay)
    # omnet_delay = omnet_delay[0].reshape(12, 12)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    #
    # # omnet = np.array([[0,1,2,3],[3,0,2,1],[2,3,0,1],[1,3,2,0]])
    # # routnet = np.array([[0,1.1,2.2,3.3],[3.1,0,2.2,1.3],[2.4,3.5,0,1.2],[1.1,3.2,2.3,0]])*0.0001
    # colors = ['r', 'b']
    #
    # yticks = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    # xticks = list(reversed(yticks))
    # xs = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
    # ys = [0,1,2,3,4,5,6,7,8,9,10,11]
    # width = 0.25
    #
    # for i in range(len(omnet_delay)):
    #     ax.bar(xs, omnet_delay[i], zs=i, zdir='y', color='r', alpha=1, width=width)
    #     ax.bar(xs+width, routenet_delay[i], zs=i, zdir='y', color='b', alpha=0.8, width=width)
    #
    # ax.set_yticks(yticks)
    # ax.set_xticks(xticks)
    #
    # ax.set_xlabel('src')
    # ax.set_ylabel('dst')
    # ax.set_zlabel('delay')
    # plt.show()

    print("Staring the Omnet Overload Test")
    # test_overload_omnet(["500_avgBW", "1000_avgBW", "2000_avgBW", "2500_avgBW", "5000_avgBW", "7500_avgBW",
    #                     "10000_avgBW", "12500_avgBW", "15000_avgBW", "17500_avgBW", "20000_avgBW", "25000_avgBW",
    #                      "30000_avgBW", "32500_avgBW", "35000_avgBW", "37500_avgBW", "40000_avgBW", "50000_avgBW",
    #                      "75000_avgBW", "100000_avgBW", "1000000_avgBW", "3000000_avgBW", "5000000_avgBW", "7000000_avgBW"], 15)
    test_singular_flow_omnet(1, 2, 15)
    test_singular_flow_omnet(5, 2, 15)
    # test_singular_flow_omnet(2, 5, 15)
    test_singular_flow_omnet(9, 11, 15)
    print("DONE")

