import sys
from getpass import getpass

import networkx as nx
import random
import os

import numpy
import torch
import yaml
from RouteNet_Fermi import main as routenetMain
from RouteNet_Fermi import evaluate

SUDO_PASS = f"echo {getpass()} | sudo -S " if os.name != 'nt' else ''

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

from sys import stderr
import warnings

warnings.filterwarnings("ignore")
seed_value = 69420
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYTHONHASHSEED'] = str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.random.set_seed(seed_value)

tf.get_logger().setLevel('INFO')

from RouteNet_Fermi.data_generator import input_fn
from RouteNet_Fermi.model import RouteNet_Fermi


class RouteNetTrainer:

    def __init__(self, dataset_name, topology_file):
        # self.topology_file = topology_file

        # Define destination for the generated samples
        self.training_dataset_path = "training"
        # paths relative to data folder
        self.graphs_path = "graphs"
        self.routings_path = "routings"
        self.tm_path = "tm"
        # Path to simulator file
        self.simulation_file = os.path.join(self.training_dataset_path, "simulation.txt")
        # Name of the dataset: Allows you to store several datasets in the same path
        # Each dataset will be stored at <training_dataset_path>/results/<name>
        self.dataset_name = dataset_name
        self.topology_file = topology_file
        self.create_folders()

    def create_folders(self):
        # Create folders
        if os.path.isdir(self.training_dataset_path):
            print("Destination path already exists. Files within the directory may be overwritten.")
        else:
            os.makedirs(os.path.join(self.training_dataset_path, self.graphs_path))
            # os.mkdir((os.path.join(self.training_dataset_path, self.graphs_path)))
            os.mkdir(os.path.join(self.training_dataset_path, self.routings_path))
            os.mkdir(os.path.join(self.training_dataset_path, self.tm_path))

    def generate_premade_topology(self, net_size, graph_file):
        G = nx.Graph()

        # Set the maximum number of ToS that will use the input traffic of the network
        G.graph["levelsToS"] = 2

        for n in range(net_size):
            G.add_node(n)
            # Assign to each node the scheduling Policy
            G.nodes[n]["schedulingPolicy"] = "SP"
            # Assign ToS to scheduling queues.
            # In this case we have two queues per port. ToS 0 is assigned to the first queue and ToS 1 and 2 to the second queue
            G.nodes[n]["tosToQoSqueue"] = "0;1"
            # Assign weights to each queue
            # G.nodes[n]["schedulingWeights"] = "60, 40"
            # Assign the buffer size of all the ports of the node
            G.nodes[n]["bufferSizes"] = 8000

        G.add_edge(0, 3)
        G[0][3]["bandwidth"] = 10000
        G.add_edge(0, 6)
        G[0][6]["bandwidth"] = 10000
        G.add_edge(0, 7)
        G[0][7]["bandwidth"] = 10000

        G.add_edge(1, 5)
        G[1][5]["bandwidth"] = 10000
        G.add_edge(1, 6)
        G[1][6]["bandwidth"] = 10000
        G.add_edge(1, 9)
        G[1][9]["bandwidth"] = 10000

        G.add_edge(2, 8)
        G[2][8]["bandwidth"] = 10000
        G.add_edge(2, 10)
        G[2][10]["bandwidth"] = 10000

        G.add_edge(3, 5)
        G[3][5]["bandwidth"] = 10000

        G.add_edge(4, 7)
        G[4][7]["bandwidth"] = 10000
        G.add_edge(4, 8)
        G[4][8]["bandwidth"] = 10000
        G.add_edge(4, 10)
        G[4][10]["bandwidth"] = 10000

        G.add_edge(5, 11)
        G[5][11]["bandwidth"] = 10000

        G.add_edge(8, 10)
        G[8][10]["bandwidth"] = 10000

        G.add_edge(9, 11)
        G[9][11]["bandwidth"] = 10000



        nx.write_gml(G, graph_file)
        return G



    '''
    Generate a graph topology file. The graphs generated have the following characteristics:
    - The network is able to process 3 ToS: 0,1,2
    - All nodes have buffer sizes of 32000 bits and WFQ scheduling. ToS 0 is assigned to first queue, and ToS 1 and 2 to second queue.
    - All links have bandwidths of 100000 bits per second
    
    - Right now, no one is calling this function - big sadj :(
    '''

    def generate_topology(self, net_size, graph_file):
        return self.generate_premade_topology(net_size, graph_file)

        # G = nx.Graph()
        #
        # # Set the maximum number of ToS that will use the input traffic of the network
        # G.graph["levelsToS"] = 3
        #
        # nodes = []
        # node_degree = []
        # for n in range(net_size):
        #     node_degree.append(random.choices([2, 3, 4, 5, 6], weights=[0.34, 0.35, 0.2, 0.1, 0.01])[0])
        #
        #     nodes.append(n)
        #     G.add_node(n)
        #     # Assign to each node the scheduling Policy
        #     G.nodes[n]["schedulingPolicy"] = "SP"
        #     # Assign ToS to scheduling queues.
        #     # In this case we have two queues per port. ToS 0 is assigned to the first queue and ToS 1 and 2 to the second queue
        #     G.nodes[n]["tosToQoSqueue"] = "0;1,2"
        #     # Assign weights to each queue
        #     G.nodes[n]["schedulingWeights"] = "60, 40"
        #     # Assign the buffer size of all the ports of the node
        #     G.nodes[n]["bufferSizes"] = 32000
        #
        # finish = False
        # while True:
        #     aux_nodes = list(nodes)
        #     n0 = random.choice(aux_nodes)
        #     aux_nodes.remove(n0)
        #     # Remove adjacents nodes (only one link between two nodes)
        #     for n1 in G[n0]:
        #         if n1 in aux_nodes:
        #             aux_nodes.remove(n1)
        #     if len(aux_nodes) == 0:
        #         # No more links can be added to this node - can not acomplish node_degree for this node
        #         nodes.remove(n0)
        #         if len(nodes) == 1:
        #             break
        #         continue
        #     n1 = random.choice(aux_nodes)
        #     G.add_edge(n0, n1)
        #     # Assign the link capacity to the link
        #     G[n0][n1]["bandwidth"] = 100000
        #
        #     for n in [n0, n1]:
        #         node_degree[n] -= 1
        #         if (node_degree[n] == 0):
        #             nodes.remove(n)
        #             if (len(nodes) == 1):
        #                 finish = True
        #                 break
        #     if finish:
        #         break
        # if not nx.is_connected(G):
        #     G = self.generate_topology(net_size, graph_file)
        #     return G
        #
        # nx.write_gml(G, graph_file)
        #
        # return G

    '''
    Generate a file with the shortest path routing of the topology G
    '''
    def generate_routing(self, G, routing_file):
        with open(routing_file, "w") as r_fd:
            lPaths = nx.shortest_path(G)
            for src in G:
                for dst in G:
                    if src == dst:
                        continue
                    path = ','.join(str(x) for x in lPaths[src][dst])
                    r_fd.write(path + "\n")

    '''
    Generate a traffic matrix file. We consider flows between all nodes in the newtork, each with the following characterstics
    - The average bandwidth ranges between 10 and max_avg_lbda
    - We consider three time distributions (in case of the ON-OFF policy we have off periods of 10 and on periods of 5)
    - We consider two packages distributions, chosen at random
    - ToS is assigned randomly
    '''
    def generate_tm(self, G, max_avg_lbda, traffic_file):
        poisson = "0"
        cbr = "1"
        on_off = "2,10,5"  # time_distribution, avg off_time exp, avg on_time exp
        time_dist = [poisson, cbr, on_off]

        pkt_dist_1 = "0,300,0.5,1700,0.5"  # genric pkt size dist, pkt_size 1, prob 1, pkt_size 2, prob 2
        pkt_dist_2 = "0,500,0.6,1000,0.2,1400,0.2"  # genric pkt size dist, pkt_size 1, prob 1,
        # pkt_size 2, prob 2, pkt_size 3, prob 3
        pkt_dist_3 = "0,300,0.5,572,0.32,486,0.18"
        pkt_dist_4 = "0,666,0.42,420,0.58"  # genric pkt size dist, pkt_size 1, prob 1, pkt_size 2, prob 2

        pkt_size_dist = [pkt_dist_1, pkt_dist_2, pkt_dist_3, pkt_dist_4]
        tos_lst = [0, 1, 2]
        nodes = list(G.nodes)

        # avg_bw_dist = [1000, 10000, 25000, 50000, 75000, 100000, 200000]

        with open(traffic_file, "w") as tm_fd:
            for src in G:
                for dst in G:
                    # avg_bw = random.randint(10, max_avg_lbda)
                    avg_bw = max_avg_lbda
                    # avg_bw = random.choice(avg_bw_dist)
                    td = random.choice(time_dist)
                    sd = random.choice(pkt_size_dist)
                    # tos = random.choice(tos_lst)
                    # set the tos to be 0 only if node number 1 is sending data to node number 2 (given this flow priority)
                    if src == nodes[1] and dst == nodes[2]:
                        tos = 0
                    else:
                        tos = 1
                    traffic_line = "{},{},{},{},{},{}".format(
                        src, dst, avg_bw, td, sd, tos)
                    tm_fd.write(traffic_line + "\n")

    def load_topology(self, file):
        return nx.read_gml(file)

    def generate_file(self, num_of_samples, num_of_nodes=12, avg_bw=15000):
        """
        We generate the files using the previously defined functions. This code will produce 100 samples where:
        - We generate 5 topologies, and then we generate 20 traffic matrices for each
        - The topology sizes range from 6 to 10 nodes
        - We consider the maximum average bandwidth per flow as 1000
        """
        # for j in range(6, 16):
        max_avg_lbda = avg_bw
        with open(self.simulation_file, "w") as fd:
            # Generate graph
            # graph_file = os.path.join(self.graphs_path, self.topology_file)
            # graph_file = self.topology_file
            graph_file = os.path.join(self.graphs_path, self.topology_file)
            # graph_file = os.path.join(self.training_dataset_path, self.graphs_path, f"graph_{j}.txt")
            # G = self.load_topology(graph_file)
            G = self.generate_topology(num_of_nodes, os.path.join(self.training_dataset_path, graph_file))
            # Generate routing
            routing_file = os.path.join(self.routings_path, "routing.txt")
            self.generate_routing(G, os.path.join(self.training_dataset_path, routing_file))
            # Generate TM:
            for i in range(num_of_samples):
                tm_file = os.path.join(self.tm_path, "tm_{}_{}.txt".format(num_of_nodes, i))
                self.generate_tm(G, max_avg_lbda, os.path.join(self.training_dataset_path, tm_file))
                sim_line = "{},{},{}\n".format(graph_file, routing_file, tm_file)
                # If dataset has been generated in windows, convert paths into linux format
                fd.write(sim_line.replace("\\", "/"))

    def write_config(self):
        # First we generate the configuration file

        conf_file = os.path.join(self.training_dataset_path, "conf.yml")
        conf_parameters = {
            "threads": 6,  # Number of threads to use
            "dataset_name": self.dataset_name,
            # Name of the dataset. It is created in <training_dataset_path>/results/<name>
            "samples_per_file": 10,  # Number of samples per compressed file
            "rm_prev_results": "y",  # If 'y' is selected and the results folder already exists, the folder is removed.
        }

        with open(conf_file, 'w') as fd:
            yaml.dump(conf_parameters, fd)

    def docker_cmd(self, training_dataset_path):
        # raw_cmd = f"docker run --rm --mount type=bind,src={os.path.join(os.getcwd(), training_dataset_path)},dst=/data bnnupc/netsim:v0.1"
        raw_cmd = f"docker run --rm --mount type=bind,src={os.path.join(os.getcwd(), training_dataset_path)},dst=/data bnnupc/bnnetsimulator"
        print(raw_cmd)
        terminal_cmd = raw_cmd
        if os.name != 'nt':  # Unix, requires sudo
            print("Superuser privileges are required to run docker. Introduce sudo password when prompted")
            # terminal_cmd = f"echo {getpass()} | sudo -S " + raw_cmd
            terminal_cmd = SUDO_PASS + raw_cmd
            raw_cmd = "sudo " + raw_cmd
        return raw_cmd, terminal_cmd

    def start_docker(self):
        # Start the docker
        raw_cmd, terminal_cmd = self.docker_cmd(self.training_dataset_path)
        if os.name != 'nt':  # Unix
            print("THE COMMAND IS")
            print(terminal_cmd)
            os.system(terminal_cmd)
        else:
            os.system('cmd /c '+terminal_cmd)


def get_lost(loss_object, training=False):
    if training:
        def training_loss(y_true, y_pred):
            loss_value = 100 * abs((y_true - y_pred) / y_true)
            with open("train_loss_values_mean.txt", "w"):
                tf.print(tf.math.reduce_mean(loss_value), output_stream="file://train_loss_values_mean.txt")
            return loss_object(y_true, y_pred)
    else:
        def validation_loss(y_true, y_pred):
            loss_value = 100 * abs((y_true - y_pred) / y_true)
            with open("validate_loss_values_mean.txt", "w"):
                tf.print(tf.math.reduce_mean(loss_value), output_stream="file://validate_loss_values_mean.txt")
            return loss_object(y_true, y_pred)

    return training_loss if training else validation_loss


def train(train_path, final_evaluation=False, ckpt_dir="./modelCheckpoints"):
    """
    Trains and evaluates the model with the provided dataset.
    The model will be trained for 20 epochs.
    At each epoch a checkpoint of the model will be generated and stored at the folder ckpt_dir which will be created automatically if it doesn't exist already.
    Training the model will also generate logs at "./logs" that can be opened with tensorboard.

    Parameters
    ----------
    train_path
        Path to the training dataset
    final_evaluation, optional
        If True after training the model will be validated using all of the validation dataset, by default False
    ckpt_dir, optional
        Relative path (from the repository's root folder) where the model's weight will be stored, by default "./modelCheckpoints"
    """

    if (not os.path.exists(train_path)):
        print(f"ERROR: the provided training path \"{os.path.abspath(train_path)}\" does not exist!", file=stderr)
        return None
    TEST_PATH = 'training/results/overload_validation'  # TODO: make another dataset for validation
    if (not os.path.exists(TEST_PATH)):
        print("ERROR: Validation dataset not found at the expected location:",
              os.path.abspath(TEST_PATH), file=stderr)
        return None
    LOG_PATH = './logs'
    if (not os.path.exists(LOG_PATH)):
        print("INFO: Logs folder created at ", os.path.abspath(LOG_PATH))
        os.makedirs(LOG_PATH)
    # Check dataset size
    dataset_size = len([0 for _ in input_fn(train_path, shuffle=True)])
    if not dataset_size:
        print(f"ERROR: The dataset has no valid samples!", file=stderr)
        return None
    # elif (dataset_size > 100):
    #     print(f"ERROR: The dataset can only have up to 100 samples (currently has {dataset_size})!", file=stderr)
    #     return None

    ds_train = input_fn(train_path, shuffle=True, training=True)
    ds_train = ds_train.repeat()
    ds_test = input_fn(TEST_PATH, shuffle=False)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    model = RouteNet_Fermi()

    loss_object = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(loss=get_lost(loss_object, training=True),
                  optimizer=optimizer,
                  run_eagerly=False)

    model.load_weights('./RouteNet_Fermi/initial_weights/initial_weights')

    latest = tf.train.latest_checkpoint(ckpt_dir)

    if latest is not None:
        print(f"ERROR: Found a pretrained models, please clear or remove the {ckpt_dir} directory and try again!")
        return None
    else:
        print("INFO: Starting training from scratch...")

    filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.2f}")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        verbose=1,
        mode="min",
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True,
        save_freq='epoch')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH, histogram_freq=1)

    model.fit(ds_train,
              epochs=20,
              steps_per_epoch=2000,
              validation_data=ds_test,
              validation_steps=20,
              callbacks=[cp_callback, tensorboard_callback],
              use_multiprocessing=True)

    # model.fit(ds_train,
    #           epochs=90,
    #           steps_per_epoch=4000,
    #           validation_data=ds_test,
    #           validation_steps=40,
    #           callbacks=[cp_callback, tensorboard_callback],
    #           use_multiprocessing=True)

    if final_evaluation:
        print("Final evaluation:")
        model.evaluate(ds_test)


def evaluate(checkpoint):
    TEST_PATH = 'training/results/overload_validation'
    if (not os.path.exists(TEST_PATH)):
        print("ERROR: Validation dataset not found at the expected location:",
              os.path.abspath(TEST_PATH), file=stderr)
        return None

    ds_test = input_fn(TEST_PATH, shuffle=False)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    model = RouteNet_Fermi()

    loss_object = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(loss=get_lost(loss_object),
                  optimizer=optimizer,
                  run_eagerly=False)

    # stored_weights = tf.train.load_checkpoint(ckpt_path)
    model.load_weights(checkpoint)

    # Evaluate model
    model.evaluate(ds_test)


def print_graphs(filename, training=False, num_epochs=20, steps_per_epoch=2000, valid_steps=20):
    training_loss_values, validate_loss_values, test_loss_values, epochs = [], [], [], []
    training_epoch, validate_epoch, test_epoch = [], [], []

    with open(filename, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if training:
            if i % (steps_per_epoch + valid_steps) <= (valid_steps - 1):
                validate_epoch.append(float(line))
            if i % (steps_per_epoch + valid_steps) == (valid_steps - 1):
                validate_loss_values.append(sum(validate_epoch) / len(validate_epoch))
                validate_epoch = []
            if i % (steps_per_epoch + valid_steps) > (valid_steps - 1):
                training_epoch.append(float(line))
            if i % (steps_per_epoch + valid_steps) == (steps_per_epoch + valid_steps - 1):
                training_loss_values.append(sum(training_epoch) / len(training_epoch))
                training_epoch = []
        else:
            test_epoch.append(float(line))
            if i % 10 == 0:
                test_loss_values.append(sum(test_epoch) / len(test_epoch))
                test_epoch = []

    if training:
        epochs = list(range(len(training_loss_values)))
        validate_loss_values = validate_loss_values[1:]
        # draw the graphs for the training session
        # epochs_new = np.linspace(np.array(epochs).min(), np.array(epochs).max(), 300)
        # spl = make_interp_spline(epochs, training_loss_values, k=3)  # type: BSpline
        # training_smooth = spl(epochs_new)
        #
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(epochs, training_loss_values, 'b-')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('training mean loss values', color='b')
        ax1.tick_params(axis='y', colors='b')

        # spl = make_interp_spline(epochs, validate_loss_values, k=3)  # type: BSpline
        # validate_smooth = spl(epochs_new)

        ax2.plot(epochs, validate_loss_values, 'r-')
        ax2.set_ylabel('validate mean loss values', color='r')
        ax2.tick_params(axis='y', colors='r')
    else:
        epochs = list(range(len(test_loss_values)))
        plt.plot(epochs, test_loss_values)
        plt.xlabel('epoch')
        plt.ylabel('mean loss value')

    plt.title(filename)
    plt.show()


def generate_datasets_for_omnet_overload_test():
    list_of_datasets = ["500_avgBW", "2000_avgBW", "2500_avgBW", "5000_avgBW", "7500_avgBW", "12500_avgBW", "15000_avgBW"]
    for dataset in list_of_datasets:
        print("Generating new dataset", dataset, "with value", dataset.split("_")[0])
        trainer = RouteNetTrainer(dataset, "graph_pre_made.txt")
        bw_value = int(dataset.split("_")[0])
        trainer.write_config()
        trainer.generate_file(1, avg_bw=bw_value)
        trainer.start_docker()
        print("Generating DONE")


if __name__ == '__main__':
    # # Generate the datasets
    # print("Generating training dataset...")
    # trainer = RouteNetTrainer("overload_train", "graph_pre_made.txt")
    # trainer.write_config()
    # trainer.generate_file(15000)
    # trainer.start_docker()
    # print("Generating DONE")

    # print("Generating validation dataset...")
    # trainer = RouteNetTrainer("overload_validation", "graph_pre_made.txt")
    # trainer.write_config()
    # trainer.generate_file(3000)
    # trainer.start_docker()
    # print("Generating DONE")
    #
    # # Train the model
    # train("./training/results/overload_train", ckpt_dir="./overloadCheckpoints")
    # # routenetMain("./training/results/dataset1")
    # print("TRAINING DONE")

    # evaluate('overloadCheckpoints/20-14.32')
    # print("EVALUATE DONE")

    # print_graphs("train_loss_values_mean.txt", training=True, num_epochs=20, steps_per_epoch=2000, valid_steps=20)
    # print_graphs("validate_loss_values_mean.txt")

    # evaluate("modelCheckpoints_10000samples_longTraining/90-80.72")
    # print_graphs("train_loss_values_mean_10000_long.txt", training=True, num_epochs=90, steps_per_epoch=4000, valid_steps=40)
    # print_graphs("validate_loss_values_mean.txt")


    # print("Generating new dataset")
    # trainer = RouteNetTrainer("1000_avgBW", "graph_pre_made.txt")
    # trainer.write_config()
    # trainer.generate_file(1)
    # trainer.start_docker()
    # print("Generating DONE")

    print("Starting generating datasets for Omnet overload test...")
    generate_datasets_for_omnet_overload_test()
    print("DONE")

