import numpy as np

import upcdataset as up
import yaml
import os
import shutil
import datanetAPI
import statistics

from getpass import getpass

SUDO_PASS = f"echo {getpass()} | sudo -S " if os.name != 'nt' else ''


class new_delay:
    def __init__(self, nedfile, topology_file):
        (self.connections,
         self.n,
         self.edges,
         self.capacities) = up.ned2lists(nedfile) # I am not sure where the NED is used, it seems like omnet doesnt use it somehow


        self.topology_file=topology_file

        self.training_dataset_path = "training"

        self.graphs_path = "graphs"
        self.routings_path = "routings"
        self.tm_path = "tm"

        self.simulation_file = os.path.join(self.training_dataset_path, "simulation.txt")
        self.dataset_name = "dataset1"

        self.create_path_directories() # make sure the directories exist, if not then create them
        self.setup_conf()

    def create_path_directories(self):
        if (os.path.isdir(self.training_dataset_path)):
            print("Destination path already exists. Files within the directory may be overwritten.")
        else:
            os.makedirs(os.path.join(self.training_dataset_path, self.graphs_path))
            os.mkdir(os.path.join(self.training_dataset_path, self.routings_path))
            os.mkdir(os.path.join(self.training_dataset_path, self.tm_path))


    def setup_conf(self):
        self.conf_file = os.path.join(self.training_dataset_path, "conf.yml")
        conf_parameters = {
            "threads": 6,  # Number of threads to use
            "dataset_name": self.dataset_name,
            # Name of the dataset. It is created in <training_dataset_path>/results/<name>
            "samples_per_file": 1,  # Number of samples per compressed file - we only have 1. EDIT: Now we have 50. #TODO: Is 50 the right amount?
            "rm_prev_results": "n",  # If 'y' is selected and the results folder already exists, the folder is removed.
        }
        with open(self.conf_file, 'w') as fd:
            yaml.dump(conf_parameters, fd)

    def sudo_command(self,raw_cmd,terminal_cmd):
        if os.name != 'nt':  # Unix, requires sudo
            print("Superuser privileges are required to run docker. Introduce sudo password when prompted")
            # terminal_cmd = f"echo {getpass()} | sudo -S " + raw_cmd
            terminal_cmd = SUDO_PASS + raw_cmd
            raw_cmd = "sudo " + raw_cmd
        return terminal_cmd,raw_cmd

    def generate_docker_cmd(self,training_dataset_path):
        raw_cmd = f"docker run --rm --mount type=bind,src={os.path.join(os.getcwd(), training_dataset_path)},dst=/data bnnupc/netsim:v0.1"
        terminal_cmd = raw_cmd
        return self.sudo_command(raw_cmd, terminal_cmd)

    def delete_results_dir(self):
        if os.path.exists('training/results'):
            if os.name != 'nt':  # Unix
                os.system("sudo chmod -R 777 training/results")
            shutil.rmtree('training/results')

    def run_omnet_docker(self):
        self.delete_results_dir()
        # running the docker
        raw_cmd, terminal_cmd = self.generate_docker_cmd(self.training_dataset_path)
        # print('cmd /c ' + terminal_cmd)
        if os.name != 'nt':  # Unix
            os.system(terminal_cmd)
        else:
            os.system('cmd /c '+terminal_cmd)

    def read_simulation_result(self):
        print("test")
        with open(self.simulation_file, "r") as file:
            result = file.readline().split('OK')[0]
            print("===== Simulation result =====")
            print(result)
            return result

    def setup_files(self,tm_file,routing_file):
        # print('===============================')
        # print("ROUTING FILE IS:::::")
        # print(routing_file)
        # print(self.routings_path)
        self.routing_file=routing_file
        self.tm_file=tm_file
        shutil.copy(routing_file, os.path.join(self.training_dataset_path, self.routings_path))
        shutil.copy(tm_file, os.path.join(self.training_dataset_path, self.tm_path))
        shutil.copy(self.topology_file, os.path.join(self.training_dataset_path, self.graphs_path))
        self.create_simulation_file()
    def create_simulation_file(self):
        with open (self.simulation_file,"w") as fd:
            # for _ in range(50):
            sim_line = "{},{},{}\n".format(self.graphs_path+"/"+self.topology_file,self.routings_path+"/"+self.routing_file,self.tm_path+"/"+self.tm_file)
             # If dataset has been generated in windows, convert paths into linux format
            fd.write(sim_line.replace("\\","/"))

    def get_mean_delay(self):
        data_folder_name = "training"
        src_path = f"{data_folder_name}/results/dataset1/"
        max_avg_lambda_range = [10, 10000]
        net_size_lst = [10]
        reader = datanetAPI.DatanetAPI(src_path,max_avg_lambda_range, net_size_lst)

        samples_lst = []
        for sample in reader:
            samples_lst.append(sample)
        print("Number of selected samples: ", len(samples_lst))

        delays_lst = []
        for s in samples_lst:
            performance_matrix = s.get_performance_matrix()
            delays_lst_per_sample = []
            for i in range(s.get_network_size()):
                for j in range(s.get_network_size()):
                    if (i == j):
                        continue
                    # Append to the list the average delay of the path i,j.
                    delays_lst_per_sample.append(performance_matrix[i, j]["AggInfo"]["AvgDelay"])
            delays_lst.append([np.array(delays_lst_per_sample)])
            # print(delays_lst_per_sample)

        final_result = np.median(delays_lst, axis=0)
        # print("===== Final result is =====")
        # print(final_result)
        return final_result

    def calc(self, traffic, routing):
        self.setup_files(traffic,routing) # will copy our files to the path of the OMNET
        a=self.run_omnet_docker()
        print(a)
        print("finished running docker")
        print("GETTING THE RESULTS...")
        return self.get_mean_delay()
        # return 1
        # return self.read_simulation_result()