import os
import json
import csv
import argparse

# calculate the results with the colected data to build function_profile_table
def calculate_data(data_dir, function_name):
    # data
    sub_dir_names = os.listdir(data_dir)
    sub_dir_names.sort()
    for sub_dir_name in sub_dir_names:
        instance_num, rps = sub_dir_name.split('-')
        file_names = os.listdir('{}/{}'.format(data_dir, sub_dir_name))
        file_names.sort()
        # 有序
        for file_name in file_names:
            if file_name.endswith('function_status.json'):
                with open('{}/{}/{}'.format(data_dir, sub_dir_name, file_name), 'r') as f:
                    status_arr = json.load(f)
                    for status in status_arr:
                        if status['name'] == function_name:
                            if status['availableReplicas'] != int(instance_num):
                                print('{}/{}/{}'.format(data_dir, sub_dir_name, file_name))
                                print(status['availableReplicas'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function-name', type=str, default='hello-python')
    parser.add_argument('--save-root', type=str, default='data', help='the path to save the collected data')
    args = parser.parse_known_args()[0]

    calculate_data(args.save_root, args.function_name)
