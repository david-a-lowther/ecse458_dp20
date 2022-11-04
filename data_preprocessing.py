import csv
import math
import random
import numpy as np


def extract_csv_info(filepath) -> []:
    """
    Extracts most crucial data_simulated from csv file (for the purpose of this project).
    Returns data as list of tuples (H, B)
    """
    file = open(filepath)
    csv_reader = csv.reader(file)
    data = list()
    i = 0
    for row in csv_reader:
        # data doesn't start until row 7
        if csv_reader.line_num >= 7:
            h = float(row[2])
            b = float(row[3])
            # append to data list as (h, b) tuple
            data.append((h, b))
    file.close()
    return data

def extract_csv_info_datasets_folder(filepath) -> []:
    """
    Extracts most crucial data_simulated from csv file (for the purpose of this project).
    Returns data as list of tuples (H, B)
    """
    file = open(filepath, encoding='utf-8-sig')
    csv_reader = csv.reader(file)
    data = list()
    i = 0
    for row in csv_reader:
        h = float(row[0])
        b = float(row[1])
        # append to data list as (h, b) tuple
        data.append((h, b))
    file.close()
    return data


def format_data_preisach(hb_data):
    hb_data = np.array(hb_data)
    return hb_data[:, 0], hb_data[:, 1]

def format_data(hb_data):
    """
    Input: list of tuples (H, B)
    Output: list of tuples (current H, current B, next H, next B)
    """
    formatted_data = list()
    for i, hb_pair in enumerate(hb_data):
        # skip last element
        if i != len(hb_data) - 1:
            current_h = hb_data[i][0]
            current_b = hb_data[i][1]
            next_h = hb_data[i+1][0]
            next_b = hb_data[i+1][1]
            # add as tuple
            formatted_data.append((current_h, current_b, next_h, next_b))

    return formatted_data


# given list of tuples (x, y)
# shuffle and split into training and testing set
# This function needs a lot of work, quick and dirty for now
def shuffle_and_split(total_data, train_percent=80):
    """
    Input: list of tuples
    Output: two lists of tuples, same format as input, one training data, one testing data
    """
    #total_data = np.asarray(total_data)
    # shuffle
    random.shuffle(total_data)
    # split into train and test
    train_size = math.floor(len(total_data) * (train_percent/100))
    #print(train_size)
    train_set = total_data[:train_size]
    test_set = total_data[train_size:]
    return train_set, test_set


def split_input_output(data_set):
    """
    Splits input list of data into input set (all but last element) and output set (last element)
    Input: list of tuples
    Output: list of tuples (input values) and list of floats (output values)
    """
    x_set = list()
    y_set = list()
    last_element_index = len(data_set[0]) - 1
    for data in data_set:
        input_data = data[:last_element_index]
        output_data = data[last_element_index]
        x_set.append(input_data)
        y_set.append(output_data)
    return x_set, y_set


# testing
# raw_data = extract_csv_info("./data_simulated/M19_29Gauge - Sheet1.csv")
# print(raw_data)
# formatted_data = format_data(raw_data)
# print(formatted_data)
#
# train, test = shuffle_and_split(formatted_data)
# print(train)
# print(test)
#
# train_x, train_y = split_input_output(train)
# test_x, test_y = split_input_output(test)

