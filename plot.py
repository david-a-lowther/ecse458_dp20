import csv
import click
import os
import matplotlib.pyplot as plt


@click.command('magnetic')
def plot_csv_magnetic(filename):
    """Plots the H vs. B data_simulated from a csv file"""
    L = extract_csv_info(filename)
    plt.plot(L[0], L[1], marker="o", color='black')
    plt.title("Hysteresis data_simulated from " + filename)
    plt.xlabel("Magnetic Field H (T)")
    plt.ylabel("Magnetic Flux B (A/m)")
    plt.show()


@click.command('energy')
def plot_csv_energy(filename, type):
    """Plots energy based data_simulated from csv file containing H vs. B data_simulated"""
    L = extract_csv_info(filename)
    E = get_energy_data(filename)
    if type == "B":
        plt.plot(L[1], E, marker="o", color='red')
        plt.title("Hysteresis data_simulated from " + filename)
        plt.xlabel("Magnetic Flux B (A/m)")
        plt.ylabel("Energy of the material (TA/m)")
        plt.show()
    elif type == "H":
        plt.plot(L[0], E, marker="o", color='red')
        plt.title("Hysteresis data_simulated from " + filename)
        plt.xlabel("Magnetic Field H (T)")
        plt.ylabel("Energy of the material (TA/m)")
        plt.show()


def get_energy_data(filename) -> []:
    """Integrates H vs B data_simulated to give energy data_simulated"""
    H, B = extract_csv_info(filename)
    energy = [0]*len(B)
    for i in range(len(B)-1):
        # Integrate
        energy[i+1] += H[i]*(B[i+1]-B[i]) + ((H[i+1]-H[i])*(B[i+1]-B[i]))/2.0
    return energy


def extract_csv_info(filename) -> []:
    """Extracts most crucial data_simulated from csv file (for the purpose of this project)"""
    file = open('data_simulated/' + filename)
    rows = []
    csvreader = csv.reader(file)
    H, B = [], []
    i = 0
    for row in csvreader:
        if i < 7:
            i += 1
            continue
        rows.append(row)
        H.append(float(row[2]))
        B.append(float(row[3]))
    file.close()
    return H, B