# A python script to plot values from a .csv file.
import csv
import os
import matplotlib.pyplot as plt
import threading


def plot_csv_magnetic(filename):
    """Plots the H vs. B data from a csv file"""
    L = extract_csv_info(filename)
    plt.plot(L[0], L[1], marker="o", color='black')
    plt.title("Hysteresis data from " + filename)
    plt.xlabel("Magnetic Field H (T)")
    plt.ylabel("Magnetic Flux B (A/m)")
    plt.show()


def plot_csv_energy(filename, type):
    """Plots energy based data from csv file containing H vs. B data"""
    L = extract_csv_info(filename)
    E = get_energy_data(filename)
    if type == "B":
        plt.plot(L[1], E, marker="o", color='red')
        plt.title("Hysteresis data from " + filename)
        plt.xlabel("Magnetic Flux B (A/m)")
        plt.ylabel("Energy of the material (TA/m)")
        plt.show()
    elif type == "H":
        plt.plot(L[0], E, marker="o", color='red')
        plt.title("Hysteresis data from " + filename)
        plt.xlabel("Magnetic Field H (T)")
        plt.ylabel("Energy of the material (TA/m)")
        plt.show()


def get_energy_data(filename) -> []:
    """Integrates H vs B data to give energy data"""
    H, B = extract_csv_info(filename)
    energy = [0]*len(B)
    for i in range(len(B)-1):
        # Integrate
        energy[i+1] += H[i]*(B[i+1]-B[i]) + ((H[i+1]-H[i])*(B[i+1]-B[i]))/2.0
    return energy


def extract_csv_info(filename) -> []:
    """Extracts most crucial data from csv file (for the purpose of this project)"""
    file = open('data/' + filename)
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


if __name__ == '__main__':
    print("\nProgram to plot data from csv files.\n")
    print("List of commands:")
    print("ls = list of files that can be plotted\n"
          "x  = exit\n"
          "plot <filename> Plots the file in question, if it is contained in the data folder. Option -e plots the "
          "energy based version\n\n"
          "The program will continue after the plot window is closed.\n")
    while True:
        cmd = input(">> ")
        if cmd.lower() == "x":
            break
        elif cmd.lower() == "ls":
            print(*os.listdir("data"), sep="\n")
            continue
        elif cmd[0:4].lower() == "plot":
            if cmd.strip()[-2:] == "-e":
                u = input(">> B to plot with respect to B, H to plot with respect to H: ")
                if u.lower() == "b":
                    try:
                        plot_csv_energy(cmd[5: (len(cmd)) - 2].strip(), "B")
                    except PermissionError or FileNotFoundError or Exception:
                        print("Error: The file is not contained in the data folder. Type ls to see the available files.")
                elif u.lower() == "h":
                    try:
                        plot_csv_energy(cmd[5: (len(cmd)) - 2].strip(), "H")
                    except PermissionError or FileNotFoundError or Exception or OSError:
                        print("Error: The file is not contained in the data folder. Type ls to see the available files.")
            else:
                try:
                    plot_csv_magnetic(cmd[5:].strip())
                except PermissionError or FileNotFoundError or Exception or OSError:
                    print("Error: The file is not contained in the data folder. Type ls to see the available files.")
        else:
            print("Error: Not a recognized command")
