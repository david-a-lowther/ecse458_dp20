# A python script to plot values from a .csv file.
import csv
import click
import os
import matplotlib.pyplot as plt
from plot import (
    plot_csv_magnetic,
    plot_csv_energy
)

@click.group()
def main_group():
    pass


@main_group.group('plot')
def plot_group():
    pass

@main_group.group('nn')
def neural_group():
    pass

plot_group.add_command(plot_csv_magnetic, "std")
plot_group.add_command(plot_csv_energy, "energy")


def main():
    main_group()


if __name__ == '__main__':
    main()
    print("\nProgram to plot data_simulated from csv files.\n")
    print("List of commands:")
    print("ls = list of files that can be plotted\n"
          "x  = exit\n"
          "plot <filename> Plots the file in question, if it is contained in the data_simulated folder. Option -e plots the "
          "energy based version\n\n"
          "The program will continue after the plot window is closed.\n")
    while True:
        cmd = input(">> ")
        if cmd.lower() == "x":
            break
        elif cmd.lower() == "ls":
            print(*os.listdir("data_simulated"), sep="\n")
            continue
        elif cmd[0:4].lower() == "plot":
            if cmd.strip()[-2:] == "-e":
                u = input(">> B to plot with respect to B, H to plot with respect to H: ")
                if u.lower() == "b":
                    try:
                        plot_csv_energy(cmd[5: (len(cmd)) - 2].strip(), "B")
                    except PermissionError or FileNotFoundError or Exception:
                        print("Error: The file is not contained in the data_simulated folder. Type ls to see the available files.")
                elif u.lower() == "h":
                    try:
                        plot_csv_energy(cmd[5: (len(cmd)) - 2].strip(), "H")
                    except PermissionError or FileNotFoundError or Exception or OSError:
                        print("Error: The file is not contained in the data_simulated folder. Type ls to see the available files.")
            else:
                try:
                    plot_csv_magnetic(cmd[5:].strip())
                except PermissionError or FileNotFoundError or Exception or OSError:
                    print("Error: The file is not contained in the data_simulated folder. Type ls to see the available files.")
        else:
            print("Error: Not a recognized command")
