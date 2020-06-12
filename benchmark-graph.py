import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import logging
import numpy as np
from docopt import docopt

class BMResult:
    def __init__(self, filenames):
        self.results = defaultdict(lambda: defaultdict(dict)) 
        self.row_info = {}
        self.sizes =  set()
        self.csv_file_paths = filenames
    def read_info_row(self, info_row):
        for index, tag in enumerate(info_row):
            self.row_info[tag] = index

    def read_result_row(self, result_row):
        if (result_row[self.row_info["name"]].endswith("_BigO") or result_row[self.row_info["name"]].endswith("_RMS")): return
        try:
            label, size = result_row[self.row_info["name"]].split("/")
            size=int(size)
            self.sizes.add(size)
        except Exception as e:
            logging.getLogger("BMResult").exception(e)
        for tag, index in self.row_info.items():
            if (not (tag == "iterations" or tag == "real_time" or tag == "cpu_time")): continue
            self.results[label][size][tag] = float(result_row[index])

    def read_csv_file(self):
        for path in self.csv_file_paths:
            with open(path) as f:
                reader = csv.reader(f, delimiter=",")
                start_reading_results = False
                for row in reader:
                    if start_reading_results:
                        self.read_result_row(row)
                    elif (row[0] == "name"):
                        self.read_info_row(row)
                        start_reading_results = True
        self.sizes = sorted(self.sizes)

    def plot_algorithm_time_efficiency(self, ax, plot_item, names_to_plot=None):
        length_of_values = len(self.sizes)
        size_factors = np.array(self.sizes)/self.sizes[0]
        n_factors = size_factors
        n_2_factors = size_factors**2
        log_n_factors = np.log2(size_factors)
        n_log_n_factors = np.log2(size_factors) * size_factors
        ax.plot(size_factors, n_factors, "+--", linewidth=3, markersize=5, markeredgewidth=2, label="O(n)")
        ax.plot(size_factors, n_2_factors, "*--", linewidth=3, markersize=5, markeredgewidth=2, label="O(n**2)")
        ax.plot(size_factors, log_n_factors, "x--", linewidth=3, markersize=5, markeredgewidth=2, label="O(logn)")
        ax.plot(size_factors, n_log_n_factors, "o--", linewidth=3, markersize=5, markeredgewidth=2, label="O(nlogn)")
        max_factor = 0
        for ith, name in enumerate(self.results.keys()):
            if names_to_plot is not None and name not in names_to_plot:continue
            values = np.array([self.results[name][s][plot_item] if s in self.results[name] else 0 for s in self.sizes ])
            values_factor = values / values[0]
            ax.plot(size_factors, values_factor, "v-", linewidth=2, markersize=6, markeredgewidth=3, label=name)
            max_factor_new = max(values_factor)
            if (max_factor_new > max_factor):
                max_factor = max_factor_new
        ax.legend()
        ax.grid()
        ax.set_ylim([0,max_factor+0.05*max_factor])

        ax.set_ylabel('time factor')
        ax.set_xlabel('size factor')
        ax.set_title(plot_item)

    @staticmethod
    def annotate_bars(rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect_group in zip(*rects):
            max_height = 0
            if (len(rect_group) < 2): return
            for r in rect_group:
                height = r.get_height()
                if (height > max_height):
                    max_height = height
            for r in rect_group:
                height = r.get_height()
                rel_height = 100*(height/max_height - 1)
                rel_height_annotation = f"{rel_height:.1f}%" if (rel_height != 0) else "-"
                if (rel_height == -100): continue
                ax.annotate(rel_height_annotation,
                            xy=(r.get_x() + r.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    def plot_bar_graph(self, ax, plot_item, names_to_plot=None):
        single_bar_width = 0.70
        x = np.arange(len(self.sizes))  # the label locations
        number_of_bar_types = len(self.results) if (names_to_plot is None) else len(names_to_plot)
        width = single_bar_width / number_of_bar_types  # the width of the bars
        bars = []
        ith = 0
        for name in self.results.keys():
            if names_to_plot is not None and name not in names_to_plot:continue
            values = [self.results[name][s][plot_item] if s in self.results[name] else 0 for s in self.sizes ]
            pos = x - (number_of_bar_types/2)*(width) + ith*width+width/2
            bars.append( ax.bar(pos, values, width, label=name))
            ith+=1
        self.annotate_bars(bars, ax)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(f'time (ns)')
        ax.set_xlabel(f'benchmark argument')
        ax.set_title(plot_item)
        ax.set_xticks(x)
        ax.set_xticklabels(self.sizes)
        ax.legend()
        ax.grid()

def do_plotting(plot_items, factor_plot, output_file, file_paths):
    bm = BMResult(file_paths)
    bm.read_csv_file()
    column_size = 2 if factor_plot else 1
    fig, axes = plt.subplots(len(plot_items), column_size, sharex=False, sharey=False)
    fig.set_size_inches(12,10)
    axes = np.resize(axes, [len(plot_items), column_size])
    for row, item in enumerate(plot_items):
        if factor_plot:
            bm.plot_bar_graph(axes[row][0], item)
            bm.plot_algorithm_time_efficiency(axes[row][1], item)
        else:
            bm.plot_bar_graph(axes[row][0], item)
    if output_file is not None:
        fig.savefig(output_file)

def process_arguments(arguments):
    print(arguments)
    plot_items = []
    if(arguments["--cpu"]):
        plot_items.append("cpu_time")
    if (arguments["--real"]):
        plot_items.append("real_time")

    do_plotting(plot_items, arguments["--plot-factor"], arguments["--output"], arguments["PATH"])

def get_arguments():
    description="""A program to visualize the google benchmark results.
    Usage:
        options_example.py  (--cpu | --real | --cpu --real) [--plot-factor] [--output=FILE] PATH...

    Arguments:
        PATH  destination path to the csv files

    Options:
        -h --help               show this help message and exit
        --version               show version and exit
        --cpu                   use cpu times in plot
        --real                  use real times in plot
        --plot-factor           plot factor graph
        --output=FILE           output file
    """
    return docopt(description, version="1.0rc")

if (__name__ == "__main__"):
    process_arguments(get_arguments())
    plt.show()
