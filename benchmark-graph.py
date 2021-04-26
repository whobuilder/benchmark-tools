from collections import defaultdict
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from complexity import Complexity


class BMResult:
    def __init__(self, filenames):
        self.results = defaultdict(lambda: defaultdict(dict))
        self.row_info = {}
        self.sizes = set()
        self.time_units = set()
        self.csv_file_paths = filenames

    def read_info_row(self, info_row):
        for index, tag in enumerate(info_row):
            self.row_info[tag] = index

    def read_result_row(self, result_row):
        name = result_row[self.row_info["name"]]
        if name.endswith("_BigO") or name.endswith("_RMS"):
            return
        try:
            benchmark_properties = name.split("/")
            label = benchmark_properties[0]
            size = benchmark_properties[1]
            size = int(size)
            self.sizes.add(size)
        # pylint: disable=broad-except
        except Exception as exc:
            logging.getLogger("BMResult").exception(exc)
        for tag, index in self.row_info.items():
            if tag == "time_unit":
                self.time_units.add(result_row[index])
            if tag not in ["iterations", "real_time", "cpu_time"]:
                continue
            self.results[label][size][tag] = float(result_row[index])

    def read_csv_file(self):
        for path in self.csv_file_paths:
            with open(path) as csv_file:
                reader = csv.reader(csv_file, delimiter=",")
                start_reading_results = False
                for row in reader:
                    if start_reading_results:
                        self.read_result_row(row)
                    elif row[0] == "name":
                        self.read_info_row(row)
                        start_reading_results = True
        if len(self.time_units) > 1:
            raise IOError("Inconsistent time units")
        self.sizes = sorted(self.sizes)

    def print_benchmark_names(self):
        for name in self.results.keys():
            print(name)

    def plot_complexity(self, plt_ax, tag, to_plot=None):
        for name in self.results.keys():
            if to_plot is not None and name not in to_plot:
                continue

            values = np.array([[s, self.results[name][s][tag]] for s in self.results[name]])

            complexity = Complexity(values[:, 0], values[:, 1], Complexity.BigO.oAuto)
            complexity.calculate_complexity()
            generic_x = np.linspace(values[:, 0][0], values[:, 0][-1], len(values[:, 0]) * 50)
            big_o_plt = plt_ax.plot(
                generic_x,
                Complexity.get_curve_fit_func(complexity.big_o)(generic_x, complexity.coeff),
                "--",
                linewidth=3,
                label=f"{complexity.big_o.name}(coeff:{complexity.coeff[0]:.1f}, rms:{100*complexity.rms:.1f}%",
            )
            plt_ax.plot(
                values[:, 0],
                values[:, 1],
                "v-",
                linewidth=2,
                markersize=6,
                markeredgewidth=3,
                color=big_o_plt[0].get_color(),
                label=name,
            )
        plt_ax.legend()
        plt_ax.grid()

    @staticmethod
    def annotate_bars(rects, plt_ax):
        """Attach a text label above each bar in *rects*,
        displaying its height."""
        for rect_group in zip(*rects):
            max_height = max(rect.get_height() for rect in rect_group)
            for rect in rect_group:
                height = rect.get_height()
                rel_h = 100 * (height / max_height - 1)
                height_annotation = f"{rel_h:.1f}%" if (rel_h != 0) else "-"
                if rel_h == -100:
                    continue
                plt_ax.annotate(
                    height_annotation,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

    def plot_bar_graph(self, plt_ax, tag, to_plot=None):
        single_bar_width = 0.70
        xticks = np.arange(len(self.sizes))  # the label locations
        bar_count = len(self.results) if (to_plot is None) else len(to_plot)
        width = single_bar_width / bar_count  # the width of the bars
        bars = []
        ith = 0
        for name in self.results.keys():
            if to_plot is not None and name not in to_plot:
                continue
            values = [
                self.results[name][s][tag] if s in self.results[name] else 0 for s in self.sizes
            ]
            pos = xticks - (bar_count / 2) * (width) + ith * width + width / 2
            bars.append(plt_ax.bar(pos, values, width, label=name))
            ith += 1
        self.annotate_bars(bars, plt_ax)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        plt_ax.set_ylabel(f"time ({list(self.time_units)[0]})")
        plt_ax.set_xlabel(f"benchmark argument")
        plt_ax.set_title(tag)
        plt_ax.set_xticks(xticks)
        plt_ax.set_xticklabels(self.sizes)
        plt_ax.legend()
        plt_ax.grid()


def do_plotting(bm_result, tags, plot_complexity, plot_bar, output_file, to_plot=None):
    col_size = 2 if not plot_complexity and not plot_bar else 1
    fig, axes = plt.subplots(len(tags), col_size, sharex=False, sharey=False)
    fig.set_size_inches(12, 10)
    axes = np.resize(axes, [len(tags), col_size])
    for row, item in enumerate(tags):
        if plot_bar:
            bm_result.plot_bar_graph(axes[row][0], item, to_plot)
        if plot_complexity:
            bm_result.plot_complexity(axes[row][0], item, to_plot)
        if not plot_bar and not plot_complexity:
            bm_result.plot_bar_graph(axes[row][0], item)
            bm_result.plot_complexity(axes[row][1], item, to_plot)
    if output_file is not None:
        fig.savefig(output_file)


def process_arguments(arguments):
    bm_result = BMResult(arguments["PATH"])
    bm_result.read_csv_file()
    if arguments["--list-benchmarks"]:
        bm_result.print_benchmark_names()
    else:
        print(arguments)
        tags = []
        if arguments["--cpu"]:
            tags.append("cpu_time")
        if arguments["--real"]:
            tags.append("real_time")
        to_plot = arguments["--filter"].split(",") if arguments["--filter"] is not None else None
        do_plotting(
            bm_result,
            tags,
            arguments["--plot-complexity"],
            arguments["--plot-bar"],
            arguments["--output"],
            to_plot,
        )


def get_arguments():
    description = """A program to visualize the google benchmark results.
    Usage:
        options_example.py  --list-benchmarks PATH...
        options_example.py  (--cpu | --real | --cpu --real) [--plot-complexity | --plot-bar] [--filter=benchmarks] [--output=FILE] PATH...

    Arguments:
        PATH  destination path to the csv files

    Options:
        -h --help               show this help message and exit
        --version               show version and exit
        --cpu                   use cpu times in plot
        --real                  use real times in plot
        --plot-complexity       plot complexity graph
        --plot-bar              plots bar graph
        --filter=benchmarks     comma separated benchmark names to plot (e.g. BM1, BM2)
        --output=FILE           output file
        --list-benchmarks       list the available benchmark names
    """
    return docopt(description, version="1.0rc")


if __name__ == "__main__":
    process_arguments(get_arguments())
    plt.show()
