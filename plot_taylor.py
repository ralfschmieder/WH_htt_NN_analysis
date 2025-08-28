import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot categories using Dumbledraw from shapes produced by shape-producer module."
    )
    parser.add_argument(
        "-i", "--input_file", type=str, required=True, help="Enable linear x-axis"
    )
    parser.add_argument(
        "-c", "--channel", type=str, required=True, help="Enable linear x-axis"
    )
    parser.add_argument(
        "-r", "--rank", type=int, required=True, help="Enable linear x-axis"
    )
    return parser.parse_args()


vars = [
    "pt_3",
    "eta_3",
    "deltaR_13",
    "deltaR_23",
    "deltaR_12",
    "pt_1",
    "eta_1",
    "pt_2",
    "eta_2",
    "m_vis",
    "Lt",
    "met",
    "m_tt",
    "mt_1",
    "mt_2",
    "mt_3",
    "pt_vis",
    "pt_W",
    "pt_123met",
    "njets",
    "jpt_1",
    "jpt_2",
]


# Function to plot the data from the CSV file
def plot_csv_data(csv_file, channel, rank):
    # Load the CSV file using numpy, skip the first line (header)
    data = np.genfromtxt(csv_file, delimiter="\n", dtype=str, skip_header=1)
    csv_path = csv_file.split("/")[:-1]  # + "taylor_ranks_allnodes"
    plot_output = (
        csv_path[0]
        + "/"
        + csv_path[1]
        + "/"
        + csv_path[2]
        + "/"
        + "taylor_ranks_node_0"
    )
    # Extract variables (column 0) and values (column 1)
    values = np.array([])
    variables = np.array([])
    for line in data:
        # print(line, "##", line.split('"')[-2], line.split('"')[-1].replace(",", ""))
        variables = np.append(variables, line.split('"')[-2])
        values = np.append(values, abs(float(line.split('"')[-1].replace(",", ""))))

    tuples = list(zip(variables, values))
    sorted_tuples = sorted(tuples, key=lambda x: x[1])
    top_entries = sorted_tuples[-rank:]
    top_vars = [
        entry[0].replace(",", " vs.").replace("'", "").replace("(", "").replace(")", "")
        for entry in top_entries
    ]

    top_vars_2 = []
    for var_, var in enumerate(top_vars):
        if not "vs. " in var:
            top_vars_2.append(var.replace("vs.", ""))
        else:
            top_vars_2.append(var)

    # top_vars = [entry.replace(" vs.", "") for entry in top_vars if not "vs. " in entry]
    top_values = [entry[1] for entry in top_entries]
    print(top_vars)
    plt.plot(top_values, top_vars_2, marker=".", markersize=9, linestyle="")
    plt.xlabel(r"$\langle t_i \rangle$")
    plt.tight_layout()
    plt.grid()
    plt.savefig(
        "{plot_output}_test".format(plot_output=plot_output)
        + "_{ch}".format(ch=channel)
        + ".png"
    )
    plt.savefig(
        "{plot_output}_test".format(plot_output=plot_output)
        + "_{ch}".format(ch=channel)
        + ".pdf"
    )


def print_top_taylor(csv_file, channel, rank):
    # Load the CSV file using numpy, skip the first line (header)
    data = np.genfromtxt(csv_file, delimiter="\n", dtype=str, skip_header=1)
    values = np.array([])
    variables = np.array([])
    taylors = {}
    for line in data:
        # print(line, "##", line.split('"')[-2], line.split('"')[-1].replace(",", ""))
        variables = np.append(variables, line.split('"')[-2])
        values = np.append(values, abs(float(line.split('"')[-1].replace(",", ""))))
    for key_str, value in zip(variables, values):
        # Remove the surrounding "(' and ',)" characters and format the key
        formatted_key = key_str.strip("(')").replace("', '", ", ")
        taylors[formatted_key] = abs(value)
    taylor_sum = {}
    for var in vars:
        taylor_sum[var] = 0.0
        for key in taylors.keys():
            if var in key:
                taylor_sum[var] += taylors[key]
    sorted_taylor = dict(sorted(taylor_sum.items(), key=lambda item: item[1]))
    # äprint(list(sorted_taylor.keys())[-15:])
    max_value = max(sorted_taylor.values())
    threshold = 0.2 * max_value
    keys_with_10_percent = [
        key for key, value in sorted_taylor.items() if value >= threshold
    ]
    print(keys_with_10_percent)
    return taylors


def get_top_variables():
    emt = [
        "deltaR_13",
        "eta_2",
        "m_vis",
        "jpt_2",
        "deltaR_12",
        "mt_2",
        "mt_3",
        "njets",
        "Lt",
        "pt_3",
        "pt_W",
        "pt_123met",
        "pt_2",
        "pt_1",
        "met",
    ]
    mmt = [
        "jpt_1",
        "deltaR_12",
        "mt_2",
        "deltaR_13",
        "deltaR_23",
        "mt_3",
        "mt_1",
        "m_vis",
        "pt_W",
        "pt_3",
        "njets",
        "pt_123met",
        "pt_2",
        "pt_1",
        "met",
    ]
    met = [
        "deltaR_12",
        "deltaR_23",
        "deltaR_13",
        "pt_vis",
        "eta_1",
        "jpt_1",
        "njets",
        "mt_2",
        "pt_3",
        "Lt",
        "pt_W",
        "pt_123met",
        "pt_2",
        "pt_1",
        "met",
    ]
    mtt = [
        "jpt_1",
        "mt_3",
        "deltaR_13",
        "deltaR_12",
        "mt_2",
        "pt_W",
        "pt_123met",
        "pt_vis",
        "Lt",
        "m_vis",
        "pt_3",
        "m_tt",
        "pt_2",
        "pt_1",
        "met",
    ]
    ett = [
        "mt_1",
        "deltaR_13",
        "mt_2",
        "deltaR_23",
        "pt_W",
        "deltaR_12",
        "pt_vis",
        "pt_123met",
        "m_tt",
        "pt_3",
        "m_vis",
        "Lt",
        "pt_2",
        "pt_1",
        "met",
    ]
    set_emt = set(emt)
    set_mmt = set(mmt)
    set_met = set(met)
    set_ett = set(ett)
    set_mtt = set(mtt)
    # common = set_emt & set_mmt & set_met
    # all = set_emt | set_mmt | set_met
    common = set_mtt & set_ett & set_emt & set_mmt & set_met
    all = set_mtt | set_ett | set_emt | set_mmt | set_met
    not_in_all = all - common
    print("all: ", all)
    print("not in all: ", not_in_all)
    print(len(all))


def main(args):
    csv_file = args.input_file
    channel = args.channel
    rank = args.rank
    plot_csv_data(csv_file, channel, rank)
    # print_top_taylor(csv_file, channel, rank)
    # get_top_variables()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
