"""
Script for preprocessing n-tuples for the neural network training
"""

import os
import argparse
import yaml
import multiprocessing

from io import StringIO
from wurlitzer import pipes, STDOUT
import logging
from typing import Tuple, Dict, Union, List
import ROOT

import helper.filters as filters
import helper.weights as weights
import helper.functions as func


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config-file",
    default=None,
    help="Path to the config file which contains information for the preselection step.",
)
parser.add_argument(
    "--nthreads",
    default=8,
    help="Number of threads to use for the preselection step. (default: 8)",
)


def run_preselection(args: Tuple[str, Dict[str, Union[Dict, List, str]]]) -> None:
    """
    This function can be used for multiprocessing. It runs the preselection step for a specified process.

    Args:
        args: Tuple with a process name and a configuration for this process

    Return:
        None
    """
    process, config = args
    log = logging.getLogger(f"preselection.{process}")

    log.info(f"Processing process: {process}")
    # bookkeeping of samples files due to splitting based on the tau origin (genuine, jet fake, lepton fake)
    process_file_dict = dict()
    for tau_gen_mode in config["processes"][process]["tau_gen_modes"]:
        process_file_dict[tau_gen_mode] = list()
    log.info(
        f"Considered samples for process {process}: {config['processes'][process]['samples']}"
    )
    # define output variables
    outputs = config["output_feature"]
    era = config["era"]
    # going through all contributing samples for the process
    for idx, sample in enumerate(config["processes"][process]["samples"]):
        # loading ntuple files
        ntuple_list = func.get_ntuples(config, process, sample)
        chain = ROOT.TChain(config["tree"])

        for ntuple in ntuple_list:
            chain.Add(ntuple)

        if "friends" in config:
            for friend in config["friends"]:
                friend_list = []
                for ntuple in ntuple_list:
                    friend_list.append(
                        ntuple.replace("CROWNRun", "CROWNFriends/" + friend)
                    )
                fchain = ROOT.TChain(config["tree"])
                for friend in friend_list:
                    fchain.Add(friend)
                chain.AddFriend(fchain)

        rdf = ROOT.RDataFrame(chain)

        if func.rdf_is_empty(rdf=rdf):
            log.info(f"WARNING: Sample {sample} is empty. Skipping...")
            continue

        rdf = func.add_mass_variables(rdf=rdf, sample=sample)

        if config["event_split"] == "even":
            rdf = rdf.Filter("(event%2==0)", "cut on even event IDs")
        elif config["event_split"] == "odd":
            rdf = rdf.Filter("(event%2==1)", "cut on odd event IDs")
        else:
            pass

        # apply general analysis event filters
        selection_conf = config["general_event_selection"]
        for cut in selection_conf:
            rdf = rdf.Filter(f"({selection_conf[cut]})", f"cut on {cut}")
        # define the one hot encoded era
        if era == "2016preVFP":
            rdf = rdf.Define("era_2016preVFP", "1")
            rdf = rdf.Define("era_2016postVFP", "0")
            rdf = rdf.Define("era_2017", "0")
            rdf = rdf.Define("era_2018", "0")
        elif era == "2016postVFP":
            rdf = rdf.Define("era_2016preVFP", "0")
            rdf = rdf.Define("era_2016postVFP", "1")
            rdf = rdf.Define("era_2017", "0")
            rdf = rdf.Define("era_2018", "0")
        elif era == "2017":
            rdf = rdf.Define("era_2016preVFP", "0")
            rdf = rdf.Define("era_2016postVFP", "0")
            rdf = rdf.Define("era_2017", "1")
            rdf = rdf.Define("era_2018", "0")
        elif era == "2018":
            rdf = rdf.Define("era_2016preVFP", "0")
            rdf = rdf.Define("era_2016postVFP", "0")
            rdf = rdf.Define("era_2017", "0")
            rdf = rdf.Define("era_2018", "1")

        # calculate event weights
        rdf = rdf.Define("weight", "1.")

        mc_weight_conf = config["mc_weights"]
        if process not in ["tau_fakes", "lep_fakes"]:
            for weight in mc_weight_conf:
                if weight == "generator":
                    rdf = weights.gen_weight(rdf=rdf, sample_info=datasets[sample])
                elif weight == "lumi":
                    rdf = weights.lumi_weight(rdf=rdf, era=config["era"])
                elif weight == "Z_pt_reweighting":
                    if process == "DYjets":
                        rdf = rdf.Redefine(
                            "weight", f"weight * ({mc_weight_conf[weight]})"
                        )
                elif weight == "Top_pt_reweighting":
                    if process == "ttbar":
                        rdf = rdf.Redefine(
                            "weight", f"weight * ({mc_weight_conf[weight]})"
                        )
                else:
                    rdf = rdf.Redefine("weight", f"weight * ({mc_weight_conf[weight]})")

        # apply special analysis event filters: tau vs jet ID, btag
        selection_conf = config["special_event_selection"]
        if process == "tau_fakes":
            if "tau_id_vs_jet" in selection_conf:
                rdf = rdf.Filter(
                    f"({selection_conf['tau_id_vs_jet'][1]})",
                    "cut on tau_id_vs_jet",
                )
                if config["channel"] in ["emt", "mmt"]:
                    rdf = rdf.Filter(
                        f"({selection_conf['mu_2_idiso'][0]})",
                        "cut on mu_2_idiso",
                    )
                elif config["channel"] == "met":
                    rdf = rdf.Filter(
                        f"({selection_conf['ele_2_idiso'][0]})",
                        "cut on ele_2_idiso",
                    )
                rdf = weights.apply_tau_fake_factors(rdf=rdf, channel=config["channel"])
        elif process == "lep_fakes":
            rdf = rdf.Filter(
                f"({selection_conf['tau_id_vs_jet'][0]})",
                "cut on tau_id_vs_jet",
            )
            if config["channel"] in ["emt", "mmt"]:
                if "mu_2_idiso" in selection_conf:
                    rdf = rdf.Filter(
                        f"({selection_conf['mu_2_idiso'][1]})",
                        "cut on mu_2_idiso",
                    )
            elif config["channel"] == "met":
                if "mu_2_idiso" in selection_conf:
                    rdf = rdf.Filter(
                        f"({selection_conf['ele_2_idiso'][1]})",
                        "cut on ele_2_idiso",
                    )
            rdf = weights.apply_lep_fake_factors(rdf=rdf, channel=config["channel"])
        else:
            if "tau_id_vs_jet" in selection_conf:
                rdf = rdf.Filter(
                    f"({selection_conf['tau_id_vs_jet'][0]})",
                    "cut on tau_id_vs_jet",
                )
            if "mu_2_idiso" in selection_conf:
                rdf = rdf.Filter(
                    f"({selection_conf['mu_2_idiso'][0]})",
                    "cut on mu_2_idiso",
                )
            if "ele_2_idiso" in selection_conf:
                rdf = rdf.Filter(
                    f"({selection_conf['ele_2_idiso'][0]})",
                    "cut on ele_2_idiso",
                )
        # splitting data frame based on the tau origin (genuine, jet fake, lepton fake)
        for tau_gen_mode in config["processes"][process]["tau_gen_modes"]:
            tmp_rdf = rdf
            if tau_gen_mode != "all":
                tmp_rdf = filters.tau_origin_split(
                    rdf=tmp_rdf, channel=config["channel"], tau_gen_mode=tau_gen_mode
                )

            # redirecting C++ stdout for Report() to python stdout
            out = StringIO()
            with pipes(stdout=out, stderr=STDOUT):
                tmp_rdf.Report().Print()
            log.info(out.getvalue())
            log.info("-" * 50)

            tmp_file_name = func.get_output_name(
                path=output_path, process=process, tau_gen_mode=tau_gen_mode, idx=idx
            )
            # check for empty data frame -> only save/calculate if event number is not zero
            if tmp_rdf.Count().GetValue() != 0:
                log.info(f"The current data frame will be saved to {tmp_file_name}")
                cols = tmp_rdf.GetColumnNames()
                cols_with_friends = [str(x).replace("ntuple.", "") for x in cols]
                missing_cols = [x for x in outputs if x not in cols_with_friends]
                if len(missing_cols) != 0:
                    raise ValueError(f"Missing columns: {missing_cols}")
                tmp_rdf.Snapshot(config["tree"], tmp_file_name, outputs)
                log.info("-" * 50)
                process_file_dict[tau_gen_mode].append(tmp_file_name)
            else:
                log.info("No events left after filters. Data frame will not be saved.")
                log.info("-" * 50)

    # combining all files of a process and tau origin
    for tau_gen_mode in config["processes"][process]["tau_gen_modes"]:
        out_file_name = func.get_output_name(
            path=output_path, process=process, tau_gen_mode=tau_gen_mode
        )
        # combining sample files to a single process file, if there are any
        if len(process_file_dict[tau_gen_mode]) != 0:
            sum_rdf = ROOT.RDataFrame(config["tree"], process_file_dict[tau_gen_mode])
            log.info(
                f"The processed files for the {process} process are concatenated. The data frame will be saved to {out_file_name}"
            )
            sum_rdf.Snapshot(config["tree"], out_file_name, outputs)
            log.info("-" * 50)
        else:
            log.info(
                f"No processed files for the {process} process. An empty data frame will be saved to {out_file_name}"
            )
            # create an empty root file and save it
            f = ROOT.TFile(out_file_name, "RECREATE")
            t = ROOT.TTree(config["tree"], config["tree"])
            t.Write()
            f.Close()
            log.info("-" * 50)

        # delete not needed temporary sample files after combination
        for rf in process_file_dict[tau_gen_mode]:
            os.remove(rf)
        log.info("-" * 50)


if __name__ == "__main__":
    args = parser.parse_args()

    # loading of the chosen config file
    with open(args.config_file, "r") as file:
        config = yaml.load(file, yaml.FullLoader)

    # loading general dataset info file for xsec and event number
    with open("datasets/datasets.yaml", "r") as file:
        datasets = yaml.load(file, yaml.FullLoader)

    # define output path for the preselected samples
    output_path = os.path.join(
        config["output_path"],
        "preselection",
        config["era"],
        config["channel"],
        config["event_split"],
    )
    func.check_path(path=output_path)

    func.setup_logger(
        log_file=output_path + "/preselection.log",
        log_name="preselection",
        subcategories=config["processes"],
    )

    # going through all wanted processes and run the preselection function with a pool of 8 workers
    args_list = [(process, config) for process in config["processes"]]

    with multiprocessing.Pool(
        processes=min(len(config["processes"]), int(args.nthreads))
    ) as pool:
        pool.map(run_preselection, args_list)

    # dumping config to output directory for documentation
    with open(output_path + "/config.yaml", "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)
