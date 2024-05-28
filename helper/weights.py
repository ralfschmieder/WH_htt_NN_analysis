import array
from typing import Dict, Any, Union, List


def gen_weight(rdf: Any, sample_info: Dict[str, str]) -> Any:
    """
    Function to apply the generator weight and cross section.

    Args:
        rdf: root DataFrame object
        sample_info: Dictionary with information about a sample

    Return:
        root DataFrame object with the applied weight
    """
    number_generated_events_weight = 1.0 / float(sample_info["nevents"])
    cross_section_per_event_weight = float(sample_info["xsec"])
    negative_events_fraction = float(sample_info["generator_weight"])
    rdf = rdf.Define(
        "numberGeneratedEventsWeight", f"(float){number_generated_events_weight}"
    )
    rdf = rdf.Define(
        "crossSectionPerEventWeight", f"(float){cross_section_per_event_weight}"
    )
    rdf = rdf.Define("negativeEventsFraction", f"(float){negative_events_fraction}")

    return rdf.Redefine(
        "weight",
        "weight * numberGeneratedEventsWeight * crossSectionPerEventWeight * (( 1.0 / negativeEventsFraction) * ( ((genWeight<0) * -1) + ((genWeight>=0) * 1)))",
    )


def stitching_gen_weight(
    rdf: Any, era: str, process: str, sample_info: Dict[str, str]
) -> Any:
    """
    Function to apply the generator weight and cross section. This is specific for samples where stitching is used, like "DYjets" or "Wjets"

    Args:
        rdf: root DataFrame object
        era: Stitching weights depend on the data-taking period
        process: Stitching weights depend on the process e.g. "DYjets" or "Wjets"
        sample_info: Dictionary with information about a sample

    Return:
        root DataFrame object with the applied weight
    """
    number_generated_events_weight = 1.0 / float(sample_info["nevents"])
    cross_section_per_event_weight = float(sample_info["xsec"])
    negative_events_fraction = float(sample_info["generator_weight"])
    rdf = rdf.Define(
        "numberGeneratedEventsWeight", f"(float){number_generated_events_weight}"
    )
    rdf = rdf.Define(
        "crossSectionPerEventWeight", f"(float){cross_section_per_event_weight}"
    )
    rdf = rdf.Define("negativeEventsFraction", f"(float){negative_events_fraction}")

    if era == "2018":
        if process == "Wjets":
            rdf = rdf.Redefine(
                "weight",
                "weight * (0.0007590865*( ((npartons<=0) || (npartons>=5))*1.0 + (npartons==1)*0.2191273680 + (npartons==2)*0.1335837379 + (npartons==3)*0.0636217909 + (npartons==4)*0.0823135765 ))",
            )
        elif process == "DYjets":
            rdf = rdf.Redefine(
                "weight",
                "weight * ( (genbosonmass>=50.0)*0.0000631493*( ((npartons<=0) || (npartons>=5))*1.0 + (npartons==1)*0.2056921342 + (npartons==2)*0.1664121306 + (npartons==3)*0.0891121485 + (npartons==4)*0.0843396952 ) + (genbosonmass<50.0) * numberGeneratedEventsWeight * crossSectionPerEventWeight * (( 1.0 / negativeEventsFraction) * ( ((genWeight<0) * -1) + ((genWeight>=0) * 1))))",
            )
        else:
            raise ValueError(f"No stitching weights for this process: {process}")
    else:
        raise ValueError(f"No stitching weights defined for this era: {era}")

    return rdf


def lumi_weight(rdf: Any, era: str) -> Any:
    """
    Function to apply the luminosity depending on the era.

    Args:
        rdf: root DataFrame object
        era: Luminosity is depended on the data-taking period

    Return:
        root DataFrame object with the applied weight
    """
    if era == "2016preVFP":
        rdf = rdf.Redefine("weight", "weight * 19.52 * 1000.")
    elif era == "2016postVFP":
        rdf = rdf.Redefine("weight", "weight * 16.81 * 1000.")
    elif era == "2017":
        rdf = rdf.Redefine("weight", "weight * 41.48 * 1000.")
    elif era == "2018":
        rdf = rdf.Redefine("weight", "weight * 59.83 * 1000.")
    else:
        raise ValueError(f"Weight calc: lumi: Era is not defined: {era}")

    return rdf


def apply_tau_id_vsJet_weight(
    rdf: Any,
    channel: str,
    wp: str,
) -> Any:
    """
    This function applies tau id vs jet scale factors based on the working point which are chosen in the cuts.

    Args:
        rdf: root DataFrame object
        channel: Analysis channel of the tau analysis e.g. "et", "mt" or "tt"
        wp: A string defining the working point

    Return:
        root DataFrame with applied tau id vs jet scale factors
    """
    if channel in ["mmt", "emt", "met"]:
        rdf = rdf.Redefine(
            "weight",
            f"weight * ((gen_match_3==5) * ((id_tau_vsJet_Tight_3>0.5)*id_wgt_tau_vsJet_Tight_3 + (id_tau_vsJet_Tight_3<0.5)) + (gen_match_3!=5))",
        )

    elif channel in ["mtt", "ett"]:
        rdf = rdf.Redefine(
            "weight",
            f"weight * ((gen_match_1==5) * ((id_tau_vsJet_{wp}_1>0.5)*id_wgt_tau_vsJet_{wp}_1 + (id_tau_vsJet_{wp}_1<0.5)) + (gen_match_1!=5))",
        )
        rdf = rdf.Redefine(
            "weight",
            f"weight * ((gen_match_2==5) * ((id_tau_vsJet_{wp}_2>0.5)*id_wgt_tau_vsJet_{wp}_2 + (id_tau_vsJet_{wp}_2<0.5)) + (gen_match_2!=5))",
        )

    else:
        raise ValueError(
            f"Weight calc: tau id vs jet: Such a channel is not defined: {channel}"
        )

    return rdf


def apply_tau_fake_factors(rdf: Any, channel: str, wp: str = None) -> Any:
    """
    This function applies fake factors. The have to be already calculated and are normally inside of friend tree files.

    Args:
        rdf: root DataFrame object
        channel: Analysis channel of the tau analysis e.g. "et", "mt" or "tt"
        wp: A string defining the working point, only relevant for "tt" channel

    Return:
        root DataFrame with applied fake factors
    """
    if channel in ["mmt", "emt", "met"]:
        rdf = rdf.Redefine(
            "weight",
            f"weight * (tau_fakerate_Era*(id_tau_vsJet_Tight_3<0.5&&id_tau_vsJet_VVVLoose_3>0.5))",
        )
    elif channel in ["mtt", "ett"]:
        rdf = rdf.Redefine(
            "weight",
            f"weight * tau_fakerate_Era*(((q_1*q_2>0) && id_tau_vsJet_VTight_2<0.5 && id_tau_vsJet_VVVLoose_2>0.5 && id_tau_vsJet_Medium_3>0.5) || ((q_1*q_3>0) && id_tau_vsJet_VTight_3<0.5 && id_tau_vsJet_VVVLoose_3>0.5 && id_tau_vsJet_Medium_2>0.5))",
        )
    else:
        raise ValueError(
            f"Weight calc: fake factors: Such a channel is not defined: {channel}"
        )

    return rdf


def apply_lep_fake_factors(rdf: Any, channel: str, wp: str = None) -> Any:
    """
    This function applies fake factors. The have to be already calculated and are normally inside of friend tree files.

    Args:
        rdf: root DataFrame object
        channel: Analysis channel of the tau analysis e.g. "et", "mt" or "tt"
        wp: A string defining the working point, only relevant for "tt" channel

    Return:
        root DataFrame with applied fake factors
    """
    if channel in ["mmt", "emt"]:
        rdf = rdf.Redefine(
            "weight",
            f"weight * (lep_2_fakerate_Era*(iso_2>0.15 || muon_is_mediumid_2<0.5))",
        )
    elif channel == "met":
        rdf = rdf.Redefine(
            "weight",
            f"weight * (lep_2_fakerate_Era*(iso_2>0.15 || electron_is_nonisowp90_2<0.5))",
        )
    else:
        raise ValueError(
            f"Weight calc: fake factors: Such a channel is not defined: {channel}"
        )

    return rdf
