#bash script to combine all eras of a process for NN input
ERAS="2016preVFP 2016postVFP 2017 2018"
DATE="22_05_24"
EVENT_SPLIT="odd even"
CHANNELS="mmt emt met mtt ett"

for SPLIT in $EVENT_SPLIT
do
for CHANNEL in $CHANNELS
do
    rm -r /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/
    mkdir -p /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/
    
    hadd /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/ggZH.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016preVFP/${CHANNEL}/${SPLIT}/ggZH.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016postVFP/${CHANNEL}/${SPLIT}/ggZH.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2017/${CHANNEL}/${SPLIT}/ggZH.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2018/${CHANNEL}/${SPLIT}/ggZH.root

    hadd /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/ggZZ.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016preVFP/${CHANNEL}/${SPLIT}/ggZZ.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016postVFP/${CHANNEL}/${SPLIT}/ggZZ.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2017/${CHANNEL}/${SPLIT}/ggZZ.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2018/${CHANNEL}/${SPLIT}/ggZZ.root

    hadd /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/lep_fakes.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016preVFP/${CHANNEL}/${SPLIT}/lep_fakes.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016postVFP/${CHANNEL}/${SPLIT}/lep_fakes.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2017/${CHANNEL}/${SPLIT}/lep_fakes.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2018/${CHANNEL}/${SPLIT}/lep_fakes.root

    hadd /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/rem_ttbar.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016preVFP/${CHANNEL}/${SPLIT}/rem_ttbar.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016postVFP/${CHANNEL}/${SPLIT}/rem_ttbar.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2017/${CHANNEL}/${SPLIT}/rem_ttbar.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2018/${CHANNEL}/${SPLIT}/rem_ttbar.root

    hadd /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/tau_fakes.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016preVFP/${CHANNEL}/${SPLIT}/tau_fakes.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016postVFP/${CHANNEL}/${SPLIT}/tau_fakes.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2017/${CHANNEL}/${SPLIT}/tau_fakes.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2018/${CHANNEL}/${SPLIT}/tau_fakes.root

    hadd /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/triboson.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016preVFP/${CHANNEL}/${SPLIT}/triboson.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016postVFP/${CHANNEL}/${SPLIT}/triboson.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2017/${CHANNEL}/${SPLIT}/triboson.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2018/${CHANNEL}/${SPLIT}/triboson.root

    hadd /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/WH_htt_minus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016preVFP/${CHANNEL}/${SPLIT}/WH_htt_minus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016postVFP/${CHANNEL}/${SPLIT}/WH_htt_minus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2017/${CHANNEL}/${SPLIT}/WH_htt_minus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2018/${CHANNEL}/${SPLIT}/WH_htt_minus.root

    hadd /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/WH_htt_plus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016preVFP/${CHANNEL}/${SPLIT}/WH_htt_plus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016postVFP/${CHANNEL}/${SPLIT}/WH_htt_plus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2017/${CHANNEL}/${SPLIT}/WH_htt_plus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2018/${CHANNEL}/${SPLIT}/WH_htt_plus.root

    hadd /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/WH_hww_plus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016preVFP/${CHANNEL}/${SPLIT}/WH_hww_plus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016postVFP/${CHANNEL}/${SPLIT}/WH_hww_plus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2017/${CHANNEL}/${SPLIT}/WH_hww_plus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2018/${CHANNEL}/${SPLIT}/WH_hww_plus.root

    hadd /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/WH_hww_minus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016preVFP/${CHANNEL}/${SPLIT}/WH_hww_minus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016postVFP/${CHANNEL}/${SPLIT}/WH_hww_minus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2017/${CHANNEL}/${SPLIT}/WH_hww_minus.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2018/${CHANNEL}/${SPLIT}/WH_hww_minus.root

    hadd /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/ZH.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016preVFP/${CHANNEL}/${SPLIT}/ZH.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016postVFP/${CHANNEL}/${SPLIT}/ZH.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2017/${CHANNEL}/${SPLIT}/ZH.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2018/${CHANNEL}/${SPLIT}/ZH.root

    hadd /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/all_eras/${CHANNEL}/${SPLIT}/diboson.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016preVFP/${CHANNEL}/${SPLIT}/diboson.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2016postVFP/${CHANNEL}/${SPLIT}/diboson.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2017/${CHANNEL}/${SPLIT}/diboson.root /ceph/rschmieder/WH_analysis/NN_analysis/${DATE}/preselection/2018/${CHANNEL}/${SPLIT}/diboson.root

done
done