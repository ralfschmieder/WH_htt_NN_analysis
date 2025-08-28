DATE="27_03_25_Medium"
EVENT_SPLIT="even"
CHANNELS_LTT="ett mtt"
CHANNELS_LLT="emt met mmt"
TAG="07_05_alldibosonindiboson"
for SPLIT in $EVENT_SPLIT
do
for CHANNEL in $CHANNELS_LTT
do
python torchscript.py -i /ceph/rschmieder/WH_analysis/NN_analysis/${DATE} -y all_eras -k ${CHANNEL} -s ${SPLIT} -e 500 -o results -c configs/neural_net_ltt.yaml -t ${TAG}
done
done
for SPLIT in $EVENT_SPLIT
do
for CHANNEL in $CHANNELS_LLT
do
python torchscript.py -i /ceph/rschmieder/WH_analysis/NN_analysis/${DATE} -y all_eras -k ${CHANNEL} -s ${SPLIT} -e 500 -o results -c configs/neural_net_llt.yaml -t ${TAG}
done
done