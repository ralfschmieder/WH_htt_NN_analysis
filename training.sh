DATE="22_05_24"
EVENT_SPLIT="odd even"
CHANNELS_LTT="mtt ett"
CHANNELS_LLT="mmt emt met"
for SPLIT in $EVENT_SPLIT
do
for CHANNEL in $CHANNELS_LTT
do
python torchscript.py -i /ceph/rschmieder/WH_analysis/NN_analysis/${DATE} -y all_eras -k ${CHANNEL} -s ${SPLIT} -e 500 -o results -c configs/neural_net_ltt.yaml
done
done
for SPLIT in $EVENT_SPLIT
do

# for CHANNEL in $CHANNELS_LLT
# do
# python torchscript.py -i /ceph/rschmieder/WH_analysis/NN_analysis/${DATE} -y all_eras -k ${CHANNEL} -s ${SPLIT} -e 500 -o results -c configs/neural_net_llt.yaml
# done
# done