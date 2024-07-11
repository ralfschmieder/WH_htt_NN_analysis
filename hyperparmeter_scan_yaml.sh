#bash script to change yaml files to new production

L_TAG="128 256"
LR_TAG="0.00001 0.0000075"
BATCH_TAG="256 512"

for LAYER in $L_TAG
do 
    for LR in $LR_TAG
    do
        for BATCH in $BATCH_TAG
            do
                DATE="10_07_24_${LAYER}_${LR}_${BATCH}"
                sed "s/L_TAG/$LAYER/g" configs/neural_net_llt_base_hyper.yaml > configs/neural_net_llt_hyperscan_1.yaml
                sed "s/LR_TAG/$LR/g" configs/neural_net_llt_hyperscan_1.yaml > configs/neural_net_llt_hyperscan_2.yaml
                sed "s/BATCH_TAG/$BATCH/g" configs/neural_net_llt_hyperscan_2.yaml > configs/neural_net_llt_hyperscan_3.yaml
                sed "s/DATE_TAG/$DATE/g" configs/neural_net_llt_hyperscan_3.yaml > configs/neural_net_llt_hyperscan_${LAYER}_${LR}_${BATCH}.yaml
                rm "configs/neural_net_llt_hyperscan_1.yaml" "configs/neural_net_llt_hyperscan_2.yaml" "configs/neural_net_llt_hyperscan_3.yaml"
        done
    done
done

DATE="03_07_24_2"
EVENT_SPLIT="even"
CHANNEL="mmt"
for LAYER in $L_TAG
do 
    for LR in $LR_TAG
    do
        for BATCH in $BATCH_TAG
        do
            python torchscript.py -i /ceph/rschmieder/WH_analysis/NN_analysis/${DATE} -y all_eras -k ${CHANNEL} --event-split even -e 1000 -o results -c configs/neural_net_llt_hyperscan_${LAYER}_${LR}_${BATCH}.yaml
        done
    done
done