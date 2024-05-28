#bash script to combine all eras of a process for NN input
ERAS="2018"
CHANNELS="mmt"

for CHANNEL in $CHANNELS
do
for ERA in $ERAS
do
python preselection.py --config-file configs/preselection_${CHANNEL}_${ERA}.yaml
done 
done