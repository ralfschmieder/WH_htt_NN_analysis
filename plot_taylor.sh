CHANNELS="mmt"
WPS="Medium" # loose"
DATE="27_03_25"
TAG="_10_04_standard_taylor"
RANK=30
for CH in $CHANNELS
do
for WP in $WPS
do
INPUT="workdir_${CH}_${DATE}_${WP}${TAG}/results/tca/node_0_coefficients.csv"
python plot_taylor.py --input_file ${INPUT} --channel ${CH} --rank ${RANK}
done
done