#bash script to change yaml files to new production

NTUPLE_PATH="root:\/\/cmsdcache-kit-disk.gridka.de\/\/store\/user\/rschmieder\/CROWN\/ntuples\/28_04_25_nosysts\/CROWNRun"
JETFAKES_LTT="jetfakes_wpVSjet_Medium_27_02_25_MediumvsJetsvsL"
JETFAKES_LLT="jetfakes_wpVSjet_Medium_27_02_25_MediumvsJetsvsL"
EVENT_SPLIT="odd"
DATE="28_04_25_nosysts_29_04_25_Medium"
ERAS="2016preVFP 2016postVFP 2017 2018"
CHANNELS_LLT="met emt mmt"
CHANNELS_LTT="mtt ett"

for ERA in $ERAS
do 
    for CHANNEL in $CHANNELS_LLT
    do
        sed "s/NTUPLE_TAG/$NTUPLE_PATH/g" configs/preselection_${CHANNEL}_${ERA}_base_medium.yaml > configs/preselection_${CHANNEL}_${ERA}_1.yaml
        sed "s/JETFAKES_TAG/$JETFAKES_LLT/g" configs/preselection_${CHANNEL}_${ERA}_1.yaml > configs/preselection_${CHANNEL}_${ERA}_2.yaml
        sed "s/EVENT_SPLIT_TAG/$EVENT_SPLIT/g" configs/preselection_${CHANNEL}_${ERA}_2.yaml > configs/preselection_${CHANNEL}_${ERA}_3.yaml
        sed "s/DATE_TAG/$DATE/g" configs/preselection_${CHANNEL}_${ERA}_3.yaml > configs/preselection_${CHANNEL}_${ERA}.yaml
        rm "configs/preselection_${CHANNEL}_${ERA}_1.yaml" "configs/preselection_${CHANNEL}_${ERA}_2.yaml" "configs/preselection_${CHANNEL}_${ERA}_3.yaml"
    done
    for CHANNEL in $CHANNELS_LTT
    do
        sed "s/NTUPLE_TAG/$NTUPLE_PATH/g" configs/preselection_${CHANNEL}_${ERA}_base_medium.yaml > configs/preselection_${CHANNEL}_${ERA}_1.yaml
        sed "s/JETFAKES_TAG/$JETFAKES_LTT/g" configs/preselection_${CHANNEL}_${ERA}_1.yaml > configs/preselection_${CHANNEL}_${ERA}_2.yaml
        sed "s/EVENT_SPLIT_TAG/$EVENT_SPLIT/g" configs/preselection_${CHANNEL}_${ERA}_2.yaml > configs/preselection_${CHANNEL}_${ERA}_3.yaml
        sed "s/DATE_TAG/$DATE/g" configs/preselection_${CHANNEL}_${ERA}_3.yaml > configs/preselection_${CHANNEL}_${ERA}.yaml
        rm "configs/preselection_${CHANNEL}_${ERA}_1.yaml" "configs/preselection_${CHANNEL}_${ERA}_2.yaml" "configs/preselection_${CHANNEL}_${ERA}_3.yaml"
    done 
done