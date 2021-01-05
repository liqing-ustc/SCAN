# export OUTPUT=outputs/cifar-10
# python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml >$OUTPUT/pretext.log
# python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml >$OUTPUT/scan.log
# python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml >$OUTPUT/selflabel.log

export DATASET=hint
export OUTPUT=outputs/$DATASET
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_$DATASET.yml >$OUTPUT/pretext.log
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_$DATASET.yml >$OUTPUT/scan.log
python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_$DATASET.yml >$OUTPUT/selflabel.log