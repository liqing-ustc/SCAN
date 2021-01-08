# export OUTPUT=outputs/cifar-10
# python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml >$OUTPUT/pretext.log
# python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml >$OUTPUT/scan.log
# python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml >$OUTPUT/selflabel.log

export DATASET=hint
export OUTPUT=outputs/$DATASET
mkdir -p $OUTPUT/pretext
mkdir -p $OUTPUT/scan
mkdir -p $OUTPUT/selflabel
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_$DATASET.yml >$OUTPUT/pretext/train.log
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_$DATASET.yml >$OUTPUT/scan/train.log
python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_$DATASET.yml >$OUTPUT/selflabel/train.log
# export MODEL=$OUTPUT/selflabel/model.pth.tar_78.2
# python eval.py --config_exp configs/selflabel/selflabel_hint.yml --model $MODEL  --save_match
# python eval.py --config_exp configs/selflabel/selflabel_hint.yml --model ${MODEL}_match --visualize_prototypes
