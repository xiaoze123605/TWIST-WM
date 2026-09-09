
# bash train_teacher.sh 0927_twist_teacher cuda:0

cd legged_gym/legged_gym/scripts


exptid=$1
device=$2

task_name="g1_priv_mimic"
proj_name="g1_priv_mimic"

# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --resume \
                --resumeid "${exptid}" \
                # --debug
