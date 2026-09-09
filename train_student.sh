# Usage:
# bash train_student.sh robot_name student_id teacher_id device
# Example:


# bash train_student.sh 0927_twist_rlbcstu 0927_twist_teacher cuda:0


exptid=${1}

teacher_exptid=${2}

device=${3}

task_name="g1_stu_rl"

proj_name="g1_stu_rl"

cd legged_gym/legged_gym/scripts


# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --teacher_exptid "${teacher_exptid}" \
                --device "${device}" \
                --resume \
                --resumeid "${exptid}" \
                --teacher_checkpoint 22500
                # --debug