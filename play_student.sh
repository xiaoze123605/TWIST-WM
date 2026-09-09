

# bash play_student.sh g1 0927_twist_rlbcstu 0927_twist_teacher

cd legged_gym/legged_gym/scripts


exptid=$1
teacher_exptid=$2
task_name="g1_stu_rl"
proj_name="g1_stu_rl"


# Run the eval script
python play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --num_envs 1 \
                --teacher_exptid "${teacher_exptid}" \
                --record_video \
