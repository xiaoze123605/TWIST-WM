# bash to_jit.sh 0927_twist_rlbcstu

cd legged_gym/legged_gym/scripts


exptid=${1}

proj_name="g1_stu_rl"

# Run the training script
python save_jit_stu_rlbc.py --robot "g1" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --checkpoint -1 \