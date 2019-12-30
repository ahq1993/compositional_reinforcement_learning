# COMPOSING TASK-AGNOSTIC POLICIES WITH DEEP REINFORCEMENT LEARNING


Requirements:
a- Rllab
b- Tensorflow
c- mujoco


# To train composite model from scratch, run:

To simulate "ant-cross-maze", run:

python mujoco_am_sac.py --log_dir="/path-to-crl-code-folder/composition_sac_code/ant-maze" --domain="ant-cross-maze"

To simulate "ant-random-goal", run:
python mujoco_am_sac.py --log_dir="/path-to-crl-code-folder/composition_sac_code/ant-rgoal" --domain="ant-random-goal"

To simulate "cheetah-hurdle", run:
python mujoco_am_sac.py --log_dir="/path-to-crl-code-folder/composition_sac_code/cheetah-hurdle" --domain="cheetah-hurdle"

To simulate "pusher", run:
 python mujoco_am_sac.py --log_dir="/path-to-crl-code-folder/composition_sac_code/pusher" --domain="pusher"
 
 
 
 @inproceedings{
qureshi2020composing,
title={Composing Task-Agnostic Policies with Deep Reinforcement Learning},
author={Ahmed H. Qureshi and Jacob J. Johnson and Yuzhe Qin and Taylor Henderson and Byron Boots and Michael C. Yip},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=H1ezFREtwH}
}
