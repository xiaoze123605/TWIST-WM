import os, sys
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.dagger_actor import DAggerActor, get_activation
import argparse

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path, checkpoint
    
def play(args):
    load_run = "../../logs/{}/{}".format(args.proj_name, args.exptid)
    checkpoint = args.checkpoint
    history_len = 10
    
    if args.robot == "g1_stu":
        num_actions = 23
        n_proprio = 3 + 2 + 3*num_actions
        # n_mimic_obs = 3 + 2 + 3 + 3*9
        # n_mimic_obs = 1 + 3*9
        n_mimic_obs = 7 + 23
        
        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (history_len + 1)
    else:
        raise ValueError(f"Robot {args.robot} not supported!")

    device = torch.device('cpu')
    policy = DAggerActor(num_observations=num_observations,
                           num_actions=num_actions,
                           actor_hidden_dims=[1024, 1024, 1024],
                           history_latent_dim=128,
                           num_hist=history_len, activation="elu",
                           ).to(device)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)
    load_run = os.path.dirname(load_path)
    print(f"Loading model from: {load_path}")
    ac_state_dict = torch.load(load_path, map_location=device)
    policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
    
    policy = policy.to(device)#.cpu()
    if not os.path.exists(os.path.join(load_run, "traced")):
        os.mkdir(os.path.join(load_run, "traced"))

    # Save the traced actor
    policy.eval()
    with torch.no_grad(): 
        num_envs = 2
        
        obs_input = torch.ones(num_envs, num_observations, device=device)
        print("obs_input shape: ", obs_input.shape)
        
        traced_policy = torch.jit.trace(policy, obs_input)
        
        save_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-jit.pt")
        traced_policy.save(save_path)
        print("Saved traced_actor at ", os.path.abspath(save_path))
        print("Robot: ", args.robot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_name', type=str)
    parser.add_argument('--exptid', type=str)
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--robot', type=str, default="g1") # options: gr1, h1, g1

    args = parser.parse_args()
    play(args)
