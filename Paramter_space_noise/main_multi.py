import time
import os
import gym
import argparse

import pickle
import copy 

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

import gym
import pybullet
# import pybulletgym.envs
# import pybullet_envs

import IPython
from IPython import display

import matplotlib.pyplot as plt

import utils1
import TD3

from TD3 import AdaptiveParamNoiseSpec
from TD3 import ddpg_distance_metric



class NormalizeAction(gym.ActionWrapper):
    def action(self, action):
        action = (action + 1) / 2  
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_seed = seed + 100
    
    eval_env = gym.make(env_name)
    eval_env.seed(eval_seed)
    eval_env.action_space.seed(eval_seed)
    #eval_env.render()
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Walker2d-v2")    # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e5, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)
    env1=gym.make(args.env)
    
    env=NormalizeAction(env)
    env1=NormalizeAction(env1)
    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    env1.seed(args.seed)
    env1.action_space.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
        policy1=TD3.TD3(**kwargs)
    # 	elif args.policy == "OurDDPG":
    # 		policy = OurDDPG.DDPG(**kwargs)
    # 	elif args.policy == "DDPG":
    # 		policy = DDPG.DDPG(**kwargs)

    # 	if args.load_model != "":
    # 		policy_file = file_name if args.load_model == "default" else args.load_model
    # 		policy.load(f"./models/{policy_file}")

    replay_buffer = utils1.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    state1, done1 = env1.reset(), False
    episode_reward = [0,0]
    episode_timesteps = 0
    episode_num = 0
    total_rewards=[]
    
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05,desired_action_stddev=0.3, adaptation_coefficient=1.05)
    
    start_time = time.time()
    
    for t in range(int(args.max_timesteps)):
        
        noise_counter=0
        policy.perturb_actor_parameters(param_noise)
        policy1.perturb_actor_parameters(param_noise)
       
        episode_timesteps += 1
        '''network 1'''
        # Select action randomly or according to policy
        
        if not done:
            if t < args.start_timesteps:
                action = env.action_space.sample()
                
            else:
                action = (
                    policy.select_action(np.array(state),perturb=True)
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            done_bool = float(done) #if episode_timesteps < env._max_episode_steps else 0
    
            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)
    
            state = next_state
            episode_reward[0] += reward
            noise_counter+=1
            
        '''network 2'''
        if not done1:
            if t < args.start_timesteps:
                action1=env1.action_space.sample()
            else:
                
                action1 = (
                    policy1.select_action(np.array(state1),perturb=True)
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                
            next_state1, reward1, done1, _ = env1.step(action1) 
            done_bool1 = float(done1) #if episode_timesteps < env._max_episode_steps else 0
            # Store data in replay buffer
            replay_buffer.add(state1, action1, next_state1, reward1, done_bool1)
    
            state1 = next_state1
            episode_reward[1] += reward1
            noise_counter+=1
        
        

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)
            policy1.train(replay_buffer,args.batch_size)

        if done and done1: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            
            # Reset environment
            ddpgs = [policy, policy1]
            #total_rewards.append(episode_reward)
            idx = np.array(episode_reward).argmax()
            myddpg = ddpgs[idx]
            
            
            
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward[idx]:.3f}")
            
            policy=copy.deepcopy(myddpg)
            policy1=copy.deepcopy(myddpg)
            
            if replay_buffer.ptr-noise_counter>0:
                noise_a=replay_buffer.action[replay_buffer.ptr-noise_counter:replay_buffer.ptr]
                noise_s=replay_buffer.state[replay_buffer.ptr-noise_counter:replay_buffer.ptr]
            else:
                noise_a=replay_buffer.action[replay_buffer.ptr-noise_counter+1e6:1e6]+replay_buffer.action[0:replay_buffer.ptr]
                noise_s=replay_buffer.state[replay_buffer.ptr-noise_counter+1e6:1e6]+replay_buffer.action[0:replay_buffer.ptr]
            
            
            unperturbed_actions = myddpg.select_action(np.array(noise_s[0,:]))
            perturbed_actions = noise_a
            ddpg_dist = ddpg_distance_metric(perturbed_actions, unperturbed_actions)
            param_noise.adapt(ddpg_dist)
            
            total_rewards=[]
            
            state, done = env.reset(), False
            state1, done1 = env1.reset(), False
            episode_reward = [0,0]
            
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print("---------------------------------------")
            print(f"Percent complete: {100*(t+1)/args.max_timesteps:.2f}, hours since start: {((time.time() - start_time) / 3600):.2f}")
            print("---------------------------------------")
            evaluations.append(eval_policy(myddpg, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")

    print("training took {:.2f} hours for {} timestamps".format((time.time() - start_time) / 3600, episode_timesteps))
    plt.figure(figsize=(20, 10))
    plt.plot(evaluations)
    plt.xlabel(f'Every {args.eval_freq} updates')
    plt.ylabel('Average rewards')
    plt.title(file_name + ", trained {:.2f} hrs for {} timestamps".format((time.time() - start_time) / 3600, args.max_timesteps))
    plt.grid()
    plt.savefig(file_name + "_rewards_10mil" + ".png", dpi=400)
    plt.show()
    
    torch.save(myddpg,'./Td3_walker2d_500k.pt')
    torch.save(myddpg,'./Td3_walker2d_500k.pth')
    e=np.array(evaluations)
