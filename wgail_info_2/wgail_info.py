import gym, roboschool
import numpy as np
import argparse
import time
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json

from models import TRPOAgent


def playGame(finetune=0):

    demo_dir = "/home/zhiyang/Desktop/intention/reacher/rl_demo/"
    param_dir = "/home/zhiyang/Desktop/intention/params/"
    pre_actions_path = ""
    #feat_dim = [7, 13, 1024]  ## (hzyjerry) how 7, 13
    feat_dim = [16, 16, 1024]
    
    img_dim = [256, 256, 3]
    aux_dim = 2             ## (hzyjerry) Reacher velocity
    encode_dim = 2          ## (hzyjerry) Two hidden states
    action_dim = 2

    np.random.seed(1024)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.98
    
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # initialize the env
    env = gym.make("RoboschoolReacherRGB_inf-v1") #TorcsEnv(throttle=True, gear_change=False)

    # define the model
    #pre_actions = np.load(pre_actions_path)["actions"]
    pre_actions = None ## (hzyjerry) no pre-actions
    agent = TRPOAgent(env, sess, feat_dim, aux_dim, encode_dim, action_dim,
                      img_dim, pre_actions)

    # Load expert (state, action) pairs
    demo = np.load(demo_dir + "demo.npz")

    # Now load the weight
    print("Now we load the weight")
    try:
        if finetune:
            agent.generator.load_weights(
                param_dir + "reacher_params_0/generator_model_37.h5")
            agent.discriminator.load_weights(
                param_dir + "reacher_params_0/discriminator_model_37.h5")
            agent.baseline.model.load_weights(
                param_dir + "reacher_params_0/baseline_model_37.h5")
            agent.posterior.load_weights(
                param_dir + "reacher_params_0/posterior_model_37.h5")
            agent.posterior_target.load_weights(
                param_dir + "reacher_params_0/posterior_target_model_37.h5")
        else:
            #agent.generator.load_weights(
            #    param_dir + "params_bc/params_2/generator_bc_model.h5")
            agent.generator.load_weights(
                param_dir + "reacher_params_0/params_2/generator_bc_model.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("Gym Experiment Start.")
    agent.learn(demo)

    print("Finish.")


if __name__ == "__main__":
    playGame()
