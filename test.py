import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, Lambda, Reshape
from tensorflow.keras.models import Model
import scipy.signal as signal

from make_env import make_env
from MHT import MultiHeadAttention

#tf.compat.v1.disable_eager_execution()


env = make_env('simple_spread')




o = env.reset()



# env.reset() - resets the environment and outputs the inital observation of all agents.  

#env.render - renders the environment, mode = 'human' and mode = 'rgb_array' gives the array of pixels.

#obs_n, reward_n, done_n, info_n = env.step(actions) - input as actions of all agents. 


# action dim  = [5, 1, 10], example of agent; [0.4, 0.2, 0.1, 0.1, 0.2, 0, 0, 0, 0, 0]

#observation dim: shape - (30,), reshape it into (1, 30)

num_agents = 5

def mlp(scope):
    with tf.compat.v1.variable_scope(scope):

        obs_in = Input(shape=(1, 30))

        h = Dense(512, activation='relu',kernel_initializer='random_normal')(obs_in)
        h = Dense(128, activation='relu',kernel_initializer='random_normal')(h)

        h = Reshape((1,128))(h)

        model = Model(inputs = obs_in,outputs = h)

        return model


def MultiHeadsAttModel(scope):

    with tf.compat.v1.variable_scope(scope):

        y = Input(shape = (5, 128))

        t = MultiHeadAttention(d_model=128, num_heads=1)
        out, attn = t(y, k=y, q=y, mask=None)

        model = Model(inputs = y, outputs = [out, attn])
        return model


def obs_agent(h, id):

    #dim of h: (1, 5, 128) to (5, 1, 128)
    h = tf.transpose(h, perm = [1, 0, 2])

    j = id - 1

    agent = h[j]
    remaining = tf.concat([h[:j], h[j+1:]], 0)
    agent = Reshape((agent.shape[0], agent.shape[1]))(agent)
    h_ = tf.concat([agent, remaining], 0)
    h_ = tf.transpose(h_, perm = [1, 0, 2])

    return(h_)


def Multi(i, scope):
    with tf.compat.v1.variable_scope(scope):

        v = Input(shape = (5, 128)) 
        q = Input(shape = (5, 128))

        v1 = Dense(16*8, activation = "relu",kernel_initializer='random_normal')(v)
        #v = Lambda(lambda x: K.permute_dimensions(x, (1, 0, 2)))(v)
        v2 = tf.transpose(v1, perm = [1, 0, 2])
        v3 = Reshape((1, 128))(v2[i])
        #att = Lambda(lambda x: K.batch_dot(x[0],x[1], axes = [2, 2]) / np.sqrt(16))([v3,q])
        att = K.batch_dot(v3, q, axes = [2, 2]) / np.sqrt(16)
        att2 = tf.nn.softmax(att)

        out = Lambda(lambda x: K.batch_dot(x[0], x[1],axes=[2,1]))([att2, v])
        out1 = Dense(128, activation = "relu",kernel_initializer='random_normal')(out)

        model = Model(inputs = [v,q], outputs = [out1])
        return model


def output(scope):
    with tf.compat.v1.variable_scope(scope):

        Out = Input(shape = (1, 128))

        out_ = Dense(64)(Out)

        critic = Dense(1)(out_)
        policy_layer = Dense(5, activation = None)(out_)
        actor = tf.nn.softmax(policy_layer)

        model = Model(inputs = Out, outputs = [critic, actor])

        return(model)


def group_to_single(arr):

    obs = np.transpose(arr, axes = [1, 0])

    at = []
    for o in range(5):
        g = []
        for l in range(3):
            g.append(np.array(obs[0][l]))
        at.append(g)
    
    single = np.array(at)

    return(single)
def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def good_discount(x, gamma):
    return discount(x,gamma)

def train(epi, gamma, b_value, i):
    #episode_buffer.append([obs, act, reward_n, obs_n, val])
    epi = np.array(epi)
    observations = epi[:,0]
    actions = epi[:,1]
    rewards = epi[:,2]
    values = epi[:,4]

    #obs = np.stack(observations)

    obs = group_to_single(observations)
    observations = obs[i]
    values = group_to_single(actions)[i]
    values = np.reshape(values, (3, 1))
    actions = group_to_single(actions)
    action = actions[i]
    reward = group_to_single(rewards)
    rewards = reward[i]


    rewards_plus = np.asarray(rewards.tolist() + [b_value])
    discounted_rewards = discount(rewards_plus,gamma)[:-1]
    value_plus = np.asarray(values.tolist() + [b_value])
    advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
    advantages = good_discount(advantages,gamma)


    target_v = np.stack(discounted_rewards)
    train_value = 1

    value, policy = network_values(observations)

    
    responsible_outputs = tf.reduce_sum(policy[i] * action, [1])

    value_loss = 0.5 * tf.reduce_sum(train_value*tf.square(target_v - tf.reshape(value, shape=[-1])))
    policy_loss = - tf.reduce_sum(tf.math.log(tf.clip_by_value(responsible_outputs,1e-15,1.0)) * advantages)

    loss = 0.5 * value_loss + policy_loss


    return(loss)



def network_values(obs):
    hi = []
    act_ = []
    val = []
        
    for i in range(num_agents):

        scope = 'name' + str(i+1)

        cn = mlp(scope)
        if obs[i].shape != (3, 30):
            ob = np.reshape(obs[i], (1, 30))
        #hi.append(cn(np.array([obs[i]])))    #(5, 1, 1, 128)
        else:
            ob = obs[i]
        hi.append(cn(ob))

    h = tf.stack(hi)

    h = tf.transpose(h, perm = [1, 0, 2, 3])
    h = Reshape((5, 128))(h)    #(batch, 5, 128)
    t = []

    for ag in range(num_agents):

        scope = 'name' + str(i+1)

        mul = Multi(ag, scope)
        h_hat = mul([h, h])
        ou = output(scope)
        h_hat1 = tf.transpose(h_hat, perm = [1, 0, 2])
        critic, actor = ou(h_hat1)

        critic = tf.transpose(critic, perm = [1, 0, 2])
        actor = tf.transpose(actor, perm = [1, 0, 2])
        act_.append(actor)

        val.append(list(np.array(critic[0])))

    return(val, act_)









for _ in range(1):

    h_ = [[] for _ in range(num_agents)] #observation embedding after representation. 

    z = np.zeros((5))

    j = 0
    
    obs = env.reset()
    episode_buffer = []
    while j < 10:
        val, policy = network_values(obs)
        act_ = []
        act2 = []
        for i in range(num_agents):

            act = np.array(policy[i])
            act = np.random.choice(5, p=act.ravel())
            t = np.zeros((5))
            t[act] = 1
            act2.append(t)
            a = np.concatenate([t, z], axis = None)
            act_.append(a.tolist())
        print(act_)







        obs_n, reward_n, done_n, info_n = env.step(act_)
        #env.render()
    


        

        episode_buffer.append([obs, act2, reward_n, obs_n, val])

        #obs = obs_n
        #env.render()

        j += 1


#obs = env.reset()
#print(np.array(obs).shape)


#print(policy)

'''
act = np.array(actor[0])
        act = np.random.choice(5, p=act.ravel())
        t = np.zeros((5))
        t[act] = 1
        a = np.concatenate([t, z], axis = None)
        act_.append(a.tolist())
'''