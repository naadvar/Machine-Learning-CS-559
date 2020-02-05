from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque


GAME = 'bird'
#here we define the number of actions, i.e either 0 or 1
ACTIONS = 2
#gamma is the discount rate of rewards
GAMMA = 0.95
#here we define the number of episodes we observe before we start training
OBSERVE = 100.
#this is used to change the epsilon value if it does not change after these many steps
EXPLORE = 1000000.
#this is used to set the final value of epislon
FINAL_EPSILON = 0.0001
#this is the initital value of epsilon that we start with
INITIAL_EPSILON = 0.001
#this is the number of entries we want in the replay buffer
REPLAY_MEMORY = 25000
#this is the sampling size from the replay buffer that we use
BATCH = 32
FRAME_PER_ACTION = 1
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
def animate(x,y):
    ax1.clear()
    ax1.plot(x, y)
import csv

def write_csv(data): #this function is used to write the action, the reward, q values to a csv file for plotting
    with open('example.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)
def write_loss(data): #this function is used to write the action, the reward, q values to a csv file for plotting
    with open('loss.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)



def createNetwork():
    #we initialize the weights and biases for the convolutional nerualnetwork
    W_conv1 = tf.Variable(tf.truncated_normal([7, 7, 4, 32], stddev = 0.01))
    b_conv1 = tf.Variable(tf.constant(0.01, shape = [32]))

    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.01))
    b_conv2 = tf.Variable(tf.constant(0.01, shape = [64]))

    W_fc1 = tf.Variable(tf.truncated_normal([1600, 200], stddev = 0.01))
    b_fc1 = tf.Variable(tf.constant(0.01, shape = [200]))

    W_fc2 = tf.Variable(tf.truncated_normal([200, ACTIONS], stddev = 0.01))
    b_fc2 = tf.Variable(tf.constant(0.01, shape = [ACTIONS]))

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides = [1, 4, 4, 1], padding = "SAME") + b_conv1)
    h_pool = tf.nn.max_pool(h_conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool, W_conv2, strides = [1, 2, 2, 1], padding = "SAME") + b_conv2)
    h_conv3_flat = tf.reshape(h_conv2, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    output = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, output, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # Here we start the q learning loss function calculation and optimization
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))



    #we use adam optimizer with 1e-6 as learning rate to minize the cost
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # Start a game by calling from the pygame module
    game_state = game.GameState()

    # This is to help build the replay buffer
    replay = deque()
     # In the first iteartion we just use the first image for times
    initial = np.zeros(ACTIONS)
    initial[0] = 1
    x_t, r_0, terminal = game_state.frame_step(initial)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    epsilon = INITIAL_EPSILON
    t = 0
    step = 0

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    
    if checkpoint and checkpoint.model_checkpoint_path: #here we have saved the network while training to help give faster outputs
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path) #we check if we have any data on the saved networks and load the parameters for instant output
    else:
        print("Could not find old network value")
    
    # start training

    while step<4500: #the number of iterations are set to 4500, we can keep training this but it will take a lot of time, so we try to cut it down


        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("we take a random action to break the pattern")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1


        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # Based on the action take we update, state and the reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1_crop = x_t1_colored[35:565, 1:421]
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_crop, (80, 80)), cv2.COLOR_BGR2GRAY) #convert the image to 80x80 and to greyscae
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY) #divide it by 255 to normalize the pixle values
        x_t1 = np.reshape(x_t1, (80, 80, 1)) #reshape it into a ternsor format
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in the replay buffer and if the size increases, we pop the last one
        replay.append((s_t, a_t, r_t, s_t1, terminal))
        if len(replay) > REPLAY_MEMORY:
            replay.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch of 32 to use the concept of experience replay
            minibatch = random.sample(replay, BATCH)

            # get the batch variables based on the bellman equation
            s_j_batch = [d[0] for d in minibatch] #the current state
            a_batch = [d[1] for d in minibatch] #the current action
            r_batch = [d[2] for d in minibatch] #the current reward
            s_j1_batch = [d[3] for d in minibatch] #the next state

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]

                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient descent
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1



        print(
             "Action", action_index, "/ Reward", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        step = step+1

        write_csv([t,r_t,epsilon,action_index,np.max(readout_t)])

def play():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

play()
