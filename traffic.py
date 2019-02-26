import tensorflow as tf 
import numpy as np 

# A=[NSG, EWG, NSLG, EWLG]

# global constants
grid_rows= 10
grid_cols = 10
action_size= 4

# creating our environment
states = np.zeros((4, grid_rows, grid_cols))
actions = np.eye(action_size)
# [np.arange(action_size)]

# hyperparameters

state_size = [grid_rows,grid_cols,4]
total_episodes =  7500
max_steps = 450
batch_size = 32
memory_size= 3000
learning_rate=0.0001

explore_start= 1.0
explore_end = 0.01
decay_rate = 5e-5

# assumed
gamma = 0.9 
training = False
pretrain_length = batch_size
target_network_update = 150
agent_network_update = 1500

agent_size = 2

class Simulation:
		def __init__():

		def get_state(self): # move to next state and get matrices

		def take_action(self, action): # change light and get reward
				self.reward = 

		# def is_finished(self):


# initialize simualtion 
simulation = Simulation()


class DQNetwork:
	def __init__(self, state_size, action_size, learning_rate,name='DQNetwork'):
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = learning_rate

		with tf.variable_scope(name):
			
			self.inputs = tf.placeholder(tf.float32,[None, *state_size] ,name="inputs")
			self.actions = tf.placeholder(tf.float32, [None, action_size], name="actions")
			self.target_Q = tf.placeholder(tf.float32,[None],name="target")
			"""
			first convnet: 
			CNN BN Relu CNN BN
			
			"""
			self.conv = tf.layers.conv2d(inputs = self.inputs,
																				 filters = 32,
																				 kernel_size = [3,3],
																				 padding = "SAME",
																				 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																				 name = "conv1")

			self.conv1_1 = tf.layers.conv2d(inputs = self.conv,
																					 filters = 32,
																					 kernel_size = [3,3],
																					 padding = "SAME",
																					 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																					 name = "conv1")
						
						self.conv1_1_batchnorm = tf.layers.batch_normalization(self.conv1_1,
																														training = True,
																														epsilon = 1e-5,
																														name = 'batch_norm1')
						
						self.conv1_1_out = tf.nn.relu(self.conv1_1_batchnorm, name="conv1_out")

			self.conv1_2 = tf.layers.conv2d(inputs = self.conv1_1_out,
																					 filters = 32,
																					 kernel_size = [3,3],
																					 padding = "SAME",
																					 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																					 name = "conv1")
						
						self.conv1_2_batchnorm = tf.layers.batch_normalization(self.conv1_2,
																														training = True,
																														epsilon = 1e-5,
																														name = 'batch_norm1')
							
						self.input2 = tf.math.add(self.conv,self.conv1_2_batchnorm,name='add1')

						"""
			Second convnet: 
			CNN BN Relu CNN BN
			
			"""
			self.relu1 = tf.nn.relu(self.input2, name="relu1") 

			self.conv2_1 = tf.layers.conv2d(inputs = self.relu1,
																				 filters = 32,
																			 kernel_size = [3,3],
																				 padding = "SAME",
																				 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																				 name = "conv1")
						
						self.conv2_1_batchnorm = tf.layers.batch_normalization(self.conv2_1,
																														training = True,
																														epsilon = 1e-5,
																														name = 'batch_norm1')
						
						self.conv2_1_out = tf.nn.relu(self.conv2_1_batchnorm, name="conv1_out")
		
			self.conv2_2 = tf.layers.conv2d(inputs = self.conv2_1_out,
																					 filters = 32,
																					 kernel_size = [3,3],
																					 padding = "SAME",
																					 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																					 name = "conv1")
						
						self.conv2_2_batchnorm = tf.layers.batch_normalization(self.conv2_2,
																														training = True,
																														epsilon = 1e-5,
																														name = 'batch_norm1')
							
						self.input3 = tf.math.add(self.relu1,self.conv2_2_batchnorm,name='add2')

						"""
						Final output

						Relu FC 
						"""
						self.relu2 = tf.nn.relu(self.input3, name="relu2")

						self.flatten = tf.layers.flatten(self.relu2)

						self.fc = tf.layers.dense(inputs = self.relu2,
																		units = 128, # CHECK
																		activation = tf.nn.relu,
																			kernel_initializer=tf.contrib.layers.xavier_initializer(),
																			name="fc1") 

						self.output = tf.layers.dense(inputs = self.fc, 
																				kernel_initializer=tf.contrib.layers.xavier_initializer(),
																					units = action_size, 
																					activation=None)

						self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)

						self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
						
						self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

 
class ReplayMemory():
	def __init__(self, max_size):
				self.buffer = deque(maxlen = max_size)
		
		def add_experience(self, experience):
				self.buffer.append(experience)
		
		def sample(self, batch_size):
				buffer_size = len(self.buffer)
				index = np.random.choice(np.arange(buffer_size),
																size = batch_size,
																replace = False)
				
				return [self.buffer[i] for i in index]

# instantiate memory
memory = []

for j in range(agent_size):
		memory_temp = ReplayMemory(max_size=memory_size)

		for i in range(pretrain_length):
				# If it's the first step
				if i == 0:
						# First we need a state shape(state) = grid_rows*grid_cols*4
						# to_do
						state = simulation.get_state()
		
				# Random action
				action = random.choice(actions)
		
				# Get the rewards
				# to_do
				reward = simulation.take_action(action)
		
				# Get the next state
				next_state = simulation.get_state()        
				# Add experience to memory
				memory_temp.add_experience((state, action, reward, next_state))
		
				# Our state is now the next_state
				state = next_state

				memory.append(memory_temp)


# training the agents
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
		
		tradeoff = np.random.rand()

		# Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
		explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
		
		if (explore_probability > tradeoff):
				# Make a random action (exploration)
				action = random.choice(actions)
				
		else:
				Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
				choice = np.argmax(Qs)
				action = actions[int(choice)]
								
		return action, explore_probability


with tf.Session() as sess:
		# Initialize the variables
		sess.run(tf.global_variables_initializer())
		decay_step = 0

		# to do
		simulation.init() 
		saver = tf.train.Saver()

	for episode in range(agent_network_update): # 1500

		for j in range(agent_size):
			saver.restore(sess,"./models/model_"+i+".ckpt")
				# Set step to 0
				step = 0

				 # Initialize the rewards of the episode
				episode_rewards = []

				# Make a new episode and observe the first state
				# simulation.new_episode()
				state = simulation.get_state()

				while step < max_steps: #450
						step += 1

						# Increase decay_step
						decay_step +=1

						action = np.zeros((agent_size, action_size))

						for i in range(agent_size):
								saver.restore(sess,"./models/model_"+i+".ckpt")
						# Predict the action to take and take it
								action[i], explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions)

						saver.restore(sess, "./models/model_"+j+".ckpt")
						# Do the action
						reward = simulation.take_action(action) # integrate all into one

						# Add the reward to total reward
						episode_rewards.append(reward)

							# Get the next state
							next_state = simulation.get_state()

							# Add experience to memory
							memory[i].add_experience((state, action, reward, next_state))
			
							# st+1 is now our current state
							state = next_state

						if(step % target_network_update == 0):
								### LEARNING PART 
								# Obtain random mini-batch from memory
								batch = memory[i].sample(batch_size)
								states_mb = np.array([each[0] for each in batch], ndmin=3) 
								actions_mb = np.array([each[1] for each in batch])
								rewards_mb = np.array([each[2] for each in batch]) 
								next_states_mb = np.array([each[3] for each in batch], ndmin=3)

								target_Qs_batch = []

								# Get Q values for next_state 
								Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})

								# Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
								for i in range(0, len(batch)):
										target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
										target_Qs_batch.append(target)	            

								targets_mb = np.array([each for each in target_Qs_batch])

								loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
																feed_dict={DQNetwork.inputs_: states_mb,
																					 DQNetwork.target_Q: targets_mb,
																					 DQNetwork.actions_: actions_mb})

				total_reward = np.sum(episode_rewards)
				print('Episode: {}'.format(episode),    
									'Total reward: {}'.format(total_reward),
									'Training loss: {:.4f}'.format(loss),
									'Explore P: {:.4f}'.format(explore_probability))
				

		save_path = saver.save(sess, "./models/model_"+j+".ckpt")


