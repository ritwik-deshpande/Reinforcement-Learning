import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None

    def maxQs(self, state):
        maxQ = float('-inf')
        for action in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]:
            if maxQ < self.Q[state][action]:
                maxQ = self.Q[state][action]
        return maxQ

    def updateNTable(self, state, action):
        self.N[state][action] += 1

    def get_optimal_action(self, state):
        max_val = float('-inf')
        optimal_action = 0
        for action in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]:
            N = self.N[state][action]
            if N < self.Ne and self.train:
                val = 1
            else:
                Qs = self.Q[state][action]
                val = Qs

            if max_val < val:
                max_val = val
                optimal_action = action
        return optimal_action

    def updateQTable(self, prev_state, prev_action, next_state, reward):

        maxQ = self.maxQs(next_state)

        alpha = self.C / (self.C + self.N[prev_state][prev_action])

        self.Q[prev_state][prev_action] = self.Q[prev_state][prev_action] + alpha * (reward + self.gamma * maxQ - self.Q[prev_state][prev_action])


    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        next_state = self.generate_state(environment)
        # TODO: write your function here
        reward = -0.1
        if self._train and self.s != None  and self.a != None:
            if dead:
                reward = -1
            elif points > self.points:
                reward = 1

            self.updateNTable(self.s, self.a)
            self.updateQTable(self.s, self.a, next_state, reward)

        optimal_action = self.get_optimal_action(next_state)

        if not dead:
            self.s = next_state
            self.points = points
            self.a = optimal_action
        else:
            self.reset()
            return utils.RIGHT

        return optimal_action


    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment
        snake_head_x, snake_head_y, snake_body, food_x, food_y = environment
        food_dir_x = 0
        food_dir_y = 0
        adjoining_wall_x = 0
        adjoining_wall_y = 0
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0

        if snake_head_x == 1:
            adjoining_wall_x = 1
        elif snake_head_x == utils.DISPLAY_WIDTH - 2 :
            adjoining_wall_x = 2

        if snake_head_y == 1:
            adjoining_wall_y = 1
        elif snake_head_y == utils.DISPLAY_HEIGHT - 2 :
            adjoining_wall_y = 2

        for snake_body_x, snake_body_y in snake_body:
            if (snake_head_y + 1 == snake_body_y) and (snake_body_x == snake_head_x):
                adjoining_body_bottom = 1

            if (snake_head_y - 1 == snake_body_y) and (snake_body_x == snake_head_x):
                adjoining_body_top = 1

            if (snake_head_x + 1 == snake_body_x) and (snake_body_y == snake_head_y):
                adjoining_body_right = 1

            if (snake_head_x - 1 == snake_body_x) and (snake_body_y == snake_head_y):
                adjoining_body_left = 1

        if food_y == snake_head_y :
            food_dir_y = 0
        elif food_y < snake_head_y:
            food_dir_y = 1
        else:
            food_dir_y = 2

        if food_x == snake_head_x :
            food_dir_x = 0
        elif food_x < snake_head_x:
            food_dir_x = 1
        else:
            food_dir_x = 2

        state = food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right

        return state
