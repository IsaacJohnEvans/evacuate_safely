#%%
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy.spatial import distance_matrix

#%%
class Agent():
    def __init__(self, position):
        self.position = position
        self.escaped = False
        self.panic = False
    
class Map():
    def __init__(self, map_shape, num_agents, num_obstacles, num_exits, speed = 1, max_runs = 10):
        self.map_shape = map_shape
        self.map = np.zeros(self.map_shape)
        self.speed = speed
        self.run = 0
        self.max_runs = max_runs
        
        # Make obstacles
        self.obstacle_bool = np.zeros(map_shape, dtype=bool)
        num_obstacles_to_spawn = num_obstacles
        while num_obstacles_to_spawn != 0:
            self.obstacle_bool[np.unravel_index(np.random.choice(np.linspace(0, self.map.size, self.map.size, endpoint=False, dtype = int), num_obstacles_to_spawn), self.map.shape)] = 1
            num_obstacles_to_spawn = num_obstacles - np.sum(self.obstacle_bool.astype(int))
        self.obstacle_positions = np.argwhere(self.obstacle_bool == True)
        
        # Make exits
        self.exit_bool = np.zeros(map_shape, dtype= bool)
        exit_loc = np.zeros((num_exits, 2), dtype=int)
        i = 0 
        while i < num_exits:
            exit_loc[i, :] = np.array([np.random.randint(-1, 1), np.random.randint(0, map_shape[0])])[:: (-1, 1)[np.random.randint(0, 2)]]
            self.exit_bool[exit_loc[i, 0], exit_loc[i, 1]] = True
            self.exit_bool[self.obstacle_bool] = False
            i = np.sum(self.exit_bool.astype(int))
        self.exit_positions = np.argwhere(self.exit_bool == True)
        
        # Spawn Agents
        self.num_agents = num_agents
        self.agent_bool = np.zeros(map_shape, dtype= bool)
        num_agents_to_spawn = num_agents
        self.agent_positions = np.zeros((num_agents, 2, self.max_runs))
        while num_agents_to_spawn != 0:
            self.agent_bool[np.random.randint(0, map_shape[0], num_agents_to_spawn), np.random.randint(0, map_shape[1], num_agents_to_spawn)] = True
            self.agent_bool[np.logical_or(self.obstacle_bool, self.exit_bool)] = False
            num_agents_to_spawn = num_agents - np.sum(self.agent_bool.astype(int))
        self.agent_positions[:, :, 0] = np.argwhere(self.agent_bool== True)
        
        
    def update(self):
        if self.run < self.max_runs - 1:
            agent_positions = self.agent_positions[:, :, self.run]
            dist_to_exit = distance_matrix(agent_positions, self.exit_positions)
            agent_exits = np.argmin(dist_to_exit, 1)
            agent_dir = np.subtract(self.exit_positions[agent_exits], agent_positions)
            agent_move = self.speed * agent_dir/np.array([np.linalg.norm(agent_dir, axis = 1)]).T
            agent_positions = np.add(agent_positions, agent_move)
            agent_positions[agent_positions < 0] = 0
            agent_positions[agent_positions >= map_shape[0]] = map_shape[0] - 1
            self.run += 1
            self.agent_exits = agent_exits
            self.agent_positions[:, :, self.run] = agent_positions
        
    def run_map(self):
        while self.run < self.max_runs -1:
            self.update()
        self.map_metrics()
        self.draw_map()
        
    def map_metrics(self):
        min_dist_out = np.sum(np.linalg.norm(self.agent_positions[:, :, 0] - self.exit_positions[self.agent_exits], axis = 1))
        X = self.agent_positions[:, 0, :]
        Y = self.agent_positions[:, 1, :]
        dist_out = np.sum(((-np.diff(X))**2 + (-np.diff(Y))**2)**0.5)
        print(min_dist_out, dist_out)
        
    def draw_map(self):
        '''
        Heatmap plot
        self.agent_bool = np.zeros(self.map_shape, dtype= bool)
        self.agent_bool[np.round(self.agent_positions[:, 0]).astype(int), np.round(self.agent_positions[:, 0]).astype(int)] = True
        self.map = np.zeros(self.map_shape, dtype=int)
        self.map[self.obstacle_bool] = 1
        self.map[self.agent_bool] = 2
        self.map[self.exit_bool] = 3
        cmap = ['w', 'k', 'r', 'b']
        sns.heatmap(self.map, cmap=cmap)
        '''

        fig = plt.figure()
        self.agent_positions = self.agent_positions[:, :, :self.run]       
        plt.scatter(self.agent_positions[:, 0, 0], self.agent_positions[:, 1, 0],color = 'r', label = 'Agent')
        plt.scatter(self.exit_positions[:, 0], self.exit_positions[:, 1], color = 'b', label = 'Exit')
        plt.scatter(self.obstacle_positions[:, 0], self.obstacle_positions[:, 1], color = 'k', label = 'Obstacles')
        plt.plot(self.agent_positions[:, 0, :].T, self.agent_positions[:, 1, :].T, 'orange')
        plt.xlim(0, self.map_shape[0])
        plt.ylim(0, self.map_shape[1])
        plt.legend()
        plt.axis("square")
        plt.show()

map_shape = (20, 20)
num_agents = 4
num_obstacles = 4
num_exits = 2
if num_agents + num_exits + num_obstacles >= map_shape[0] * map_shape[1]:
    raise Exception('There are too many objects for the size of map.')

map1 = Map(map_shape, num_agents, num_obstacles, num_exits)


map1.run_map()


# %%
