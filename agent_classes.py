#%%
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy.spatial import distance_matrix
from scipy.ndimage import binary_dilation
from sklearn.cluster import KMeans

#%%    
class Map():
    def __init__(self, map_shape, num_agents, num_obstacles, num_exits, num_drones = 0, max_runs = 10, obstacle_bool = np.array([]), exit_pos = np.array([])):
        self.map_shape = map_shape
        self.map = np.zeros(self.map_shape)
        self.run = 0
        self.max_runs = max_runs
        self.leading = np.array([np.linspace(0.5, 1, num_agents)]).T
        
        # Make obstacles
        if obstacle_bool.size == 0:
            self.obstacle_bool = np.zeros(np.array(map_shape), dtype=bool)
            num_obstacles_to_spawn = num_obstacles
            while num_obstacles_to_spawn != 0:
                self.obstacle_bool[np.unravel_index(np.random.choice(np.linspace(0, self.map.size, self.map.size, endpoint=False, dtype = int), num_obstacles_to_spawn), self.map.shape)] = 1
                num_obstacles_to_spawn = num_obstacles - np.sum(self.obstacle_bool.astype(int))
        else:
            self.obstacle_bool = obstacle_bool
        
        #Make edge obstacles
        edges = np.zeros((2*np.sum(np.array(map_shape))-2, 2))
        edges[:map_shape[0]*2 +4, 0] = np.repeat(np.array([np.linspace(-1, map_shape[1], map_shape[1]+2)]).T, 2, axis= 1).T.flatten()
        edges[:map_shape[0]*2 +4, 1] = np.repeat(np.array([-1, map_shape[1]]), (map_shape[1] + 2))
        edges[map_shape[0]*2 +4:, 0] = np.repeat(np.array([-1, map_shape[1]]), (map_shape[1] - 3))
        edges[map_shape[0]*2 +4:, 1] = np.repeat(np.array([np.linspace(0, map_shape[1]-1, map_shape[1]-3)]).T, 2, axis= 1).T.flatten()
        self.obstacle_positions = np.concatenate((np.argwhere(self.obstacle_bool == True), edges))
        
        # Make exits
        self.exit_bool = np.zeros(map_shape, dtype= bool)
        
        exit_loc = np.zeros((num_exits, 2), dtype=int)
        i = 0 
        while i < num_exits:
            exit_loc[i, :] = np.array([np.random.randint(-1, 1), np.random.randint(0, map_shape[0])])[:: (-1, 1)[np.random.randint(0, 2)]]
            self.exit_bool[exit_loc[i, 0], exit_loc[i, 1]] = True
            self.exit_bool[self.obstacle_bool] = False
            i = np.unique(exit_loc, axis =0).shape[0]
        self.exit_positions = np.unique(exit_loc, axis =0)
        
        # Spawn Agents
        self.num_agents = num_agents
        self.agent_in = np.ones(num_agents, int)
        self.agent_bool = np.zeros(map_shape, dtype= bool)
        num_agents_to_spawn = num_agents
        self.agent_positions = np.zeros((num_agents, 2, self.max_runs))
        while num_agents_to_spawn != 0:
            self.agent_bool[np.random.randint(0, map_shape[0], num_agents_to_spawn), np.random.randint(0, map_shape[1], num_agents_to_spawn)] = True
            self.agent_bool[np.logical_or(self.obstacle_bool, self.exit_bool)] = False
            num_agents_to_spawn = num_agents - np.sum(self.agent_bool.astype(int))
        self.agent_positions[:, :, 0] = np.argwhere(self.agent_bool== True)
        self.calm = np.zeros((num_agents, max_runs))

        exit_dis = np.zeros(map_shape)
        exit_dis[self.exit_bool] = 1
        iter = 1
        struct = np.ones((3, 3))
        struct[1, 1] = 0
        '''
        while (exit_dis[np.logical_not(self.obstacle_bool)] == 0).any():
            iter += 1
            exit_dis[np.logical_and(binary_dilation(exit_dis, struct.astype(bool), mask=np.logical_not(self.obstacle_bool)), exit_dis ==0)] = iter
        exit_dis[self.obstacle_bool] = np.inf
        '''
        # Make drones
        self.drone_pos = np.zeros((num_drones, 2, max_runs))
        self.num_drones = num_drones
        self.drone_speed = 5
        self.drone_pos[:, :, 0] = np.repeat(self.exit_positions, num_drones // num_exits + 1, axis = 0)[:num_drones]
        self.drone_state = np.zeros((num_drones, 1))
        
    def cal_exit_force(self, positions, exit_positions):
        dist_to_exit = distance_matrix(positions, exit_positions)
        agent_exits = np.argmin(dist_to_exit, 1)
        exit_force = np.subtract(exit_positions[agent_exits], positions)
        exit_force = np.divide(exit_force, np.array([np.linalg.norm(exit_force, axis =1)]).T)
        return dist_to_exit, exit_force, agent_exits
        
    def dijkstra(self, drone_pos, obstacle_pos):
        node_dist = np.ones_like(self.map_shape) * np.inf
        node_dist[drone_pos] = 0
        node_costs = np.ones_like(self.map_shape)
        node_costs[self.obstacle_bool] = 100
        struct = np.ones((3, 3))
        node_costs[binary_dilation(self.obstacle_bool.astype(int), struct)] = 10
        while node_dist[obstacle_pos] == np.inf:
            node = np.zeros(self.map_shape)
            node[np.argmin(node_dist)]
            node_dist[node_dist[binary_dilation(node, struct) & node_dist == np.inf]] += node_costs[node_dist[binary_dilation(node, struct) & node_dist == np.inf]]
        
    def distance_between(self, thing1_pos, thing2_pos, mode):
        d = distance_matrix(thing1_pos, thing2_pos)
        d[d == 0] = 1
        
        # dx and dy are the directions that the forces act in
        if mode == 0:
            dx = np.subtract(thing1_pos[:, 0], np.array([thing2_pos[:, 0]]).T)/d
            dy = np.subtract(thing1_pos[:, 1], np.array([thing2_pos[:, 1]]).T)/d
        elif mode == 1:
            dx = np.subtract(np.array([thing1_pos[:, 0]]).T, thing2_pos[:, 0])/d
            dy = np.subtract(np.array([thing1_pos[:, 1]]).T, thing2_pos[:, 1])/d
        
        dx[np.isnan(dx)] = 0
        dy[np.isnan(dy)] = 0
        return d, dx, dy
        
    def apply_force(self, force, dx, dy):
        force_x = np.sum(np.multiply(force, dx), axis = 1)
        force_y = np.sum(np.multiply(force, dy), axis = 1)
        return np.concatenate((np.array([force_x]).T, np.array([force_y]).T), axis = 1)
        
    def update(self):
        if self.run < self.max_runs - 1:
            if self.run % 20 == 0:
                print(self.run)
            agent_pos = self.agent_positions[:, :, self.run]
            dist_to_exit, exit_force, self.agent_exits = self.cal_exit_force(agent_pos, self.exit_positions)
            self.agent_in[np.min(dist_to_exit, axis= 1)<1] = 0
            
            d, dx, dy = self.distance_between(agent_pos, agent_pos, 0)
            d = d * np.array([self.agent_in]).T
            agent_force = np.multiply(np.exp(-(d/0.5)), np.array([self.agent_in]).T)
            agent_force_dir = self.apply_force(agent_force, dx, dy)
            
            # Calmness of the agents is based on the n closest agents
            close_agents = 10
            calm = np.ones(num_agents)
            calm[np.any((d < 1) & (d>0), axis=1)] = np.sum(np.sort(d, axis=1)[:, :close_agents], axis=1)[np.any((d < 1) & (d>0), axis=1)]/close_agents
            calm[calm < 0.5] = 0.5
            
            # Crushes of the agents is based on the n closest agents
            crush_agents = 10
            crush = np.ones(num_agents)
            crush = np.count_nonzero((d < 1), axis = 1).astype(float)
            crush[crush < crush_agents] = 1
            crush[crush > crush_agents] = crush[crush > crush_agents]/ crush_agents
            crush[crush > 2] = 2
            
            agent_force_dir = np.multiply(agent_force_dir, np.array(crush_agents).T)
            
            do, dox, doy = self.distance_between(agent_pos, self.obstacle_positions, 1)
            obstacle_force = np.multiply(np.exp(-(do*5))*100, np.array([self.agent_in]).T)
            obstacle_force_dir = self.apply_force(obstacle_force, dox, doy)
            
            # Drone force
            drone_pos = self.drone_pos[:, :, self.run]
            if np.sum(self.drone_state.T[0]) > 0:
                dd, drx, dry = self.distance_between(agent_pos, drone_pos[self.drone_state.astype(bool).T[0], :], 1)
                drone_force = -np.exp(-dd/5)*2
                drone_force_dir = self.apply_force(drone_force, drx, dry)
            else:
                drone_force_dir = np.zeros((self.num_agents, 2))
            
            # Drone movement
            if np.sum(self.agent_in) > num_drones:
                kmeans = KMeans(n_clusters=self.num_drones).fit(np.unique(np.multiply(agent_pos, np.repeat(np.array([self.agent_in]).T, 2, axis =1)),axis=0))
                cluster_centers = kmeans.cluster_centers_
            else:
                cluster_centers = self.drone_pos[:, :, 0]
                cluster_centers[:(np.sum(self.agent_in)), :] = agent_pos[self.agent_in.astype(bool), :]
            
            drone_exit_dis = distance_matrix(drone_pos, self.exit_positions)
            self.drone_state[np.any(np.abs(drone_exit_dis) < 1, axis = 1)] = 0
            
            drone_centroid_dis = distance_matrix(drone_pos, cluster_centers)
            self.drone_state[np.any(np.abs(drone_centroid_dis) < 1, axis = 1)] = 1
            
            drone_dir = np.subtract(cluster_centers, drone_pos)
            drone_dir[np.isnan(drone_dir)] = 0
            drone_dir = drone_dir/np.array([np.linalg.norm(drone_dir, axis = 1)]).T
            
            dist_to_exit, drone_exit_force, drone_exits = self.cal_exit_force(drone_pos, self.exit_positions)
            drone_exit_force[np.isnan(drone_exit_force)] = 0
            
            # Drone obstacle avoidance
            drone_obs_dist, drone_obs_dist_x, drone_obs_dist_y = self.distance_between(drone_pos, self.obstacle_positions, 1)
            drone_obs_force = np.exp(-(drone_obs_dist)*5)*100
            drone_obs_force_dir = self.apply_force(drone_obs_force, drone_obs_dist_x, drone_obs_dist_y)
            
            self.drone_pos[:, :, self.run + 1] = drone_pos + drone_obs_force_dir + np.multiply(drone_dir, np.repeat(-1*(self.drone_state-1), 2, axis = 1)) + np.multiply(np.add(drone_exit_force * 0.7, drone_dir*0.3), np.repeat(self.drone_state, 2, axis = 1))
            
            # Sum forces
            total_force = (exit_force*self.leading + agent_force_dir + obstacle_force_dir + drone_force_dir)* np.array([calm]).T
            
            # Weighted movement force
            nearby = 5
            weighted_force = np.zeros_like(total_force)
            dc = np.zeros_like(d, dtype=bool)
            dc[d < nearby] = True
            dc[self.agent_in, :] = False
            dc[self.agent_in, :] = True
            for i in range(num_agents):
                weighted_force[i, :] += np.sum(total_force[dc[i, :], :], axis =0)/np.sum(dc[i, :])
            total_force = np.multiply(total_force, (np.ones((num_agents, 1))- self.leading))/3 + total_force
            
            agent_move = np.multiply(total_force/np.array([np.linalg.norm(total_force, axis = 1)]).T, np.array([self.agent_in]).T)
            agent_move[np.isnan(agent_move)] = 0
            agent_pos = np.add(agent_pos, agent_move)
            self.run += 1
            self.agent_positions[:, :, self.run] = agent_pos
        
    def run_map(self):
        while self.run < self.max_runs -1 and self.agent_in.any():
            self.update()
        self.map_metrics()
        self.draw_map()
        
    def map_metrics(self):
        min_dist_out = np.sum(np.linalg.norm(self.agent_positions[:, :, 0] - self.exit_positions[self.agent_exits], axis = 1))
        X = self.agent_positions[:, 0, :]
        Y = self.agent_positions[:, 1, :]
        dist_out = np.sum(((-np.diff(X))**2 + (-np.diff(Y))**2)**0.5)
        print(min_dist_out, dist_out, dist_out/min_dist_out, self.run, np.sum(self.agent_in)/num_agents)
        
    def draw_map(self):
        # Animation plot
        fig1, ax = plt.subplots()
        ax.scatter(self.exit_positions[:, 0], self.exit_positions[:, 1], color = 'b', label = 'Exit')
        ax.scatter(self.obstacle_positions[:, 0], self.obstacle_positions[:, 1], color = 'k', label = 'Obstacles')
        scat = ax.scatter(self.agent_positions[:, 0, 0], self.agent_positions[:, 1, 0],color = 'r', label = 'Agent')
        scat1 = ax.scatter(self.drone_pos[:, 0, 0], self.drone_pos[:, 1, 0],color = 'orange', label = 'Drone')
        #plt.plot(self.agent_positions[:, 0, :].T, self.agent_positions[:, 1, :].T, 'orange')
        ax.set(xlim=[-1, self.map_shape[0]], ylim=[-1, self.map_shape[1]])
        ax.axis('square')
        ax.legend(loc='upper left')
        #plt.savefig('crush_crowd.png', dpi = 500)
        #plt.show()
        def update(frame):
            x = self.agent_positions[:, 0, frame]
            y = self.agent_positions[:, 1, frame]
            data = np.stack([x, y]).T
            scat.set_offsets(data)
            
            x1 = self.drone_pos[:, 0, frame]
            y1 = self.drone_pos[:, 1, frame]
            data1 = np.stack([x1, y1]).T
            scat1.set_offsets(data1)
            return (scat), #scat1)
        
        animation = ani.FuncAnimation(fig=fig1, func=update, frames=self.run, interval=50, repeat = True, repeat_delay = 500)
        plt.show()
        animation.save(filename="crowd_5.gif", writer="pillow")
        
map_shape = (50, 50)
num_agents = 100
num_obstacles = 50
num_exits = 10
max_runs = 500
num_drones = 1
if num_agents + num_exits + num_obstacles >= map_shape[0] * map_shape[1]:
    raise Exception('There are too many objects for the size of map.')

obstacle_bool = np.zeros(map_shape, dtype=bool)
num_line = np.linspace(0, map_shape[0], map_shape[0], endpoint=False, dtype=int)

obstacle_bool[num_line%10 < 5, 0:map_shape[0]:5] = True
#obstacle_bool[0:map_shape[0]:10, (num_line%10) < 5] = True

'''
num_obstacles_to_spawn = num_obstacles
while num_obstacles_to_spawn != 0:
    obstacle_bool[np.unravel_index(np.random.choice(np.linspace(0, 2500, 2500, endpoint=False, dtype = int), num_obstacles_to_spawn), map_shape)] = 1
    num_obstacles_to_spawn = num_obstacles - np.sum(obstacle_bool.astype(int))
obstacle_bool = binary_dilation(obstacle_bool, np.ones((3,3)))
'''

obstacle_bool[:5, :] = False
obstacle_bool[-5:, :] = False
obstacle_bool[:, :5] = False
obstacle_bool[:, -5:] = False

map1 = Map(map_shape, num_agents, num_obstacles, num_exits, num_drones, max_runs, obstacle_bool)

#map1.draw_map()

map1.run_map()