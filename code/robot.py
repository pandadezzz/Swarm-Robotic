import numpy as np
import pybullet as p
import itertools
import math

class Robot():
	""" 
	The class is the interface to a single robot
	"""
	#common varibles for all robots
	Kf=10
	Kt=5
	Ko=10
	Df=2*math.sqrt(Kf)
	Dt=2*math.sqrt(Kf)
	dmax=1
	total_bots =6
	E = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])
	goal_p = [20,10]
	goal_x = goal_p[0]
	goal_y = goal_p[1]
	form_constraint =2

	x,y,c = goal_x,goal_y,form_constraint
	target_p = [x-c,y-c,x-c,y+c,x+c,y-c,x+c,y+c,x-c+1,y-c,x-c,y-c+1 ]# 0,1,2,3 will be the outter corners


	def __init__(self, init_pos, robot_id, dt,obstacles):
		print ('initial position', robot_id, init_pos)

		self.obstacles =obstacles
		self.id = robot_id
		self.dt = dt
		self.pybullet_id = p.loadSDF("../models/robot.sdf")[0]
		self.joint_ids = list(range(p.getNumJoints(self.pybullet_id)))
		self.initial_position = init_pos
		self.reset()

		# No friction between bbody and surface.
		p.changeDynamics(self.pybullet_id, -1, lateralFriction=5., rollingFriction=0.)

		# Friction between joint links and surface.
		for i in range(p.getNumJoints(self.pybullet_id)):
			p.changeDynamics(self.pybullet_id, i, lateralFriction=5., rollingFriction=0.)
			
		self.messages_received = []
		self.messages_to_send = []
		self.neighbors = []

		# extra parameters

		self.waypoint = [(2.5,2),(2.5,6),(-5,6),(-5,10)]
		self.waypoint_index=0 # used to record the current waypoint



		self.D = self.getIncidenceMatrix(self.E,self.total_bots)
		self.L = self.getLaplacian(self.E,self.total_bots)
		self.target_x, self.target_y= self.split_p(self.target_p)

		self.z_ref_x=np.dot(np.transpose(self.D),self.target_x)
		self.z_ref_y=np.dot(np.transpose(self.D),self.target_y)

		self.z_ref = self.combine_p(self.z_ref_x,self.z_ref_y)
		self.p_dot = np.zeros([1,self.total_bots*2])#.reshape(1,self.total_bots)




	def reset(self):
		p.resetBasePositionAndOrientation(self.pybullet_id, self.initial_position, (0., 0., 0., 1.))
			
	def set_wheel_velocity(self, vel):
		""" 
		Sets the wheel velocity,expects an array containing two numbers (left and right wheel vel) 
		"""
		assert len(vel) == 2, "Expect velocity to be array of size two"
		p.setJointMotorControlArray(self.pybullet_id, self.joint_ids, p.VELOCITY_CONTROL,
			targetVelocities=vel)

	def get_pos_and_orientation(self):
		"""
		Returns the position and orientation (as Yaw angle) of the robot.
		"""
		pos, rot = p.getBasePositionAndOrientation(self.pybullet_id)
		euler = p.getEulerFromQuaternion(rot)
		return np.array(pos), euler[2]
	
	def get_messages(self):
		return self.messages_received
		
	def send_message(self, robot_id, message):
		self.messages_to_send.append([robot_id, message])
		
	def get_neighbors(self):
		return self.neighbors
	
	def get_distance(self,p1,p2):
		#p1 looks like [0,1,2] x,y,z 
		dis = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
		return dis

	def getBotPos(self,pos_dict):
		#given a dictionary, return array of bot positions
		# in format [x1,y1,x2,y2......]
		pos = []
		for i in range(self.total_bots):
			bot_pos = pos_dict[i]
			x = bot_pos[0]
			y = bot_pos[1]
			pos.append(x)
			pos.append(y)

		return np.array(pos)

	def getLaplacian(self,E,n_vertex):
		L = np.zeros([n_vertex,n_vertex]) #our Laplacian matrix
		Delta = np.zeros([n_vertex,n_vertex]) #this is the degree matrix
		A = np.zeros([n_vertex,n_vertex]) #this is the adjacency matrix
		for e in E: #for each edge in E
			#add degrees
			Delta[e[1],e[1]] +=1
			#add the input in the adjacency matrix
			A[e[1],e[0]] = 1
			#symmetric connection as we have undirected graphs
			Delta[e[0],e[0]] +=1
			A[e[0],e[1]] = 1
		L = Delta - A
		return L

	# get incidence matrix for directed graph E (list of edges)
	def getIncidenceMatrix(self,E,n_vertex):
		n_e = len(E)
		D = np.zeros([n_vertex,n_e])
		for e in range(n_e):
			#add the directed connection
			D[E[e][0],e] = -1
			D[E[e][1],e] = 1
		return D

	def split_p(self,p):
		x=[]
		y=[]
		for i in range(int(len(p)/2)):
			x.append(p[2*i])
			y.append(p[2*i+1])
		return np.array(x),np.array(y)

	def combine_p(self,x,y):
		p=[]
		for i in range(len(x)):
			p.append(x[i])
			p.append(y[i])
		return np.array(p)

	def distance(self,x1,y1,x2,y2):
		# distance between 2 points
		dis = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
		return dis

	def getF(self,dis,px,py,ox,oy):
		F_x = -Ko * ((dis - 1)/(dis)**3) * (px - ox)
		F_y = -Ko * ((dis - 1)/(dis)**3) * (py - oy)
		return F_x,F_y
		
	def compute_next_pos(self):
		n_vertex=self.total_bots

		pos_x,pos_y = self.split_p(self.bot_pos) # current pos
		pos_dot_x, pos_dot_y = self.split_p(self.p_dot[0]) # current velocity
		print('before',self.p_dot[0])

		#print(np.dot(self.L,pos_x).reshape(1,6))
		Lx = np.dot(self.L,pos_x).reshape(1,n_vertex)
		Ly = np.dot(self.L,pos_y).reshape(1,n_vertex)

		Dzx_desire= np.dot(self.D,self.z_ref_x).reshape(1,n_vertex)
		Dzy_desire= np.dot(self.D,self.z_ref_y).reshape(1,n_vertex)

		#print(self.L,self.p_dot)

		Lx_dot = np.dot(self.L,pos_dot_x).reshape(1,n_vertex)
		Ly_dot = np.dot(self.L,pos_dot_y).reshape(1,n_vertex)

		x_formation = self.Kf*(Dzx_desire-Lx)-self.Df*Lx_dot
		y_formation = self.Kf*(Dzy_desire-Ly)-self.Df*Ly_dot

		##### translation
		#get the direction

		dir_x = self.target_x-pos_x
		dir_y = self.target_y-pos_y
		x_diff_min =[]
		y_diff_min =[] 
		for i in range(n_vertex):
			x_diff_min.append(min(dir_x[i],1))
			y_diff_min.append(min(dir_y[i],1))

		x_diff_min=np.array(x_diff_min).reshape(1,n_vertex)
		y_diff_min=np.array(y_diff_min).reshape(1,n_vertex)

		x_dot_dir = np.array(pos_dot_x).reshape(1,n_vertex)
		y_dot_dir = np.array(pos_dot_y).reshape(1,n_vertex)

		x_target = self.Kt*x_diff_min-self.Dt*x_dot_dir
		y_target = self.Kt*y_diff_min-self.Dt*y_dot_dir
		'''
		for i in range(n_vertex):
			for obs in range(len(obstacles)):
				dis[i][obs]= distance(pos_x[i],pos_y[i],obs_x[i],obs_y[i])

		for i in range (n_vertex):
				for obs in range (len(obstacles)):
					if di[i][obs] < dmax:
						F_obs_x[i][obs],F_obs_y[i][obs] = F_calc(dis[i][obs],pos_x[i],pos_y[i],obs_x[obs],obs_y[obs])

		F_x= np.zeros(n_vertex)
		F_y= np.zeros(n_vertex)
		for x in range (len(F_obs_x)):
				for y in range (len(F_obs_x[x])):
					F_x[x] = F_x[x] + F_obs_x[x][y]
					F_y[x] = F_y[x] + F_obs_x[x][y]
		'''

		#### accerlation

		x_pos_d_dot = x_formation+x_target#+F_x
		y_pos_d_dot = y_formation+y_target#+F_y
		#pos_d_dot=self.combine_p(x_pos_d_dot[0],y_pos_d_dot[0])
		
		# velocity = current + acc
		x_pos_dot_next = pos_dot_x + self.dt*x_pos_d_dot
		y_pos_dot_next = pos_dot_y + self.dt*y_pos_d_dot

		self.p_dot[0] = self.combine_p(x_pos_dot_next[0],y_pos_dot_next[0])
		print('after',self.p_dot[0])
		#self.p_dot[0]
		#position self.dt
		x_pos_next = pos_x + self.dt*x_pos_dot_next
		y_pos_next = pos_y + self.dt*y_pos_dot_next

		#get new position of this robot
		new_x = x_pos_next[0][self.id]
		new_y = y_pos_next[0][self.id]
		#pos[t+1] = combine_p(x_pos_next[0],y_pos_next[0])
		return new_x,new_y

	def getAStarRoute(self):
		obs = self.obstacles
		#looks like [[0., -1., 0,0],[0., 1., 0,1], where 4th element is orientation. 1=y axis oritation dimension of wall is 2,0.15, 0.5  LxWxH
		# first compute the map. now we only know the midpoint of wall and the length of wall

	def navigate_toWaypoints(self):
		#return dx, dy

		x1,y1= self.pos[0],self.pos[1] # our current location
		target_x,target_y= self.waypoint[self.waypoint_index][0],self.waypoint[self.waypoint_index][1]

		#check if close enough
		if self.distance(x1,y1,target_x,target_y)<0.5 and self.waypoint_index<len(self.waypoint)-1:
			#close enough
			self.waypoint_index+=1
			#get new target
			print("new waypoint")
			target_x,target_y= self.waypoint[self.waypoint_index][0],self.waypoint[self.waypoint_index][1]

		return target_x-x1,target_y-y1	

	def potentialField(self):
		# obstracle avoidance 

		x1,y1= self.pos[0],self.pos[1]

		#get all positions in correct format

		d= np.zeros((self.total_bots,self.total_bots))
		F_obs_x = np.zeros((self.total_bots,self.total_bots))
		F_obs_y = np.zeros((self.total_bots,self.total_bots))
		for i in range (self.total_bots):
			for j in range (self.total_bots):
				#get distance
				d[i][j] = distance(x_pos[number_of_robot],y_pos[number_of_robot],obs_x[obstracle],obs_y[obstracle])
		for number_of_robot in range (n_vertex):
			for obstracle in range (len(obstacles)):
				if d[number_of_robot][obstracle] < 1:
					F_obs_x[number_of_robot][obstracle],F_obs_y[number_of_robot][obstracle] = F_calc(d[number_of_robot][obstracle],x_pos[number_of_robot],y_pos[number_of_robot],obs_x[obstracle],obs_y[obstracle])
		
		F_x = [0]*n_vertex    
		F_y = [0]*n_vertex
		
		for x in range (len(F_obs_x)):
			for y in range (len(F_obs_x[x])):
				F_x[x] = F_x[x] + F_obs_x[x][y]
				F_y[x] = F_y[x] + F_obs_y[x][y]



	def compute_controller(self):
		""" 
		function that will be called each control cycle which implements the control law
		TO BE MODIFIED
		
		we expect this function to read sensors (built-in functions from the class)
		and at the end to call set_wheel_velocity to set the appropriate velocity of the robots
		"""
		
		# here we implement an example for a consensus algorithm

		

		neig = self.get_neighbors() # returns a list of neighbor and their ID i.e [1,2,3]
		messages = self.get_messages()
		self.pos, rot = self.get_pos_and_orientation()
		
		
		self.all_positions={self.id:self.pos}
		for message in messages:
			#message = id,pos_dict
			id = message[0]
			pos_dict = message[1]
			for key,value in pos_dict.items():
				if key!= self.id:
					#update location
					self.all_positions[key]=value

		#send message of positions to all neighbors indicating our position
		for n in neig:
			self.send_message(n, self.all_positions)
		print(self.all_positions)
		#print(len(positions))
		# check if we received the position of our neighbors and compute desired change in position
		# as a function of the neighbors (message is composed of [neighbors id, position])
		'''
		current_x=self.pos[0]
		current_y=self.pos[1]
		new_x,new_y = current_x,current_y

		if (len(self.positions)==self.total_bots):
			# only start when we have location of all robots

			# get the postions of bots
			self.bot_pos = self.getBotPos(self.positions)
			#in format [x1,y1,x2,y2......]

			#D = getIncidenceMatrix(E,n_vertex)
			#L = getLaplacian(E,n_vertex
			new_x,new_y= self.compute_next_pos()
		'''
		#print(self.id,new_x,current_x)

		if self.id==5:   #new_x!= current_x or new_y!= current_y:
			#print('update')
			# if we need to update postion
			#dx=new_x-current_x
			#dy=new_y-current_y
			dx,dy = self.navigate_toWaypoints()
			dx*=10
			dy*=10

			vel_norm = np.linalg.norm([dx, dy]) #norm of desired velocity
			if vel_norm < 0.01:
				vel_norm = 0.01
			des_theta = np.arctan2(dy/vel_norm, dx/vel_norm)
			right_wheel = np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
			left_wheel = -np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
			self.set_wheel_velocity([left_wheel, right_wheel])
		'''
		dx = 0.
		dy = 0.
		if messages:
			
			for m in messages:
				pos_dict = m[1]

				if self.get_distance(pos,neighbor_pos)>=1:
					dx += neighbor_pos[0] - pos[0]
					dy += neighbor_pos[1] - pos[1]
			# integrate
			des_pos_x = pos[0] + self.dt * dx
			des_pos_y = pos[1] + self.dt * dy
			
		
			#compute velocity change for the wheels
			
		
'''
	
	   
