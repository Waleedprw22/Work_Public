import sim
import pybullet as p
import numpy as np

MAX_ITERS = 10000
delta_q = 2.5 #1 before, 2.5

def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    """
    Draw a line between two positions corresponding to the input configurations
    :param q_1: configuration 1
    :param q_2: configuration 2
    :param env: environment
    :param color: color of the line, please leave unchanged.
    """
    # obtain position of first point
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 9)[0]
    # obtain position of second point
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 9)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)


def SemiRandomSample(steer_goal_p, q_goal):
    num = np.random.random()	
    qrand = np.ones(6)
    
    if num < steer_goal_p: # return qgoal
        return q_goal
    else:
        for i in range(0,6):
           # qrand[i] = np.random.uniform(-np.pi, np.pi)
           qrand[i] = np.random.randint(-180, 180)
           qrand[i] = qrand[i]/180
   
        return qrand
    
def Nearest(V,E, qrand):
    qnear_d= 999999999 
    
    
    for v in V:
       
       if qnear_d>= np.sqrt((sum(((qrand - v)**2)))): #L2 distance
            qnear_d = np.sqrt((sum(((qrand - v)**2))))
            qnear = v
    
   
    return qnear, qnear_d


def Steer(qrand, qnear, delta_q,env,qnear_d):
    qnew = qnear + (qrand-qnear)* (delta_q / qnear_d) #since im using L2 distance
    visualize_path(qnew, qnear, env, color=[0, 1, 0])
   
   
    return qnew


def ObstacleFree(qnew,env):
    #env.set_joint_positions(qnew)
    if env.check_collision(qnew):
        return True
    else:
        return False

def Neighbors(q_curr,E):
    neighbors = []
    for i in range(0,len(E)): #find current node's neighbors
        if list(q_curr) in E[i]:
            if E[i].index(list(q_curr)) == 0:
                neighbors.append(tuple(E[i][1]))
            #if E[i].index(list(q_curr)) == 1:
            #    neighbors.append(tuple(E[i][0]))
        else:
            pass
    
    return neighbors

    
def path(V,E):
    state = (V[len(V)-1]).tolist()
    path = [state]
    
    while tuple(state) != tuple(V[0]):
       
        for (parent, child) in E:
            #if np.all(child == state):
            if tuple(child) == tuple(state):
                
                path.append(parent)
                state = parent
                break
    path.reverse()
    #print('test')
    print(path)
    return path

def BFS(q_init, q_goal, q_new,V,E): #q_new replaces q_goal
    
   

    # the set of visited cells
    closed = set([tuple(q_init)])

    # the set of cells on the current frontier
    frontier = [q_init]

    # back pointers to find the path once reached the goal B. The keys
    # should both be tuples of cell positions (x, y). Value is parent
    pointers = {tuple(q_init):None} #child: parent relation

    # the length of the frontier array, update this variable at each step. 
    frontier_size = [0]
    
    while len(frontier) > 0:
        
        state = frontier.pop(0)
       
        #if state.all() == q_goal.all():
        
        if sum(abs(state - q_goal)) < delta_q:
           
            parent = pointers[tuple(state)] #saves parent of B to parent
         
            path = [q_goal, parent]
            while parent != None:
                node = parent # The "parent" is now a node whose parent we need to find.
               # print('node')
               # print(node)
                path.append(pointers[node]) 
                parent = pointers[node] #Save parent of node to "parent"
            path = list(filter(lambda item: item is not None, path))
                
            #print(len(path))
            #print(path)
            return path #changed closed to pointers
        frontier_size.append(len(frontier))
        adj_nodes = Neighbors(state, E)

        
       
       
        for child in adj_nodes:
            #print('ok')
            #print(child)
            if child not in closed and sum(abs(child - q_goal)) >= delta_q:
                closed.add(child)
                    
                pointers.update({child: state})

                frontier.append(child)

            if child not in closed and sum(abs(child - q_goal)) < delta_q:

                closed.add(child)
                    
                pointers.update({child: state})
                frontier.append(child)
        
        return None#frontier_size is array, trying to find the
    
    return None
def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :returns path: list of configurations (joint angles) if found a path within MAX_ITERS, else None
    """
    # ========= TODO: Problem 3 ========
    # Implement RRT code here. This function should return a list of joint configurations
    # that the robot should take in order to reach q_goal starting from q_init
    # Use visualize_path() to visualize the edges in the exploration tree for part (b)
    
    V = np.array([q_init])
    #print('qinit')
    #print(q_init)
 

    E = []
    i = 1
    path_ = []
    while i<= MAX_ITERS:
        q_rand = SemiRandomSample(steer_goal_p, q_goal) # with steer_goal_p
        
        q_nearest, qnear_d = Nearest(V,E, q_rand)
        q_new = Steer(q_rand,q_nearest, delta_q,env, qnear_d)
        
        if ObstacleFree(q_new,env):
            V = np.append(V,[q_new], axis=0)
            #print(V)

            E.append([q_nearest.tolist(), q_new.tolist()])
            


            if sum(abs(q_new - q_goal)) < delta_q:
                
                V = np.append(V,[q_goal], axis = 0)
                
                E.append([q_new.tolist(), q_goal.tolist()])

                #print(E)
                #print('qnew')
                #print(q_new)
                #path = calculate the path from q_init to q_goal
             
                #return BFS(q_init, q_goal, q_new, V,E) #q_new replaces q_goal
                path_ = path(V,E)
                print('word')
                print(q_goal)
                return path_
            
        print(i)
        i = i + 1
    return None
    

    # ==================================
    #return None

def execute_path(path_conf, env):
    # ========= TODO: Problem 3 ========
    # 1. Execute the path while visualizing the location of joint 5 
    #    (see Figure 2 in homework manual)
    #    You can get the position of joint 5 with:
    #         p.getLinkState(env.robot_body_id, 9)[0]
    #    To visualize the position, you should use sim.SphereMarker
    #    (Hint: declare a list to store the markers)
    # 2. Drop the object (Hint: open gripper, step the simulation, close gripper)
    # 3. Return the robot to original location by retracing the path 
    #position =[]
    #position.append(p.getLinkState(env.robot_body_id, 9)[0])
    sphere = []
    #sphere.append(sim.SphereMarker(position, radius=0.05, rgba_color=(1, 0, 0, 0.8), text=None, orientation=None, p_id=0))

    
  
    #print('yo')
    #print(path_conf)
    #i = len(path_conf) - 1
    i = 0
    print(i)
    while i< len(path_conf):
        
        print('oo')
        env.move_joints(np.array(path_conf[i]), speed=.3) #change .03 to .3
        position = p.getLinkState(env.robot_body_id, 9)[0]
        sphere.append(sim.SphereMarker(position, radius=0.05, rgba_color=(1, 0, 0, 0.8), text=None, orientation=None, p_id=0))
        i = i + 1

    env.open_gripper()
    env.step_simulation(1)
    env.close_gripper()

    i = len(path_conf) -1 
    while i> 0:
        
        print('oo2')
        env.move_joints(np.array(path_conf[i]), speed=.3) #change .03 to .3
        position = p.getLinkState(env.robot_body_id, 9)[0]
        sphere.append(sim.SphereMarker(position, radius=0.05, rgba_color=(1, 0, 0, 0.8), text=None, orientation=None, p_id=0))
        i = i -1


    # ==================================
    #return None