'''
    James Hooper
    NETID: jah171230
    CS 4365 Artificial Intelligence
    Homework 1: Search Algorithms
'''
import sys
import numpy as np

'''
    State Class
    ~ have an initial root state then send that to an algorithm function
    ~ states are connected via 'pointers'
    ~ state is comprised of:
        ~ puzzle layout: [x,x,x]
                         [x,*,x]
                         [x,x,x]
        ~ parent puzzle layout
        ~ goal state
        ~ moves that are allowed
        ~ depth value (check if exceeds 10)
        ~ heuristic cost value
'''
class State:
    def __init__(self, state, parent, goal, moves, depth, h_cost):
        self.state = state
        self.parent = parent
        self.goal = goal
        self.moves = moves
        self.depth = depth
        self.h_cost = h_cost

    # print state array
    def print_state(self):
        for rows in range(0,3):
            for cols in range(0,3):
                print(self.state[rows,cols], end = " ")
            # new line except last line since return will cause a new line
            if(rows != 2):
                print('')
        return ''

    # print list of possible movements
    def print_moves(self):
        return self.moves

'''
    Calculate all possible movements for a specific state
        ~ simply returns the proper index values that will be switched in allowed movements
'''
def movements(state):
    moves = []
    # find index values for '*' in state array
    index = np.where(state == '*')
    row = index[0][0]
    column = index[1][0]
    '''
          columns
           0 1 2
         0[x,x,x]
    rows 1[x,x,x]
         2[x,x,x]
    '''
    # there is a blank spot up a row
    if(row > 0):
        moves.append([row-1, column])
    # there is a blank spot down a row
    if(row < 2):
        moves.append([row+1, column])
    # there is a blank spot left a column
    if(column > 0):
        moves.append([row, column-1])
    # there is a blank spot right a column
    if(column < 2):
        moves.append([row, column+1])

    # With each move we can swap the '*' tile with the potential move tile
    return moves

'''
    Expand Movements
        ~ this function will return a list of expanded state layouts (not the whole state itself)
        ~ reason being that we don't want to calculate the heuristics for the non-A* functions
'''
def expand_movements(parent_state):
    expansion = []
    # Obtain empty tile location
    index = np.where(parent_state.state == '*')
    row_2 = index[0][0]
    column_2 = index[1][0]

    for moves in parent_state.moves:
        # Create layout that initializes to parent state layout
        # Make sure to copy to actually create a new instance of a numpy array
        new_layout = parent_state.state.copy()
        # Obtain non-empty tile location to be moved
        row_1 = moves[0]
        column_1 = moves[1]
        # Have the empty tile become the desired tile to swap
        new_layout[row_2, column_2] = parent_state.state[row_1, column_1]
        # Make the swapped tile locaiton become an empty tile
        new_layout[row_1,column_1] = '*'
        #expansion.append(State(new_layout, parent_state, goal_state, movements(new_layout), 0, 0))
        # Append new_layout to progress through
        expansion.append(new_layout)

    return expansion

'''
Heuristic Functions
'''
# Heuristic 1
def misplaced_tiles(state, goal_state):
    h1 = 0
    for i in range(0,3):
        for j in range(0,3):
            # Does not count where * "tile" is, only the real tiles
            if (state[i, j] != '*'):
                if(state[i,j] != goal_state[i,j]):
                    # misplaced tile
                    h1+=1
    return h1

# Heuristic 2
def manhattan_distance(state, goal_state):
    h2 = 0
    for i in range(0,3):
        for j in range(0,3):
            # go through each tile index of the state
            # find where this value is in the goal state
            # calculate m_dist with these found indexes
            if(state[i,j] != '*'):
                x1 = i
                y1 = j
                index = np.where(goal_state == state[i][j])
                x2 = index[0][0]
                y2 = index[1][0]
                h2 += abs(x2 - x1) + abs(y2 - y1)
    return h2

'''
    Searches/Algorithms
'''
def bfs(state):
    # Define Max Depth value
    max_depth = 0

    # Check if initial state layout is the same as the goal node
    # Remember: the goal test is applied to each node when it is generated
    #           rather than when it is selected for expansion
    if np.array_equal(state.state, state.goal):
        return state

    # FIFO queue of states
    frontier = []
    # Explored set of state layouts
    explored = []
    # Add the initial state to the frontier
    frontier.append(state)

    while frontier:
        # Treat 0 as beginning and appending to end
        parent = frontier.pop(0)
        # add s state to explored
        explored.append(parent.state)

        # Expand the movements
        # This just gives back the layouts
        actions = expand_movements(parent)

        # for each layout/move from the expanded layouts/moves
        for layout in actions:
            # define the actual State
            child = State(layout, parent, parent.goal, movements(layout), parent.depth+1, None)
            # We need to check if a numpy array exists in a list of numpy arrays
            # We will do this twice for the explored and frontier sets
            in_explored = any((layout == x).all() for x in explored)
            in_frontier = any((layout == x).all() for x in frontier)
            if not(in_explored or in_frontier):
                # Check if state layout is the same as the goal node
                if np.array_equal(child.state, parent.goal):
                    return child
                frontier.append(child)
                # If depth 10 we exit at this point since no layout matches
                #max_depth = child.depth
        #if max_depth == 10:
          #  return  # child, false

def ids():
    # ids code
    print("ids")

def astar1():
    # astar1 code
    print("astar1")

def astar2():
    # astar2 code
    print("astar2")

'''
    Returns steps from initial state to goal state
        ~ Input: goal found child node
        ~ Output: goal path from initial node to child node
'''
def goal_path(node):
    goal_path = []
    goal_path.append(node.state)
    while node.parent != None:
        # append parent layout
        goal_path.append(node.parent.state)
        node = node.parent
    return reversed(goal_path)

'''
    Prints out the goal path
'''
def print_goal_path(goal_path):
    print("Initial input state\n")
    for state in goal_path:
        print(state, "\n")
    print("Goal state")

if __name__ == '__main__':
    # if the arguments for the algorithm and initial puzzle sequence exist
    # there has to be an initial input, then the alg, then 9 more inputs after
    if(len(sys.argv) != 11):
       print("Improper arguments.")
       exit(0)
    '''
        Let's:
            ~ get our initial state
            ~ define our goal state
            ~ and create our root State class to be sent to the algorithm
    '''
    initial_state = np.array([[sys.argv[2], sys.argv[3], sys.argv[4]],
                              [sys.argv[5], sys.argv[6], sys.argv[7]],
                              [sys.argv[8], sys.argv[9], sys.argv[10]]])
    goal_state = np.array([['1','2','3'],
                           ['4','5','6'],
                           ['7','8','*']])
    '''
        Let's get our algorithm
    '''
    algorithms = ["bfs", "ids", "astar1", "astar2"]
    chosen_algorithm = ""
    found_algorithm = False
    for algorithm in algorithms:
        if(sys.argv[1] == algorithm):
            chosen_algorithm = algorithm
            found_algorithm = True

    '''
        Let's call the algorithm we got
            ~ State arguments:
                ~ state, parent, goal, moves, depth, h_cost
    '''
    # Error; algorithm not found
    if(found_algorithm == False):
        print("Chosen algorithm does not exist.")

    # BFS Algorithm
    elif(chosen_algorithm == algorithms[0]):
        # set root state
        root = State(initial_state, None, goal_state, movements(initial_state), 0, 0)
        # call algorithm
        found_goal = bfs(root)
        goal_path = goal_path(found_goal)
        print_goal_path(goal_path)

    # IDS Algorithm
    elif(chosen_algorithm == algorithms[1]):
        # set root state
        root = State(initial_state, None, goal_state, movements(initial_state), 0, 0)
        # call algorithm
        ids()

    # Astar1 Algorithm
    elif (chosen_algorithm == algorithms[2]):
        # set root state
        root = State(initial_state, None, goal_state, movements(initial_state), 0,
                     misplaced_tiles(initial_state, goal_state))
        # call algorithm
        astar1()

    # Astar2 Algorithm
    elif (chosen_algorithm == algorithms[3]):
        # set root state
        root = State(initial_state, None, goal_state, movements(initial_state), 0,
                     manhattan_distance(initial_state, goal_state))
        print(root.state,'\n')
        for moves in expand_movements(root):
            print(moves,'\n')
        # call algorithm
        astar2()

# once we get the path, go up the parent, then reverse the list
