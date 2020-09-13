'''
    James Hooper
    NETID: jah171230
    CS 4365 Artificial Intelligence
    Homework 1: Search Algorithms
'''
import sys

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
    def __init__(self, state, parent, goal, moves, depth, g_cost, f_cost):
        self.state = state
        self.parent = parent
        self.goal = goal
        self.moves = moves
        self.depth = depth
        self.g_cost = g_cost
        self.f_cost = f_cost

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
    index = find_char(state, '*')
    row = index[0]
    column = index[1]
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
    index = find_char(parent_state.state, '*')
    row_2 = index[0]
    column_2 = index[1]

    for moves in parent_state.moves:
        # Create layout that initializes to parent state layout
        # Make sure to copy to actually create a new instance of a numpy array
        new_layout = matrix_3x3_copy(parent_state.state)
        # Obtain non-empty tile location to be moved
        row_1 = moves[0]
        column_1 = moves[1]
        # Have the empty tile become the desired tile to swap
        new_layout[row_2][column_2] = parent_state.state[row_1][column_1]
        # Make the swapped tile locaiton become an empty tile
        new_layout[row_1][column_1] = '*'
        #expansion.append(State(new_layout, parent_state, goal_state, movements(new_layout), 0, 0))
        # Append new_layout to progress through
        expansion.append(new_layout)
    return expansion

def matrix_3x3_copy(matrix):
    # 3x3 matrix
    new_matrix=[[0,0,0],[0,0,0],[0,0,0]]
    # copy over
    for i in range(0,3):
        for j in range(0,3):
            new_matrix[i][j] = matrix[i][j]
    return new_matrix

'''
    Searches/Algorithms
'''
# Breadth First Search
def bfs(state):
    # Define Max Depth value
    max_depth = 0
    pass_depth = False
    # Total number of states ever enqueued
    enqueued_num = 0

    # Check if initial state layout is the same as the goal node
    # Remember: the goal test is applied to each node when it is generated
    #           rather than when it is selected for expansion
    if state_equal(state.state, state.goal):
        return [state, pass_depth, enqueued_num]

    # FIFO queue of states
    frontier = []
    # Explored set of state layouts
    explored = []
    # Add the initial state to the frontier
    frontier.append(state)
    enqueued_num += 1

    while frontier:
        # Treat 0 as beginning and appending to end
        parent = frontier.pop(0)
        # enqueued_num -= 1
        # add s state to explored
        explored.append(parent.state)

        # Expand the movements
        # This just gives back the layouts
        actions = expand_movements(parent)

        # for each layout/move from the expanded layouts/moves
        for layout in actions:
            # define the actual State
            child = State(layout, parent, parent.goal, movements(layout), parent.depth+1, None, None)
            # We need to check if a numpy array exists in a list of numpy arrays
            # We will do this twice for the explored and frontier sets
            in_explored = state_search_layouts(layout, explored)
            in_frontier = state_search_states(layout, frontier)
            if not(in_explored or in_frontier):
                # Check if state layout is the same as the goal node
                if state_equal(child.state, parent.goal):
                    return [child, pass_depth, enqueued_num]
                frontier.append(child)
                enqueued_num += 1
            # If depth 10 we exit at this point since no layout matches
            max_depth = child.depth
        if max_depth == 10:
            pass_depth = True
            return [child, pass_depth, enqueued_num]

# Depth Limited Search
# Recursive version
def dls(state, max_depth):
    # If current state is goal state
    if state_equal(state.state, state.goal):
        return [state, True]
    # Else if reached max_depth for iteration
    elif max_depth <= 0 :
        return [state, False]
    else:
        # Expand the movements
        # This just gives back the layouts
        actions = expand_movements(state)
        # for each layout/move from the expanded layouts/moves
        for layout in actions:
            # define the actual State
            child = State(layout, state, state.goal, movements(layout), state.depth + 1, None, None)
            result = dls(child, max_depth - 1)
            if result[1] == True:
                return result # [state, True]
    return [state, False]

# Iterative Deepening Search
# implements DLS (form of Depth First Search)
def ids(state, max_depth):
    # this initial case will not be caught by the range()
    if max_depth == 0:
        result = dls(state, 0)
        node = result[0]
        found = result[1]
        if found == True:
            return [node, found]

    # max_depth is always 10
    for depth in range(max_depth):
        result = dls(state, depth)
        node = result[0]
        found = result[1]
        if found == True:
            return [node, found]
    return [state, False]

def astar1(state):
    # h_cost calculated once we actually need to
    # !!!! make sure set lambda can actually get the right f_cost to compare
    # astar1 code
    print("astar1")
    # Create the open and closed sets
    # open_set = nodes that have been visited but not expanded yet
    # closed_set = nodes that have been visited and expanded
    open_set = set()
    closed_set = set()





def astar2():
    # astar2 code
    print("astar2")


'''
Heuristic Functions
'''
# Heuristic 1
def misplaced_tiles(state, goal_state):
    h1 = 0
    for i in range(0,3):
        for j in range(0,3):
            # Does not count where * "tile" is, only the real tiles
            if (state[i][j] != '*'):
                if(state[i][j] != goal_state[i][j]):
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
            if state[i][j] != '*':
                x1 = i
                y1 = j
                index = find_char(goal_state, state[i][j])
                x2 = index[0]
                y2 = index[1]
                h2 += abs(x2 - x1) + abs(y2 - y1)
    return h2

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
    number_of_moves = -1
    print("--------------------")
    print("Initial state\n")
    for state in goal_path:
        print_state(state)
        print()
        number_of_moves += 1
    print("Goal state")
    print("--------------------")
    print("Number of moves = ", number_of_moves)

def find_char(state, x):
    index = [0,0]
    for i in range(0,3):
        for j in range(0,3):
            if state[i][j] == x:
                index[0] = i
                index[1] = j
    return index

# List of layouts (ex. explored from bfs)
def state_search_layouts(s, layout_list):
    # for every state in state list
    for layout in layout_list:
        if state_equal(s, layout):
            return True
    return False

# List of state objects (ex. frontier from bfs)
def state_search_states(s, state_list):
    # for every state in state list
    for state in state_list:
        if state_equal(s, state.state):
            return True
    return False

def state_equal(state1, state2):
    equal = True
    for i in range(0,3):
        for j in range(0,3):
            if state1[i][j] != state2[i][j]:
                equal = False
    return equal

def print_state(state):
    for i in range(0,3):
        for j in range(0,3):
            print(state[i][j], " ", end="")
        print()

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
    initial_state = [[sys.argv[2], sys.argv[3], sys.argv[4]],
                     [sys.argv[5], sys.argv[6], sys.argv[7]],
                     [sys.argv[8], sys.argv[9], sys.argv[10]]]
    goal_state = [['1','2','3'],
                  ['4','5','6'],
                  ['7','8','*']]

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
        print("Breadth-First Search")
        # set root state
        root = State(initial_state, None, goal_state, movements(initial_state), 0, 0, 0)

        # call algorithm
        ret = bfs(root)
        found_goal = ret[0]
        hit_max_depth = ret[1]
        enqueued_num = ret[2]
        if hit_max_depth:
            print("Hit max depth of 10 without finding goal.")
        else:
            goal_path = goal_path(found_goal)
            print_goal_path(goal_path)
            print("Number of states enqueued = ", enqueued_num)

    # IDS Algorithm
    elif(chosen_algorithm == algorithms[1]):
        print("Iterative Deepening Search")
        # set root state
        root = State(initial_state, None, goal_state, movements(initial_state), 0, 0, 0)
        # call algorithm
        # max-depth is always 10
        result = ids(root, 10)
        found_goal = result[0]
        found = result[1]
        if found == True:
            goal_path = goal_path(found_goal)
            print_goal_path(goal_path)
            print("Number of states enqueued = ")
        else:
            print("Hit max depth of 10 without finding goal.")

    # Astar1 Algorithm
    elif (chosen_algorithm == algorithms[2]):
        print("A* Search with Misplaced Tiles Heuristic")
        # set root state
        root = State(initial_state, None, goal_state, movements(initial_state), 0, 0, 0)
        # call algorithm
        astar1()

    # Astar2 Algorithm
    elif (chosen_algorithm == algorithms[3]):
        print("A* Search with Manhattan Distance Heuristic")
        # set root state
        root = State(initial_state, None, goal_state, movements(initial_state), 0, 0, 0)
        # call algorithm
        astar2()

# once we get the path, go up the parent, then reverse the list
