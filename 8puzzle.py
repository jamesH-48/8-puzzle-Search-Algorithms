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
    def __init__(self, state, parent, goal, moves, depth, g_cost, h_cost, f_cost):
        self.state = state
        self.parent = parent
        self.goal = goal
        self.moves = moves
        self.depth = depth
        self.g_cost = g_cost
        self.f_cost = f_cost
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
        # add s state to explored
        explored.append(parent.state)

        # Expand the movements
        # This just gives back the layouts
        actions = expand_movements(parent)

        # for each layout/move from the expanded layouts/moves
        for layout in actions:
            # Check if next depth is past 10
            max_depth = parent.depth + 1
            # If depth is past 10 then return
            if max_depth == 11:
                pass_depth = True
                return [None, pass_depth, enqueued_num]
            # define the actual State
            child = State(layout, parent, parent.goal, movements(layout), parent.depth+1, None, None, None)
            # We need to check if an array exists in a list of arrays
            # We will do this twice for the explored and frontier sets
            in_explored = state_search_layouts(layout, explored)
            in_frontier = state_search_states(layout, frontier)
            if not(in_explored or in_frontier):
                # Check if state layout is the same as the goal node
                if state_equal(child.state, parent.goal):
                    return [child, pass_depth, enqueued_num]
                frontier.append(child)
                enqueued_num += 1

# Depth Limited Search
# Recursive version
def dls(state, max_depth, explored):
    # denote as explored
    explored.append(state.state)
    # call global variable to track states
    global enq_ids
    enq_ids += 1
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
            child = State(layout, state, state.goal, movements(layout), state.depth + 1, None, None, None)
            # check if visited/explored before
            in_explored = state_search_layouts(layout, explored)
            if not in_explored:
                # Recurse through
                result = dls(child, max_depth - 1, explored)
                if result[1] == True:
                    return result # [state, True]
    return [state, False]

# Iterative Deepening Search
# implements DLS (form of Depth First Search)
# Global value for enqueued for Iterative Deepening Search only
enq_ids = 0
def ids(state, max_depth):
    # this initial case will not be caught by the range()
    if max_depth == 0:
        # enqueued once
        global enq_ids
        enq_ids = 1
        result = dls(state, 0)
        node = result[0]
        found = result[1]
        if found == True:
            return [node, found, enq_ids]

    # max_depth is always 10
    # + 1 to the depth since the call needs to check nodes at depth 10 not just before it
    for depth in range(max_depth + 1):
        # reset this for each iteration
        explored = []
        # Set enqueued value to 0 for each iteration
        # Only if you want to see the states seen for the iteration that found the goal state
        # enq_ids = 0
        result = dls(state, depth, explored)
        node = result[0]
        found = result[1]
        if found == True:
            return [node, found, enq_ids]
    return [state, False, 0]

# A* Algorithm w/ Misplaced Tiles Heuristic
def astar1(state):
    # Define Max Depth value
    max_depth = 0
    pass_depth = False
    # Total number of states ever enqueued
    enqueued_num = 0

    # Create the open and closed sets
    # open_set = nodes that have been visited but not expanded yet
    # closed_set = nodes that have been visited and expanded
    open_set = set()
    open_set.add(state)
    enqueued_num += 1
    closed_set = set()

    #print_state(state.state)

    while open_set:
        # Get minimum f-cost state from open set
        curState = min(open_set, key=lambda state: state.f_cost)

        # Print out for understanding choices for algorithm
        # cont = input("Parent Cont: ")
        # print_state(curState.state)
        # print("f_cost:",curState.f_cost)
        # print("h_cost:", curState.h_cost)

        # Found goal
        if state_equal(curState.state, curState.goal):
            return [curState, pass_depth, enqueued_num]
        # Remove from open set
        open_set.remove(curState)
        # Add to closed set
        closed_set.add(curState)

        # Expand the movements
        # This just gives back the layouts
        actions = expand_movements(curState)
        # for each layout/move from the expanded layouts/moves
        for layout in actions:
            # If past depth 10 we exit at this point since no layout matches
            max_depth = curState.depth + 1
            if max_depth == 11:
                pass_depth = True
                return [child, pass_depth, enqueued_num]

            child = State(layout, curState, curState.goal, movements(layout), curState.depth + 1, None, None, None)

            # Print out for understanding choices for algorithm
            # cont = input("Children Cont: ")
            # print_state(child.state)

            # Temporary g_cost
            # Use this for comparing two of the same states since the h_costs will be the same
            # The cost between states is always 1
            temp_gcost = curState.g_cost + 1

            # set_node = node within set that matches layout
            # layout_found = if we found a node within the set that matches the layout
            open_set_node, open_layout_found = state_search_sets(layout, open_set)
            closed_set_node, closed_layout_found = state_search_sets(layout, closed_set)
            if open_layout_found:
                # If the node found has a better g_cost then skip over this current one
                if open_set_node.g_cost <= temp_gcost:
                    continue
                else:
                    # Take away the old node in the open set that isn't as optimal
                    # And replace it with the new more optimal node found
                    open_set.remove(open_set_node)
                    open_set.add(child)
            elif closed_layout_found:
                # If the node found has a better g_cost then skip over this current child
                if closed_set_node.g_cost <= temp_gcost:
                    continue
                # This implies that the new found node with the same layout has a better g-cost
                closed_set.remove(closed_set_node)
                open_set.add(child)
            else:
                # If not found in any set, add to open_set
                # Calculate h_cost
                # If already found h_cost would already be set
                # The unique layout will always be in either the closed or open set
                open_set.add(child)
                enqueued_num += 1
                child.h_cost = misplaced_tiles(layout, curState.goal)

            # Before we go to the next child/successor
            # Update g_cost
            child.g_cost = temp_gcost
            # Update/Set f_cost
            child.f_cost = child.g_cost + child.h_cost

# A* Algorithm w/ Manhattan Distance Heuristic
def astar2(state):
    # Define Max Depth value
    max_depth = 0
    pass_depth = False
    # Total number of states ever enqueued
    enqueued_num = 0

    # Create the open and closed sets
    # open_set = nodes that have been visited but not expanded yet
    # closed_set = nodes that have been visited and expanded
    open_set = set()
    open_set.add(state)
    enqueued_num += 1
    closed_set = set()

    #print_state(state.state)

    while open_set:
        # Get minimum f-cost state from open set
        curState = min(open_set, key=lambda state: state.f_cost)

        # Print out for understanding choices for algorithm
        # cont = input("Parent Cont: ")
        # print_state(curState.state)
        # print("f_cost:",curState.f_cost)
        # print("h_cost:", curState.h_cost)

        # Found goal
        if state_equal(curState.state, curState.goal):
            return [curState, pass_depth, enqueued_num]
        # Remove from open set
        open_set.remove(curState)
        # Add to closed set
        closed_set.add(curState)

        # Expand the movements
        # This just gives back the layouts
        actions = expand_movements(curState)
        # for each layout/move from the expanded layouts/moves
        for layout in actions:
            # If past depth 10 we exit at this point since no layout matches
            max_depth = curState.depth + 1
            if max_depth == 11:
                pass_depth = True
                return [child, pass_depth, enqueued_num]

            # define the actual State
            child = State(layout, curState, curState.goal, movements(layout), curState.depth + 1, None, None, None)
            # If depth 10 we exit at this point since no layout matches
            max_depth = child.depth

            # Print out for understanding choices for algorithm
            # cont = input("Children Cont: ")
            # print_state(child.state)

            # Temporary g_cost
            # Use this for comparing two of the same states since the h_costs will be the same
            # The cost between states is always 1
            temp_gcost = curState.g_cost + 1

            # set_node = node within set that matches layout
            # layout_found = if we found a node within the set that matches the layout
            open_set_node, open_layout_found = state_search_sets(layout, open_set)
            closed_set_node, closed_layout_found = state_search_sets(layout, closed_set)
            if open_layout_found:
                # If the node found has a better g_cost then skip over this current one
                if open_set_node.g_cost <= temp_gcost:
                    continue
                else:
                    # Take away the old node in the open set that isn't as optimal
                    # And replace it with the new more optimal node found
                    open_set.remove(open_set_node)
                    open_set.add(child)
            elif closed_layout_found:
                # If the node found has a better g_cost then skip over this current child
                if closed_set_node.g_cost <= temp_gcost:
                    continue
                # This implies that the new found node with the same layout has a better g-cost
                closed_set.remove(closed_set_node)
                open_set.add(child)
            else:
                # If not found in any set, add to open_set
                # Calculate h_cost
                # If already found h_cost would already be set
                # The unique layout will always be in either the closed or open set
                open_set.add(child)
                enqueued_num += 1
                child.h_cost = manhattan_distance(layout, curState.goal)

            # Before we go to the next child/successor
            # Update g_cost
            child.g_cost = temp_gcost
            # Update/Set f_cost
            child.f_cost = child.g_cost + child.h_cost


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

# Searches for a state within a set (ex. used in A*)
def state_search_sets(layout, set):
    n = None
    # For each node in the set
    for node in set:
        equal = True
        # Check if the layout matches any node layout within the set
        for i in range(0, 3):
            for j in range(0, 3):
                if layout[i][j] != node.state[i][j]:
                    equal = False
        # We found the proper node from the set that matches the layout so return
        if equal == True:
            return node, True
    return n, False

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
    # Entry print
    print("8 Puzzle")
    print("~~~~~~~~~~~~~~~~~~~~")
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
        root = State(initial_state, None, goal_state, movements(initial_state), 0, 0, 0, 0)

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
        root = State(initial_state, None, goal_state, movements(initial_state), 0, 0, 0, 0)
        # call algorithm
        # max-depth is always 10
        result = ids(root, 10)
        found_goal = result[0]
        found = result[1]
        enqueued = result[2]
        if found:
            goal_path = goal_path(found_goal)
            print_goal_path(goal_path)
            print("Number of states enqueued = ", enqueued)
        else:
            print("Hit max depth of 10 without finding goal.")

    # Astar1 Algorithm
    elif (chosen_algorithm == algorithms[2]):
        print("A* Search with Misplaced Tiles Heuristic")
        # set root state
        root_h_cost = misplaced_tiles(initial_state, goal_state)
        # print("Root Cost",root_h_cost)
        root = State(initial_state, None, goal_state, movements(initial_state), 0, 0, root_h_cost, root_h_cost)
        # call algorithm
        ret = astar1(root)
        found_goal = ret[0]
        hit_max_depth = ret[1]
        enqueued_num = ret[2]
        if hit_max_depth:
            print("Hit max depth of 10 without finding goal.")
        else:
            goal_path = goal_path(found_goal)
            print_goal_path(goal_path)
            print("Number of states enqueued = ", enqueued_num)

    # Astar2 Algorithm
    elif (chosen_algorithm == algorithms[3]):
        print("A* Search with Manhattan Distance Heuristic")
        # set root state
        root_h_cost = manhattan_distance(initial_state, goal_state)
        # print("Root Cost", root_h_cost)
        root = State(initial_state, None, goal_state, movements(initial_state), 0, 0, root_h_cost, root_h_cost)
        # call algorithm
        ret = astar2(root)
        found_goal = ret[0]
        hit_max_depth = ret[1]
        enqueued_num = ret[2]
        if hit_max_depth:
            print("Hit max depth of 10 without finding goal.")
        else:
            goal_path = goal_path(found_goal)
            print_goal_path(goal_path)
            print("Number of states enqueued = ", enqueued_num)
