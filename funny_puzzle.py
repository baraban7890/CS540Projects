import heapq as hq
import numpy as np


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]): # I think this is good
    distance = 0
    to_mat = np.array(to_state)
    from_mat = np.array(from_state)
    to_mat = to_mat.reshape(3,3)
    from_mat = from_mat.reshape(3,3)
    value = 1
    for i in range(7):
        ydiff = abs(np.where(from_mat == value)[0] - np.where(to_mat == value)[0])
        xdiff = abs(np.where(from_mat == value)[1] - np.where(to_mat == value)[1])
        distance += int(xdiff) + int(ydiff)
        value += 1
        
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    return distance




def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    #Either 3,4, or 5 valid states depending on where it is
    succ_states = []
    state_mat = np.array(state)
    state_mat = state_mat.reshape(3,3)
    indices = np.where(state_mat==0)
    index1 = indices[0] #row list
    index2 = indices[1] #col list
    if index1[0]-1 >= 0 and state_mat[index1[0]-1][index2[0]] != 0: #if row-1 is > 0 and element to left isn't 0
        newmat = np.array(state_mat)
        newmat[index1[0]-1][index2[0]] = 0
        newmat[index1[0]][index2[0]] = state_mat[index1[0]-1][index2[0]]
        succ_states.append(list(newmat.flatten()))
    if index1[0]+1 <= 2 and state_mat[index1[0]+1][index2[0]] != 0:
        newmat = np.array(state_mat)
        newmat[index1[0]+1][index2[0]] = 0
        newmat[index1[0]][index2[0]] = state_mat[index1[0]+1][index2[0]]
        succ_states.append(list(newmat.flatten()))
    if index2[0]-1 >=0 and state_mat[index1[0]][index2[0]-1] != 0:
        newmat = np.array(state_mat)
        newmat[index1[0]][index2[0]-1] = 0
        newmat[index1[0]][index2[0]] = state_mat[index1[0]][index2[0]-1]
        succ_states.append(list(newmat.flatten()))
    if index2[0]+1 <=2 and state_mat[index1[0]][index2[0]+1] != 0:
        newmat = np.array(state_mat)
        newmat[index1[0]][index2[0]+1] = 0
        newmat[index1[0]][index2[0]] = state_mat[index1[0]][index2[0]+1]
        succ_states.append(list(newmat.flatten()))
        
        
    if index1[1]-1 >= 0 and state_mat[index1[1]-1][index2[1]] != 0: #if row-1 is > 0 and element to left isn't 0
        newmat = np.array(state_mat)
        newmat[index1[1]-1][index2[1]] = 0
        newmat[index1[1]][index2[1]] = state_mat[index1[1]-1][index2[1]]
        succ_states.append(list(newmat.flatten()))
    if index1[1]+1 <= 2 and state_mat[index1[1]+1][index2[1]] != 0:
        newmat = np.array(state_mat)
        newmat[index1[1]+1][index2[1]] = 0
        newmat[index1[1]][index2[1]] = state_mat[index1[1]+1][index2[1]]
        succ_states.append(list(newmat.flatten()))
    if index2[1]-1 >=0 and state_mat[index1[1]][index2[1]-1] != 0:
        newmat = np.array(state_mat)
        newmat[index1[1]][index2[1]-1] = 0
        newmat[index1[1]][index2[1]] = state_mat[index1[1]][index2[1]-1]
        succ_states.append(list(newmat.flatten()))
    if index2[1]+1 <=2 and state_mat[index1[1]][index2[1]+1] != 0:
        newmat = np.array(state_mat)
        newmat[index1[1]][index2[1]+1] = 0
        newmat[index1[1]][index2[1]] = state_mat[index1[1]][index2[1]+1]
        succ_states.append(list(newmat.flatten()))
            
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    hnew = []
    tracking = []
    hq.heappush(hnew,(get_manhattan_distance(state),state,(0,get_manhattan_distance(state),-1)))
    output = []
    oldlist = []
    maxqueuelength = 0
    while len(hnew) != 0:
        maxqueuelength = max(len(hnew),maxqueuelength)
        parent = hq.heappop(hnew)
        oldlist.append(parent[1])
        tracking.append(parent)
        if get_manhattan_distance(parent[1]) == 0:
            index = len(tracking)-1
            moves = 0
            while index != 0:
                output.append(tracking[index])
                index = tracking[index][2][2]
            output.append(tracking[0])
            for i in reversed(range(len(output))):
                print(str(output[i][1]) + " h=" + str(output[i][2][1]) + " moves: " + str(moves))
                moves += 1
            print("Max queue length: " + str(maxqueuelength))
            return
        successors = get_succ(parent[1]) #state
        for s in successors:
            s = [int(i) for i in s]
            if s not in oldlist:
                d = get_manhattan_distance(s)
                hq.heappush(hnew,(parent[2][0]+d+1,s,(parent[2][0]+1,d,tracking.index(parent))))
            else:
                pass
        maxqueuelength = max(len(hnew),maxqueuelength)
        
if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    #print_succ([2,5,1,4,0,6,7,0,3])
    #print()

    #print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    #print()

    #print()
