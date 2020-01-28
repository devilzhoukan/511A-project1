# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()

class Node:
    def __init__(self, state, parent=None, path=()):
        self.state = state
        self.parent = parent
        if self.parent is None:
            self.parent = state
        self.path = path

    def next_Node(self, action, next_state):
        next_path = list(self.path)
        next_path.append(action)
        next_path = tuple(next_path)
        next_Node = Node(next_state, self.state, next_path)
        return next_Node
        

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    fringe = util.Stack()
    start_Node = Node(problem.getStartState())
    # avoid cycle in DFS
    visited = set()
    fringe.push(start_Node)

    while not fringe.isEmpty():
        current_Node = fringe.pop()

        # Check if we make it
        if problem.isGoalState(current_Node.state):
            print("Success!")
            # print(current_Node.path)
            return current_Node.path

        # Check if visited
        if current_Node.state in visited:
            continue

        # Mark as visited
        visited.add(current_Node.state)

        successors = problem.getSuccessors(current_Node.state)
        for successor in successors:
            s_state = successor[0]
            s_action = successor[1]
            if s_state == current_Node.parent or s_state in visited:
                continue
            new_Node = current_Node.next_Node(s_action, s_state)
            fringe.push(new_Node)

    print('fail')
    return[]


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    start_Node = Node(problem.getStartState())
    # avoid cycle in BFS
    visited = set()
    fringe.push(start_Node)

    while not fringe.isEmpty():
        current_Node = fringe.pop()

        # Check if we make it
        if problem.isGoalState(current_Node.state):
            print("Success!")
            print(current_Node.path)
            return current_Node.path

        # Check if visited
        if current_Node.state in visited:
            continue

        # Mark as visited
        visited.add(current_Node.state)

        successors = problem.getSuccessors(current_Node.state)
        for successor in successors:
            s_state = successor[0]
            s_action = successor[1]
            if s_state == current_Node.parent or s_state in visited:
                continue
            new_Node = current_Node.next_Node(s_action, s_state)
            fringe.push(new_Node)

    print('fail')
    return []

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
