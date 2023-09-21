# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    closed = set()  # an empty set
    fringe = util.Stack()
    succList = []
    node = (problem.getStartState(), [], 0)
    fringe.push(node)    
    while True:

        if fringe.isEmpty() == True:
            util.raiseNotDefined()
        
        checkState, actionList, totalCost = fringe.pop()     
        if problem.isGoalState(checkState):
            return actionList
        
        if  checkState not in closed:
            closed.add(checkState)
            succList = problem.getSuccessors(checkState)
                        
            for n in succList:
                
                if n[0] not in closed:  #check if it's visited or not
                    newActionList = actionList.copy()
                    newActionList.append(n[1])

                    node = (n[0], newActionList, 0)
                    
                    fringe.push(node)
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    closed = set()  # an empty set
    fringe = util.Queue()
    succList = []
    node = (problem.getStartState(), [])
    fringe.push(node)
    closed.add(problem.getStartState())
    while True:
        if fringe.isEmpty == True:
            util.raiseNotDefined()
        checkState, actionList = fringe.pop()
        # problem.getCostOfActions(action)
        if problem.isGoalState(checkState) == True:
            return actionList
        # print(problem.getSuccessors(checkState))
        succList = problem.getSuccessors(checkState)
        for n in succList:
            if n[0] not in closed:
                newActionList = actionList.copy()
                newActionList.append(n[1])
                node = (n[0], newActionList)
                fringe.push(node)
                closed.add(n[0])


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    closed = set()  # an empty set
    fringe = util.PriorityQueue()
    succList = []
    node = (problem.getStartState(), [], 0)
    fringe.push(node, 0)
    closed.add(problem.getStartState())

    while True:
        if fringe.isEmpty == True:
            util.raiseNotDefined()
        checkState, actionList, totalCost = fringe.pop()
        if problem.isGoalState(checkState) == True:
            return actionList
        succList = problem.getSuccessors(checkState)
        for n in succList:
            if n[0] not in closed:
                newActionList = actionList.copy()
                newActionList.append(n[1])
                newCost = totalCost+n[2]
                node = (n[0], newActionList, newCost)
                fringe.update(node, newCost)
                if(problem.isGoalState(n[0]) == False):
                    closed.add(n[0])


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    """_summary_
    You should see that A* finds the optimal solution slightly faster than uniform cost search 
    (about 549 vs. 620 search nodes expanded in our implementation, 
    but ties in priority may make your numbers differ slightly).
    라고 적혀있는데 Search nodes expanded가 466이 나와버림...뭔가 이상한건가...????????? 무서움;;
    Returns:
        _type_: _description_
    """
    closed = set()  # an empty set
    fringe = util.PriorityQueue()
    succList = []
    node = (problem.getStartState(), [], 0)
    fringe.push(node, 0)
    closed.add(problem.getStartState())

    while True:
        if fringe.isEmpty == True:
            util.raiseNotDefined()
        checkState, actionList, totalCost = fringe.pop()
        if problem.isGoalState(checkState) == True:
            return actionList
        succList = problem.getSuccessors(checkState)
        for n in succList:
            if n[0] not in closed:
                hNum = heuristic(n[0], problem) + n[2] + totalCost
                newActionList = actionList.copy()
                newActionList.append(n[1])
                newCost = n[2] + totalCost
                node = (n[0], newActionList, newCost)
                fringe.update(node, hNum)
                if (problem.isGoalState(n[0]) == False):
                    closed.add(n[0])

def TreeSearch(problem, fringe):
    # insert(MAKE-NODE(INITIAL-STATE[problem]), fringe)

    pass


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch