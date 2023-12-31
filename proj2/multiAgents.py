# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        #1. higher number better.
        #2. if we can kill ghost, score up
        #3. As features, try the reciprocal of important values (such as distance to food) 
        #   rather than just the values themselves.

        foodDistance = 0
        ghostDistance = 0
        score = successorGameState.getScore()
        minFood = 999999
        minGhost = 999999

        for food in newFood.asList():
            #   calculate distance from current position to food
            foodDistance = manhattanDistance(newPos,food)
            #   find minimum distance of food
            if foodDistance < minFood:
                minFood = foodDistance
        
        for ghost in newGhostStates:
            ghostDistance = manhattanDistance(newPos, ghost.getPosition())
            
            # let's give condition that if pacman is close with ghost,
            # it's not good

            if ghostDistance < 2 and ghost.scaredTimer == 0:
                # print("too close")
                return -99999
            # elif ghost.scaredTimer > 0 and ghostDistance < 3:
            #     # print("trying to eat")
            #     scroe += ghostDistance
        
            if ghostDistance < minGhost:
                minGhost = ghostDistance
        #   if food is too far away, it's not good option.
        score += 1.0/ minFood
        score -= 1.0/ ghostDistance
        
        return score
     

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1ew

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        value = float('-inf')
        action = Directions.LEFT
        for ac in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, ac)
            miniValue = self.minimax(successor, 0, False, 1)
            #print(miniValue)
            if miniValue > value:
                value = miniValue
                action = ac
        #print("Final value: ", value)
        return action
        
    def minimax(self, gameState, depth, maxPlayer, ghostnum):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            #print("Current depth is: ", depth)
            return self.evaluationFunction(gameState)
        
        if maxPlayer:
            value = float('-inf')
            for ac in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, ac)
                value = max(value, self.minimax(successor, depth, False, 1))
            #print("Max value: ", value)
            #print("Current depth is: ", depth)
            return value
        else:
            value = float('inf')
            if ghostnum+1 == gameState.getNumAgents():
                for ac in gameState.getLegalActions(ghostnum):
                    successor = gameState.generateSuccessor(ghostnum, ac)
                    value = min(value, self.minimax(successor, depth+1, True, 0))
                #print("Min value Max next = ", value)
            else:
                for ac in gameState.getLegalActions(ghostnum):
                    successor = gameState.generateSuccessor(ghostnum, ac)
                    value = min(value, self.minimax(successor, depth, False, ghostnum+1))
            #print("Min value = ", value)
            return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value = float('-inf')
        action = Directions.LEFT
        alpha = float('-inf')
        beta = float('inf')
        for ac in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, ac)
            miniValue = self.minimaxAlphaBeta(successor, 0, False, 1, alpha, beta)
            # print(miniValue)
            if miniValue > value:
                value = miniValue
                action = ac
            if value > beta:
                return value
            alpha = max(alpha, value)
        # print("Final value: ", value)
        return action

    def minimaxAlphaBeta(self, gameState, depth, maxPlayer, ghostnum, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            # print("Current depth is: ", depth)
            return self.evaluationFunction(gameState)

        if maxPlayer:
            value = float('-inf')
            for ac in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, ac)
                value = max(value, self.minimaxAlphaBeta(successor, depth, False, 1, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            # print("Max value: ", value)
            # print("Current depth is: ", depth)
            return value
        else:
            value = float('inf')
            if ghostnum+1 == gameState.getNumAgents():
                for ac in gameState.getLegalActions(ghostnum):
                    successor = gameState.generateSuccessor(ghostnum, ac)
                    value = min(value, self.minimaxAlphaBeta(successor, depth+1, True, 0, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                # print("Min value Max next = ", value)
            else:
                for ac in gameState.getLegalActions(ghostnum):
                    successor = gameState.generateSuccessor(ghostnum, ac)
                    value = min(value, self.minimaxAlphaBeta(successor, depth, False, ghostnum+1, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
            # print("Min value = ", value)
            return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        value = float('-inf')
        action = Directions.LEFT
        for ac in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, ac)
            miniValue = self.expectimax(successor, 0, False, 1)
            # print(miniValue)
            if miniValue > value:
                value = miniValue
                action = ac
        # print("Final value: ", value)
        return action

    def expectimax(self, gameState, depth, maxPlayer, ghostnum):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            # print("Current depth is: ", depth)
            return self.evaluationFunction(gameState)

        if maxPlayer:
            value = float('-inf')
            for ac in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, ac)
                value = max(value, self.expectimax(successor, depth, False, 1))
            # print("Max value: ", value)
            # print("Current depth is: ", depth)
            return value
        else:
            value = 0
            count = 0
            if ghostnum+1 == gameState.getNumAgents():
                for ac in gameState.getLegalActions(ghostnum):
                    successor = gameState.generateSuccessor(ghostnum, ac)
                    value += self.expectimax(successor, depth+1, True, 0)
                    count += 1
                # print("Min value Max next = ", value)
            else:
                for ac in gameState.getLegalActions(ghostnum):
                    successor = gameState.generateSuccessor(ghostnum, ac)
                    value += self.expectimax(successor, depth, False, ghostnum+1)
                    count += 1
            # print("Min value = ", value)
            return value / count
        
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    1. Food distance
    2. Ghost distance
    3. higher number better.
    4. if we can kill ghost, score up
    5. As features, try the reciprocal of important values (such as distance to food)
        rather than just the values themselves.
    6. Give priority weight for every features to calculate better score
    
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhost = currentGameState.getGhostStates()
    currentCapsule = currentGameState.getCapsules()
    currentGameScore = currentGameState.getScore()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    

    foodDistance = 0
    ghostDistance = 0
    capsuleDistance = 0
    score = 0
    minFood = 999999
    minGhost = 999999
    minCapsule = 999999

    for food in currentFood.asList():
        #   calculate distance from current position to food
        foodDistance = manhattanDistance(currentPos, food)
        #   find minimum distance of food
        if foodDistance < minFood:
            minFood = foodDistance

    for ghost in currentGhost:
        ghostDistance = manhattanDistance(currentPos, ghost.getPosition())

        # let's give condition that if pacman is close with ghost,
        # it's not good

        # if ghostDistance < 2 and ghost.scaredTimer == 0:
        #     # print("too close")
        #     return -99999
        # # elif ghost.scaredTimer > 0 and ghostDistance < 3:
        # #     # print("trying to eat")
        # #     scroe += ghostDistance
        # if ghost.scaredTimer > 0 and ghostDistance < 2:
        #     score += 1

        if ghost.scaredTimer == 0 and ghostDistance < 2:
            score -= 100
            return score
        elif ghost.scaredTimer > 0:
            score += 100 / ghostDistance

    for capsule in currentCapsule:
        capsuleDistance = manhattanDistance(currentPos, capsule)
        
    #   if food is too far away, it's not good option.
    score += 1 / minFood * 10
    score -= 1 / ghostDistance * 40
    if capsuleDistance != 0:
        score += 1 / capsuleDistance * 20
    score += currentGameScore * 100
      
    return score

# Abbreviation
better = betterEvaluationFunction
