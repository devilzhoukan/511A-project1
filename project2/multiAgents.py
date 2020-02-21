# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util, math

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        newPos = successorGameState.getPacmanPosition()  # tuple, (x, y)
        newFood = successorGameState.getFood()  # game.Grid
        newGhostStates = successorGameState.getGhostStates()  # game.AgentState
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]  # list [int]

        "*** YOUR CODE HERE ***"
        foodScore, pelletScore, ghostScore = 100.0, 160.0, -500.0
        killScore = 1000
        discount = 0.25
        stopPenalty = -40.0
        score = 0.0
        safeTime = 5.0

        # if new action get food, add food score
        score += currentGameState.hasFood(newPos[0], newPos[1]) * foodScore

        foodList = newFood.asList()

        # f(x) = exp(-x)
        # x increase, f(x) decrease fast
        for food in foodList:
            score += foodScore * (math.exp(-1.0 * discount * manhattanDistance(newPos, food)) - 1)

        # it always stays in 2 position
        # dicount solves part of the problem
        # Min ManDis is not a good idea
        
        # if new action get pellet, add pellet score
        curCapsuleList = currentGameState.data.capsules
        score += (newPos in curCapsuleList) * pelletScore

        # let pellet attract pacman
        for capsule in curCapsuleList:
            score += pelletScore * math.exp(-1.0 * discount * manhattanDistance(newPos, capsule))

        ghostPosList = []
        for state, lastTime in zip(newGhostStates, newScaredTimes):
            ghostPosList.append(state.getPosition())

        score += (newPos in ghostPosList) * ghostScore
        for lastTime, ghostPos in zip(newScaredTimes, ghostPosList):
            if lastTime < safeTime:
                score += ghostScore * math.exp(-1.0 * discount * manhattanDistance(newPos, ghostPos))
                score += (manhattanDistance(newPos, ghostPos) < 2) * ghostScore

        if action == Directions.STOP:
            score += stopPenalty

        for lastTime, ghostPos in zip(newScaredTimes, ghostPosList):
            # "kill here"
            if lastTime > safeTime:
                score += (manhattanDistance(newPos, ghostPos) < 1) * killScore
            if lastTime > safeTime * 2:
                score += killScore * math.exp(-1.0 * discount * manhattanDistance(newPos, ghostPos))


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
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        if self.depth == 0 or gameState.isWin() or gameState.isLose():
            return Directions.STOP
        actions = gameState.getLegalPacmanActions()
        ghostNum = gameState.getNumAgents() - 1
        maxScore = float('-inf')
        rAction = Directions.STOP
        for action in actions:
            successorGameState = gameState.generatePacmanSuccessor(action)
            if successorGameState.isWin():
                return action
            score = self.actGhost(successorGameState, ghostNum, 1)
            if score > maxScore:
                maxScore = score
                rAction = action
        return rAction

    def actGhost(self, gameState, ghostNum, depth):
        # Act a ghost
        # Min value
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(ghostNum)
        if not actions:
            return self.evaluationFunction(gameState)
        successorGameStates = [gameState.generateSuccessor(ghostNum, action) for action in actions]
        ghostNum -= 1
        if ghostNum == 0:
            if depth == self.depth:
                scores = [self.evaluationFunction(state) for state in successorGameStates]
            else:
                scores = [self.actPacman(state, depth) for state in successorGameStates]
        else:
            scores = [self.actGhost(state, ghostNum, depth) for state in successorGameStates]
        return min(scores)

    def actPacman(self, gameState, depth):
        # Max value
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalPacmanActions()
        successorGameStates = [gameState.generatePacmanSuccessor(action) for action in actions]
        ghostNum = gameState.getNumAgents() - 1
        scores = [self.actGhost(state, ghostNum, depth + 1) for state in successorGameStates]
        return max(scores)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        if self.depth == 0 or gameState.isWin() or gameState.isLose():
            return Directions.STOP
        actions = gameState.getLegalPacmanActions()
        ghostNum = gameState.getNumAgents() - 1
        AlphaScore = float('-inf')
        rAction = Directions.STOP
        for action in actions:
            successorGameState = gameState.generatePacmanSuccessor(action)
            if successorGameState.isWin():
                return action
            score = self.actGhost(successorGameState, AlphaScore, float('inf'), ghostNum, 1)
            if score > AlphaScore:
                AlphaScore = score
                rAction = action
        return rAction

    def actGhost(self, gameState, alpha, beta, ghostNum, depth):
        # Act a ghost
        # Min value
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(ghostNum)
        if not actions:
            return self.evaluationFunction(gameState)
        result = float('inf')
        for action in actions:
            successorGameState = gameState.generateSuccessor(ghostNum, action)
            if ghostNum == 0:
                if depth == self.depth:
                    score = self.evaluationFunction(successorGameState)
                else:
                    score = self.actPacman(successorGameState, alpha, beta, depth)
            else:
                score = self.actGhost(successorGameState, alpha, beta, ghostNum - 1, depth)
            result = min(result, score)
            if result < alpha:
                return result
            beta = min(beta, result)
        return result

    def actPacman(self, gameState, alpha, beta, depth):
        # Max value
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalPacmanActions()
        ghostNum = gameState.getNumAgents() - 1
        result = float('-inf')
        for action in actions:
            successorGameState = gameState.generatePacmanSuccessor(action)
            score = self.actGhost(successorGameState, alpha, beta, ghostNum, depth + 1)
            result = max(result, score)
            if result > beta:
                return result
            alpha = max(result, alpha)
        return result

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
        if self.depth == 0 or gameState.isWin() or gameState.isLose():
            return Directions.STOP
        actions = gameState.getLegalPacmanActions()
        ghostNum = gameState.getNumAgents() - 1
        maxScore = float('-inf')
        rAction = Directions.STOP
        for action in actions:
            successorGameState = gameState.generatePacmanSuccessor(action)
            if successorGameState.isWin():
                return action
            score = self.actGhost(successorGameState, ghostNum, 1)
            if score > maxScore:
                maxScore = score
                rAction = action
        return rAction

    def actGhost(self, gameState, ghostNum, depth):
        # Act a ghost
        # Expect value
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(ghostNum)
        if not actions:
            return self.evaluationFunction(gameState)
        successorGameStates = [gameState.generateSuccessor(ghostNum, action) for action in actions]
        ghostNum -= 1
        if ghostNum == 0:
            if depth == self.depth:
                scores = [self.evaluationFunction(state) for state in successorGameStates]
            else:
                scores = [self.actPacman(state, depth) for state in successorGameStates]
        else:
            scores = [self.actGhost(state, ghostNum, depth) for state in successorGameStates]
        return sum(scores) / len(scores)

    def actPacman(self, gameState, depth):
        # Max value
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalPacmanActions()
        successorGameStates = [gameState.generatePacmanSuccessor(action) for action in actions]
        ghostNum = gameState.getNumAgents() - 1
        scores = [self.actGhost(state, ghostNum, depth + 1) for state in successorGameStates]
        return max(scores)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foodScore, pelletScore, ghostScore = 100.0, 160.0, -500.0
    discount = 0.2
    score = 0.0
    killScore = 1000
    safeTime = 10.0

    foodList = currentGameState.getFood().asList()
    pos = currentGameState.getPacmanPosition()

    for food in foodList:
        score += foodScore * (math.exp(-1.0 * discount * manhattanDistance(pos, food)) - 2)

    # pellet
    capsuleList = currentGameState.data.capsules
    for capsule in capsuleList:
        score += pelletScore * math.exp(-1.0 * discount * manhattanDistance(pos, capsule))

    ghostPosList = []

    curGhostStates = currentGameState.getGhostStates()  # game.AgentState
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]  # list [int]
    for state, lastTime in zip(curGhostStates, curScaredTimes):
        ghostPosList.append(state.getPosition())

    for lastTime, ghostPos in zip(curScaredTimes, ghostPosList):
        if lastTime < safeTime:
            score += ghostScore * math.exp(-1.0 * discount * manhattanDistance(pos, ghostPos))
            score += (manhattanDistance(pos, ghostPos) < 2) * (-100000)

    for lastTime, ghostPos in zip(curScaredTimes, ghostPosList):
        # "kill here"
        if lastTime > safeTime:
            score += (manhattanDistance(pos, ghostPos) < 1) * killScore
        if lastTime > safeTime * 2:
            score += killScore * math.exp(-1.0 * discount * manhattanDistance(pos, ghostPos))

    return score


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

