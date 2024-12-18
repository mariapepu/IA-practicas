from itertools import permutations
import math
import numpy as np

import chess

class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game

    Methods:
    --------
    startGame(pos:stup) -> None

    """

    UP = np.array((-1, 0))
    DOWN = np.array((1, 0))
    LEFT = np.array((0, -1))
    RIGHT = np.array((0, 1))

    def __init__(self, TA, myinit=True):
        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.listVisitedSituations = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW
        self.depthMax = 8
        self.checkMate = False

    def copyState(self, state):
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState

    def isVisitedSituation(self, color, myState):
        if (len(self.listVisitedSituations) > 0):
            perm_state = list(permutations(myState))

            isVisited = False
            for j in range(len(perm_state)):
                for k in range(len(self.listVisitedSituations)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedSituations.__getitem__(k)[1]) and color == \
                            self.listVisitedSituations.__getitem__(k)[0]:
                        isVisited = True
            return isVisited
        else:
            return False

    # DONE (DOESN'T WORK!!)
    def getCurrentColorState(self, white):
        return (self.myCurrentStateW if white else self.myCurrentStateB)

    # GIVEN (DOESN'T WORK!!)
    def getCurrentStateW(self):
        return self.myCurrentStateW

    # GIVEN (DOESN'T WORK!!)
    def getCurrentStateB(self):
        return self.myCurrentStateB

    # DONE
    def getListNextPieceStates(self, myState, white):
        return self.getListNextStatesW(myState) if white else self.getListNextStatesB(myState)

    def getListNextStatesW(self, myState):
        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()
        return self.listNextStates

    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()
        return self.listNextStates

    def isSameState(self, a, b):
        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):
            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):
            if b[k] not in a:
                isSameState2 = False

        return isSameState1 and isSameState2

    def isVisited(self, myState):
        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(myState))

            isVisited = False
            for j in range(len(perm_state)):
                for k in range(len(self.listVisitedStates)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True
            return isVisited
        else:
            return False

    # DONE
    def isCheck(self, currentState, white):
        return self.isWatchedWk(currentState) if white else self.isWatchedBk(currentState)

    def isWatchedBk(self, currentState):
        self.newBoardSim(currentState)
        bkPosition = self.getPieceState(currentState, 12)[0:2]
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)

        # Si les negres maten el rei blanc, no és una configuració correcta
        if wkState == None:
            return False
        # Mirem les possibles posicions del rei blanc i mirem si en alguna pot "matar" al rei negre
        for wkPosition in self.getNextPositions(wkState):
            if bkPosition == wkPosition:
                # Tindríem un checkMate
                return True
        if wrState != None:
            # Mirem les possibles posicions de la torre blanca i mirem si en alguna pot "matar" al rei negre
            for wrPosition in self.getNextPositions(wrState):
                if bkPosition == wrPosition:
                    return True
        return False

    def isWatchedWk(self, currentState):
        self.newBoardSim(currentState)
        wkPosition = self.getPieceState(currentState, 6)[0:2]
        bkState = self.getPieceState(currentState, 12)
        brState = self.getPieceState(currentState, 8)

        # If whites kill the black king , it is not a correct configuration
        if bkState == None:
            return False
        # We check all possible positions for the black king, and chck if in any of them it may kill the white king
        for bkPosition in self.getNextPositions(bkState):
            if wkPosition == bkPosition:
                # That would be checkMate
                return True
        if brState != None:
            # We check the possible positions of the black tower, and we check if in any of them it may kill the white king
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True
        return False

    # DONE
    def getListNextColorStates(self, currentState, white):
        self.listNextStates = []
        for piece in self.getColorState(currentState, white):
            self.chess.boardSim.getListNextStatesW([piece]) if white else self.chess.boardSim.getListNextStatesB([piece])
            for stateChange in self.chess.boardSim.listNextStates.copy():
                replace = lambda state: stateChange[0] if piece == state else state
                nextState = list(map(replace, currentState))

                #Check for eaten pieces
                for whitePiece in nextState:
                    for blackPiece in nextState:
                        if whitePiece != blackPiece and whitePiece[:2] == blackPiece[:2]:
                            nextState.remove(blackPiece) if white else nextState.remove(whitePiece)

                self.listNextStates.append(nextState)
        return self.listNextStates

    # DONE
    def isCheckMate(self, state, white):
        if self.isCheck(state, white) and not self.isCheck(state, not white):
            for nextState in self.getListNextColorStates(state, white):
                if not self.isCheck(nextState, white):
                    return False
            return True
        return False

    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]
        self.chess.newBoardSim(TA)

    def getPieceState(self, state, piece):
        for i in state:
            if i[2] == piece:
                return i
        return None

    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates

    def getCurrentStateSim(self):
        listStates = []
        for i in self.chess.boardSim.currentStateW:
            listStates.append(i)
        for j in self.chess.boardSim.currentStateB:
            listStates.append(j)
        return listStates

    def getNextPositions(self, state):
        # Given a state, we check the next possible states
        # From these, we return a list with position, i.e., [row, column]
        if state == None:
            return None
        if state[2] > 6:
            nextStates = self.getListNextStatesB([state])
        else:
            nextStates = self.getListNextStatesW([state])
        nextPositions = []
        for i in nextStates:
            nextPositions.append(i[0][0:2])
        return nextPositions

    # DONE
    def getColorState(self, currentState, white):
        return self.getWhiteState(currentState) if white else self.getBlackState(currentState)

    def getWhiteState(self, currentState):
        whiteState = []
        wkState = self.getPieceState(currentState, 6)
        whiteState.append(wkState)
        wrState = self.getPieceState(currentState, 2)
        if wrState != None:
            whiteState.append(wrState)
        return whiteState

    def getBlackState(self, currentState):
        blackState = []
        bkState = self.getPieceState(currentState, 12)
        blackState.append(bkState)
        brState = self.getPieceState(currentState, 8)
        if brState != None:
            blackState.append(brState)
        return blackState

    def getMovement(self, state, nextState):
        # Given a state and a successor state, return the position of the piece that has been moved in both states
        pieceState = None
        pieceNextState = None
        for piece in state:
            if piece not in nextState:
                movedPiece = piece[2]
                pieceNext = self.getPieceState(nextState, movedPiece)
                if pieceNext != None:
                    pieceState = piece
                    pieceNextState = pieceNext
                    break
        return [pieceState, pieceNextState]

    def reconstruct_path(self, starting_position, Q):
        state = starting_position
        path = [tuple(np.array(state))]

        while not self.isCheckMate(state, True):  # Continue until reaching the goal state
            action = np.argmax(Q[state])  # Choose the action with the highest Q-value
            state = tuple(np.array(state) + np.array([self.UP, self.DOWN, self.LEFT, self.RIGHT][action]))
            path.append(state)

        return path

    # Millor accio
    def get_best_action(self, state, epsilon, q_table):
        available_actions = self.getListNextStatesW(state)

        # Epsilon-greedy
        if np.random.rand() > epsilon:
            action = np.random.choice(available_actions)
        else:
            q_values = q_table[state][available_actions]
            max_q_value = np.max(q_values)
            best_actions = [action for action in available_actions if q_table[state][action] == max_q_value]
            action = np.random.choice(best_actions)

        """
        if not drunken_sailor():
            return action
        else:
            available_actions = available_actions[available_actions != action]
            return np.random.choice(available_actions)
        """
        return action

    def drunken_sailor(self, ratio=0.01):
        if np.random.rand() < ratio:
            return True
        else:
            return False

    def heuristica(self, currentState, color):
        # In this method, we calculate the heuristics for both the whites and black ones
        # The value calculated here is for the whites,
        # but finally from everything, as a function of the color parameter, we multiply the result by -1
        value = 0

        bkState = self.getPieceState(currentState, 12)
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)
        brState = self.getPieceState(currentState, 8)
        filaBk = bkState[0]
        columnaBk = bkState[1]
        filaWk = wkState[0]
        columnaWk = wkState[1]

        if wrState != None:
            filaWr = wrState[0]
            columnaWr = wrState[1]
        if brState != None:
            filaBr = brState[0]
            columnaBr = brState[1]

        # We check if they killed the black tower
        if brState == None:
            value += 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)
            if distReis >= 3 and wrState != None:
                filaR = abs(filaBk - filaWr)
                columnaR = abs(columnaWr - columnaBk)
                value += (min(filaR, columnaR) + abs(filaR - columnaR)) / 10
            # If we are white, the closer our king from the opponent, the better
            # we substract 7 to the distance between the two kings, since the max distance they can be at in a board is 7 moves
            value += (7 - distReis)
            # If they black king is against a wall, we prioritize him to be at a corner, precisely to corner him
            if bkState[0] == 0 or bkState[0] == 7 or bkState[1] == 0 or bkState[1] == 7:
                value += (abs(filaBk - 3.5) + abs(columnaBk - 3.5)) * 10
            # If not, we will only prioritize that he approaches the wall, to be able to approach the check mate
            else:
                value += (max(abs(filaBk - 3.5), abs(columnaBk - 3.5))) * 10

        # They killed the black tower. Within this method, we consider the same conditions than in the previous condition
        # Within this method we consider the same conditions than in the previous section, but now with reversed values.
        if wrState == None:
            value += -50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)
            if distReis >= 3 and brState != None:
                filaR = abs(filaWk - filaBr)
                columnaR = abs(columnaBr - columnaWk)
                value -= (min(filaR, columnaR) + abs(filaR - columnaR)) / 10
            # If we are white, the close we have our king from the opponent, the better
            # If we substract 7 to the distance between both kings, as this is the max distance they can be at in a chess board
            value += (-7 + distReis)
            if wkState[0] == 0 or wkState[0] == 7 or wkState[1] == 0 or wkState[1] == 7:
                value -= (abs(filaWk - 3.5) + abs(columnaWk - 3.5)) * 10
            else:
                value -= (max(abs(filaWk - 3.5), abs(columnaWk - 3.5))) * 10

        # We are checking blacks
        if self.isWatchedBk(currentState):
            value += 20
        # We are checking whites
        if self.isWatchedWk(currentState):
            value += -20

        # If black, values are negative, otherwise positive
        if not color:
            value = (-1)*value

        return value

    def reward(self, state):
        return -1 if not self.isCheckMate(state, True) else 100

    def q_learning(self, init_state, alpha, gamma, epsilon, num_episodes=4, convergence_threshold=0.0001, convergence_window=10):
        #mean_q_value_changes = []
        q_table = dict()
        print_first = False
        print_second = False
        state = init_state
        turn = 1
        for episode in range(num_episodes):
            #prev_Q = np.copy(q_table)

            while not self.isCheckMate(state, True):  # Until reaching the goal
                next_state = self.get_best_action(state, epsilon, q_table) #TODO !!!

                #next_state = np.array(state) + np.array([self.UP, DOWN, LEFT, RIGHT][action])  # Up Down Left Right
                # Reward function
                reward = self.reward(state)

                action = self.getMovement(state, next_state)
                self.chess.move(action[0], action[1])

                #state_index = state_to_index(state)
                #next_state_index = state_to_index(next_state)
                q_table[state][action] = (q_table[state][action] +
                                          alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action]))

                state = self.getCurrentState()
                print("\nTurn " + str(turn) + ":")
                turn += 1
                self.chess.board.print_board()

            #mean_q_value_changes.append(np.mean(np.abs(q_table - prev_Q)))

            # Check for convergence
            # if len(mean_q_value_changes) > convergence_window:

            """mean_change = np.mean(mean_q_value_changes[-convergence_window:])
            if not print_first:
                print("First Q-table:\n", q_table, '\n')
                print_first = True

            if mean_change < 0.001 and not print_second:
                print("Second Q-table:\n", q_table, '\n')
                print_second = True

            if mean_change < convergence_threshold:
                print("Converged in episode ", episode)
                break"""
        return q_table

    def mitjana(self, values):
        sum = 0
        N = len(values)
        for i in range(N):
            sum += values[i]
        return sum / N

    def desviacio(self, values, mitjana):
        sum = 0
        N = len(values)
        for i in range(N):
            sum += pow(values[i] - mitjana, 2)
        return pow(sum / N, 1 / 2)

    def calculateValue(self, values):
        if len(values) == 0:
            return 0
        mitjana = self.mitjana(values)
        desviacio = self.desviacio(values, mitjana)
        # If deviation is 0, we cannot standardize values, since they are all equal, thus probability will be equiprobable
        if desviacio == 0:
            # We return another value
            return values[0]

        esperanca = 0
        sum = 0
        N = len(values)
        for i in range(N):
            # Normalize value, with mean and deviation - zcore
            normalizedValues = (values[i] - mitjana) / desviacio
            # make the values positive with function e^(-x), in which x is the standardized value
            positiveValue = pow(1 / math.e, normalizedValues)
            # Here we calculate the expected value, which in the end will be expected value/sum
            # Our positiveValue/sum represent the probabilities for each value
            # The larger this value, the more likely
            esperanca += positiveValue * values[i]
            sum += positiveValue
        return esperanca / sum

    def printStatus(self):
        print("")
        print("White is in Check?     ", aichess.isCheck(aichess.getCurrentState(), True))
        print("Black is in Check?     ", aichess.isCheck(aichess.getCurrentState(), False))
        print("White is in CheckMate? ", aichess.isCheckMate(aichess.getCurrentState(), True))
        print("Black is in CheckMate? ", aichess.isCheckMate(aichess.getCurrentState(), False))

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     sys.exit(usage())

    # initialize board
    TA = np.zeros((8, 8))
    # load initial state
    # white pieces
    TA[7][0] = 2
    TA[7][4] = 6
    # black pieces
    TA[0][4] = 12

    # iterations
    num_episodes = 1000
    alpha = 0.2  # Learning rate //   Tested with 0.1 0.2 0.3 0.4 0.5
    gamma = 0.8  # Discount factor // Tested with 0.9 0.8 0.7 0.6 0.5
    epsilon = 0.9  # Tested beetwen 0.1-0.9

    # initialize board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentState = aichess.chess.board.currentStateW.copy()
    print("printing board")
    aichess.chess.boardSim.print_board()
    #aichess.printStatus() #for TESTING purposes ONLY

    result_q_table = aichess.q_learning(currentState, alpha, gamma, epsilon, num_episodes)