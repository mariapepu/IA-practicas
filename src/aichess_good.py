#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import math

import chess
import board
import numpy as np
import sys
import queue
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations


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
        Promotes a pawn that has reached the other side to another, or the same, piece

    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.listVisitedSituations = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW;
        self.depthMax = 8;
        self.checkMate = False

    def copyState(self, state):
        
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState
        
    def isVisitedSituation(self, color, mystate):
        
        if (len(self.listVisitedSituations) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedSituations)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedSituations.__getitem__(k)[1]) and color == \
                            self.listVisitedSituations.__getitem__(k)[0]:
                        isVisited = True

            return isVisited
        else:
            return False


    def getCurrentStateW(self):

        return self.myCurrentStateW

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

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False

    def isWatchedBk(self, currentState):

        self.newBoardSim(currentState)

        if self.getPieceState(currentState, 12) is None:
            print('A')

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
            # We check the possible positions of the black tower, and we chck if in any o them it may killt he white king
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True

        return False



    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)

    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState

    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
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
        # Given a state and a successor state, return the postiion of the piece that has been moved in both states
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

    def heuristica(self, currentState, color):
        #In this method, we calculate the heuristics for both the whites and black ones
        #The value calculated here is for the whites, 
        # but finally from verything, as a function of the color parameter, we multiply the result by -1
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
                value += (min(filaR, columnaR) + abs(filaR - columnaR))/10
            # If we are white white, the closer our king from the oponent, the better
            # we substract 7 to the distance between the two kings, since the max distance they can be at in a board is 7 moves
            value += (7 - distReis)
            # If they black king is against a wall, we prioritize him to be at a corner, precisely to corner him
            if bkState[0] == 0 or bkState[0] == 7 or bkState[1] == 0 or bkState[1] == 7:
                value += (abs(filaBk - 3.5) + abs(columnaBk - 3.5)) * 10
            #If not, we will only prioritize that he approahces the wall, to be able to approach the check mate
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
            # If we are white, the close we have our king from the oponent, the better
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
            value = (-1) * value

        return value
    
    def getListNextPieceStates(self, myState, white):
            return self.getListNextStatesW(myState) if white else self.getListNextStatesB(myState)

    def isCheck(self, currentState, white):
            return self.isWatchedWk(currentState) if white else self.isWatchedBk(currentState)

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

    def isCheckMate(self, state, white):
            if self.isCheck(state, white) and not self.isCheck(state, not white):
                for nextState in self.getListNextColorStates(state, white):
                    if not self.isCheck(nextState, white):
                        return False
                return True
            return False

    def getColorState(self, currentState, white):
        return self.getWhiteState(currentState) if white else self.getBlackState(currentState)

    def minimaxTree(self, state, depth, isWhiteTurn):
            if self.isCheck(state, not isWhiteTurn):
                return [(float("inf") if isWhiteTurn else float("-inf")), state, 0]
            elif (self.isCheckMate(state, isWhiteTurn)):
                return [(float("-inf") if isWhiteTurn else float("inf")), state, 0]
            elif (depth == 0):
                return [self.heuristica(state, True), state, depth+1]
            else:
                result = [float("-inf") if isWhiteTurn else float("inf"), None, depth]
                self.newBoardSim(state)
                for nextState in self.getListNextColorStates(state, isWhiteTurn):
                    h = self.minimaxTree(nextState, depth - 1, not isWhiteTurn)
                    if ((h[0] > result[0]) if isWhiteTurn else (h[0] < result[0])):
                        result[0] = h[0]
                        result[1] = nextState
                        result[2] = h[2]+1
                    elif result[1] == None and not self.isCheck(nextState, isWhiteTurn):
                        result[1] = nextState
                    elif (h[0] == result[0]) and (h[2]<result[2]-1) and (not self.isCheck(nextState, isWhiteTurn)):
                        result[1] = nextState
                        result[2] = h[2]+1
                #self.newBoardSim(state)
                #self.chess.boardSim.print_board()
                #print("HEURISTICA: " + str(result[0]))
                return result
            
    def minimax(self, depthWhite, depthBlack):
        isWhiteTurn = True
        turn = 1
        state = self.getCurrentState()
        while (not self.isCheckMate(state, isWhiteTurn)) and (not self.isCheck(state, not isWhiteTurn)):
            minimax = self.minimaxTree(state, (depthWhite if isWhiteTurn else depthBlack), isWhiteTurn)
            move = self.getMovement(state, minimax[1])
            self.chess.move(move[0], move[1])

            state = self.getCurrentState()
            print("\nTurn " + str(turn) + ":")
            turn += 1
            self.chess.board.print_board()

            isWhiteTurn = not isWhiteTurn
        return [state, turn]


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import math

import chess
import board
import numpy as np
import sys
import queue
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations


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
        Promotes a pawn that has reached the other side to another, or the same, piece

    """


    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.listVisitedSituations = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW;
        self.depthMax = 8;
        self.checkMate = False


    def copyState(self, state):
        
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState
        

    def isVisitedSituation(self, color, mystate):
        
        if (len(self.listVisitedSituations) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedSituations)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedSituations.__getitem__(k)[1]) and color == \
                            self.listVisitedSituations.__getitem__(k)[0]:
                        isVisited = True

            return isVisited
        else:
            return False


    def getCurrentStateW(self):

        return self.myCurrentStateW


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

        isSameState = isSameState1 and isSameState2
        return isSameState


    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False


    def isWatchedBk(self, currentState):

        self.newBoardSim(currentState)

        if not self.getPieceState(currentState, 12):
            raise Exception('The black king is not in the board')

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
            # We check the possible positions of the black tower, and we chck if in any o them it may killt he white king
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True

        return False


    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)


    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState


    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
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
        # Given a state and a successor state, return the postiion of the piece that has been moved in both states
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


    def heuristica(self, currentState, color):
        #In this method, we calculate the heuristics for both the whites and black ones
        #The value calculated here is for the whites, 
        # but finally from verything, as a function of the color parameter, we multiply the result by -1
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
                value += (min(filaR, columnaR) + abs(filaR - columnaR))/10
            # If we are white white, the closer our king from the oponent, the better
            # we substract 7 to the distance between the two kings, since the max distance they can be at in a board is 7 moves
            value += (7 - distReis)
            # If they black king is against a wall, we prioritize him to be at a corner, precisely to corner him
            if bkState[0] == 0 or bkState[0] == 7 or bkState[1] == 0 or bkState[1] == 7:
                value += (abs(filaBk - 3.5) + abs(columnaBk - 3.5)) * 10
            #If not, we will only prioritize that he approahces the wall, to be able to approach the check mate
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
            # If we are white, the close we have our king from the oponent, the better
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
            value = (-1) * value

        return value
    

    def isValidMovement(self, current_state, successor):
        '''
        Return if a successor is valid given a color and the current state.

        Parameters:
        -----------
        current_state: List[List[int]]
            Current state of the board

        successor: List[List[int]]
            Successor state
        
        Returns:
        --------
        True if successor is valid, False otherwise
        '''
    
        # Get the positioin of the moved piece
        _, moved_piece = self.getMovement(current_state, successor)

        # Chek if white or black king is taken
        if moved_piece[2] > 6: # Black pieces moved
            self.getPieceState(successor, 6)[0:2] != moved_piece[0:2] # White king is taken
        else: # White pieces moved
            return self.getPieceState(successor, 12)[0:2] != moved_piece[0:2] # Black king is taken


    def getSuccessors(self, current_state, color):
        '''
        Returns all valid successors for the white player given a current state.

        Parameters:
        -----------
        current_state: List[List[int]]
            Current state of the board
        color: bool
            True if white player, False if black player
        
        Returns:
        --------
        valid_successors: List[List[List[int]]]
            List of all valid successors states for the white player.
        '''

        # Store valid successors
        valid_successors = []

        # Get all possible successors depending on the color moved, even if they are not valid
        successors = self.getListNextStatesW(self.getWhiteState(current_state)) if color else self.getListNextStatesB(self.getBlackState(current_state))

        # Check if each successor is a valid state
        for s in successors:
            # Join the current state of the black player to the white successor
            s = s + self.getBlackState(current_state) if color else s + self.getWhiteState(current_state)

            # If is valid, add to the list of valid successors
            if self.isValidMovement(current_state, s):
                valid_successors.append(s)
        
        return valid_successors
    

    def isBlackInCheckMate(self, current_state):
        '''
        '''

        black_successors = self.getBlackSuccessors(current_state)
        for successor in black_successors:
            if not self.isWatchedBk(successor):
                return False
        
        return True
    

    def isCheckMate(self, current_state, color):
        '''
        Check if a player is in checkmate looking if his king is whatched for each of his possible moves.
        '''

        successors = self.getSuccessors(current_state, color=True) if color else self.getSuccessors(current_state, color=False)

        for s in successors:
            is_watched = self.isWatchedWk(s) if color else self.isWatchedBk(s)
            if not is_watched:
                return False

        return True

    
    def minimaxWhite(self, current_state, depth):
        best_successor, _ = self.maxValueWhite(current_state, depth) 
        return best_successor


    def maxValueWhite(self, current_state, depth):

        if self.isCheckMate(current_state, color=True):
            return current_state, 1000
        
        if depth == 0:
            return current_state, self.heuristica(current_state, True)
        
        best_successor = None
        utility = float('-inf')
        
        for successor in self.getSuccessors(current_state, color=True):
            _, u = self.minValueWhite(successor, depth - 1)

            if u > utility:
                utility = u
                best_successor = successor

        return best_successor, utility


    def minValueWhite(self, current_state, depth):

        if self.isCheckMate(current_state):
            return current_state, -1000
        
        if depth == 0:
            return current_state, self.heuristica(current_state, False)
        
        best_successor = None
        utility = float('inf')
        
        for successor in self.getBlackSuccessors(current_state):

            _, u = self.maxValueWhite(successor, depth - 1)

            if u < utility:
                utility = u
                best_successor = successor

        return best_successor, utility
    
    def minimaxBlack(self, current_state, depth):
        best_successor, _ = self.maxValueBlack(current_state, depth) 
        return best_successor


    def maxValueBlack(self, current_state, depth):

        print(current_state)

        if self.isCheck(current_state):
            return current_state, 1000
        
        if depth == 0:
            return current_state, self.heuristica(current_state, False)
        
        best_successor = None
        utility = float('-inf')
        
        for successor in self.getBlackSuccessors(current_state):

            #print('HOLA')

            _, u = self.minValueBlack(successor, depth - 1)

            if u > utility:
                utility = u
                best_successor = successor

        return best_successor, utility


    
    
    def minimaxGame(self, depthWhite, depthBlack):
        
        current_state = self.getCurrentState()

        while True:
            # Next white move
            white_next_state = self.minimaxWhite(current_state, depthWhite)
            movement = self.getMovement(current_state, white_next_state)
            self.chess.move(movement[0], movement[1])

            self.chess.board.print_board()

            # Asignamos el movimeinto del blanco como nuevo estado actual
            current_state = white_next_state

            if self.isBlackInCheckMate(current_state):
                return True
            
            # Ejecutamos minimax para las negras
            black_next_state = self.minimaxBlack(current_state, depthBlack)
            movement = self.getMovement(current_state, black_next_state)
            self.chess.move(movement[0], movement[1])

            self.chess.board.print_board()

            # Asignamos el movimiento de las negrsa como nuevo estado actual
            current_state = black_next_state

            if self.isWhiteInCheckMate(current_state):
                return False


    def alphaBetaPoda(self, depthWhite,depthBlack):
        
        currentState = self.getCurrentState()
        # Your code here  
        
    def expectimax(self, depthWhite, depthBlack):
        
        currentState = self.getCurrentState()
        # Your code here       
        
        

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
        # If deviation is 0, we cannot standardize values, since they are all equal, thus probability willbe equiprobable
        if desviacio == 0:
            # We return another value
            return values[0]

        esperanca = 0
        sum = 0
        N = len(values)
        for i in range(N):
            #Normalize value, with mean and deviation - zcore
            normalizedValues = (values[i] - mitjana) / desviacio
            # make the values positive with function e^(-x), in which x is the standardized value
            positiveValue = pow(1 / math.e, normalizedValues)
            # Here we calculate the expected value, which in the end will be expected value/sum            
            # Our positiveValue/sum represent the probabilities for each value
            # The larger this value, the more likely
            esperanca += positiveValue * values[i]
            sum += positiveValue

        return esperanca / sum
     

if __name__ == "__main__":
    #   if len(sys.argv) < 2:
    #       sys.exit(usage())

    # intiialize board
    TA = np.zeros((8, 8))

    #Configuració inicial del taulell
    TA[7][0] = 2
    TA[7][4] = 6
    TA[0][7] = 8
    TA[0][4] = 12

    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)

    print("printing board")
    #aichess.chess.boardSim.print_board()
    #aichess.getListNextStatesB(aichess.getCurrentState())
    #print(aichess.chess.boardSim.listNextStates)

  # Run exercise 1
    aichess.minimaxGame(4, 4)
  # Add code to save results and continue with other exercises
    #algorithms = ['MiniMax', 'Alpha-Beta Prunning', 'ExpectMiniMax']

    #print('Que algoritmo quieres probar')
    #for i, a in enumerate(algorithms):
        #print(f'\t{i + 1}. {a}')

    #option = -1

    #while option < 1 or option > len(algorithms):
        #option = int(input('> '))

    #print("printing board")
    #aichess.chess.boardSim.print_board()

    #if option == 1:
        #ganador = aichess.minimaxGame(4,4)

        #if ganador:
            #print(f"Ha ganado blanco")
        #else:
            #print(f"Ha ganado negro")

    #elif option == 2:
        #ganador = aichess.alphaBetaPoda(4,4)
        #print(f"Ha ganado {ganador}")

    #elif option == 3:
        #ganador = aichess.expectimax(4,4)
        #print(f"Ha ganado {ganador}")


  


    

if __name__ == "__main__":
    #   if len(sys.argv) < 2:
    #       sys.exit(usage())

    # intiialize board
    TA = np.zeros((8, 8))

    #Configuració inicial del taulell
    TA[7][0] = 2
    TA[7][4] = 6
    TA[0][7] = 8
    TA[0][4] = 12

    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)

    print("printing board")
    aichess.chess.boardSim.print_board()
    #aichess.getListNextStatesB(aichess.getCurrentState())
    #print(aichess.chess.boardSim.listNextStates)

  # Run exercise 1
    aichess.minimax(4,4)