#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import math

import time

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


    def isKingEaten(self, moved_piece, successor, color):
        
        for s in successor:
            if color and moved_piece[:2] == s[:2] and s[2] == 12:
                return True # Black king eaten
            if not color and moved_piece[:2] == s[:2] and s[2] == 6:
                return True # White king eaten
    

    def isValidMovement(self, current_state, successor, color):
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

        # Check if the king is eaten
        if self.isKingEaten(moved_piece, successor, color):
            return False
        
        if moved_piece == None:
            return False
        
        return True


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

        try:
            # Get all possible successors depending on the color moved, even if they are not valid
            successors = self.getListNextStatesW(self.getWhiteState(current_state)) if color else self.getListNextStatesB(self.getBlackState(current_state))
        except Exception as e:
            print(current_state)
            print(self.chess.boardSim.currentState)
            raise Exception(e)


        # Check if each successor is a valid state
        for s in successors:
            # Join the current state of the black player to the white successor
            s = s + self.getBlackState(current_state) if color else s + self.getWhiteState(current_state)
                        
            # If is valid, add to the list of valid successors
            if self.isValidMovement(current_state, s, color):

                for piece_i in s:
                    for piece_j in s:
                        if piece_i[:2] == piece_j[:2] and piece_i[2] != piece_j[2]:
                            white_piece = piece_i if piece_i[2] <= 6 else piece_j
                            black_piece = piece_i if piece_i[2] > 6 else piece_j
                            if color:
                                s.remove(black_piece)
                            else:
                                s.remove(white_piece)

                valid_successors.append(s)

        return valid_successors
    

    def isCheckMate(self, current_state, color):
        '''
        Check if a player is in checkmate looking if his king is whatched for each of his possible moves.
        '''

        successors = self.getSuccessors(current_state, color=True) if color else self.getSuccessors(current_state, color=False)

        for s in successors:
            is_watched = self.isWatchedWk(current_state) if color else self.isWatchedBk(current_state)
            if not is_watched:
                return False


    def minimaxWhite(self, current_state, depth):
        best_successor, _ = self.maxValue(current_state, depth) 
        return best_successor
    

    def minimaxBlack(self, current_state, depth):
        best_successor, _ = self.minValue(current_state, depth)
        return best_successor


    def maxValue(self, current_state, depth):
        '''
        '''

        # Check if the white player is in checkmate
        if self.isCheckMate(current_state, color=True):
            return current_state, float('-inf')
        
        # If not checkmate, check if the maximum depth has been reached
        if depth == 0:
            return current_state, self.heuristica(current_state, True)
        
        # If not terminal state, search for the best successor
        best_successor = None
        utility = float('-inf')
        successors = self.getSuccessors(current_state, color=True)

        for s in successors:
            self.newBoardSim(s)
            _, u = self.minValue(s, depth - 1)

            if u > utility:
                utility = u
                best_successor = s

        return best_successor, utility


    def minValue(self, current_state, depth):
        '''
        '''
        
        # Check if the white player is in checkmate
        if self.isCheckMate(current_state, color=False):
            return current_state, float('inf')
        
        # If not checkmate, check if the maximum depth has been reached
        if depth == 0:
            return current_state, self.heuristica(current_state, False)
        
        # If not terminal state, search for the best successor
        best_successor = None
        utility = float('inf')
        successors = self.getSuccessors(current_state, color=False)
        for s in successors:
            self.newBoardSim(s)
            _, u = self.maxValue(s, depth - 1)

            if u < utility:
                utility = u
                best_successor = s

        return best_successor, utility
    
    
    def minimaxGame(self, depthWhite, depthBlack):
        
        current_state = self.getCurrentState()

        while True:

            # White's next move
            white_best_successor = self.maxValue(current_state, depthWhite)
            movement = self.getMovement(current_state, white_best_successor)

            self.chess.board.print_board()

            # Asignamos el movimeinto del blanco como nuevo estado actual
            current_state = white_best_successor

            # Check if the white player has won
            if self.isCheckMate(current_state, color=False):
                return True
            
            # Ejecutamos minimax para las negras
            black_best_successor = self.minValue(current_state, depthBlack)
            movement = self.getMovement(current_state, black_best_successor)
            self.chess.move(movement[0], movement[1])

            self.chess.board.print_board()

            # Asignamos el movimiento de las negrsa como nuevo estado actual
            current_state = black_best_successor

            if self.isCheckMate(current_state, color=True):
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
    aichess.chess.boardSim.print_board()

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


  

