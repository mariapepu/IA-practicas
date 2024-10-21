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

        print(currentState)

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

    #### AUXILIARY FUNCTIONS ####

    def isBlackInCheckMate(self, current_state):
        '''
        Checks if the black king is in check mate

        params:
            current_state: list representation of the current state of the board

        returns:
            True if black king is in check mate, False otherwise
        '''

        # Creamos un tablero para simular todas las posibles posiciones del rey negro
        self.newBoardSim(current_state)

        # Comprobamos si para cada posible movimiento, el rey se encuentra en jaque
        for next_state in self.getListNextStatesB(self.chess.boardSim.currentStateB):
            if not self.isWatchedBk(next_state): # Si en una posicion no esta en jaque el rey aun puede escapar
                return False
        
        return True
    
    def isWhiteInCheckMate(self, current_state):
        '''
        Checks if the white king is in check mate

        params:
            current_state: list representation of the current state of the board

        returns:
            True if black king is in check mate, False otherwise
        '''

        # Creamos un tablero para simular todas las posibles posiciones del rey blanco
        self.newBoardSim(current_state)

        # Comprobamos si para cada posible movimiento, el rey se encuentra en jaque
        for next_state in self.getListNextStatesW(self.chess.boardSim.currentStateW):
            if not self.isWatchedBk(next_state): # Si en una posicion no esta en jaque el rey aun puede escapar
                return False
        
        return True
    
    ###############
    ### MINIMAX ###
    ###############

    # WHITES
    def minValueWhite(self, current_state, current_depth):

        # Check if the current state is terminal or the depth has reached max
        if self.isWhiteInCheckMate(current_state) or current_depth == self.depthMax:
            return self.heuristica(current_state, True)  # Returns the utility
        
        # Iniciamos búsqueda
        best_successor = None
        utility = float('inf')

        # Sets a new simulation board
        self.newBoardSim(current_state)

        # Loop to search black pieces next states
        for next_state in self.getListNextStatesB(self.chess.boardSim.currentState):

            # Recursive call to perform the search
            _, next_state_utility = self.maxValueWhite(next_state, current_depth + 1)


            if next_state_utility < utility:
                utility = next_state_utility
                best_successor = next_state

        return best_successor, utility


    def maxValueWhite(self, current_state, current_depth):


        # Comprovamos si las negras pueden dar con check mate o hemos llegado al limite de profundidad
        if self.isBlackInCheckMate(current_state) or current_depth == self.depthMax:
            return self.heuristica(current_state, True)
        
        # Iniciamos búsqueda
        best_successor = None
        utility = float('-inf')

        #self.chess.boardSim.print_board()

        for next_state in self.getListNextStatesW(self.chess.boardSim.currentStateW):

            #print(next_state)
            
            _, next_state_utility = self.minValueWhite(next_state, current_depth + 1)
            
            if next_state_utility > utility:
                utility = next_state_utility
                best_successor = next_state

        return best_successor, utility


    # BLACKS
    def minValueBlack(self, current_state, current_depth):



        # Comprovamos si las blancas pueden dar con check mate o hemos llegado al limite de profundidad
        if self.isBlackInCheckMate(current_state) or current_depth == self.depthMax:
            return self.heuristica(current_state, False)
        
        # Iniciamos búsqueda
        best_successor = None
        utility = float('inf')

        self.newBoardSim(current_state)
        #self.chess.boardSim.print_board()

        for next_state in self.getListNextStatesW(current_state):

            _, next_state_utility = self.maxValueBlack(next_state, current_depth + 1)

            if next_state_utility < utility:
                utility = next_state_utility
                best_successor = next_state

        return best_successor, utility


    def maxValueBlack(self, current_state, current_depth):

        

        # Comprovamos si las negras pueden dar con check mate o hemos llegado al limite de profundidad
        if self.isWhiteInCheckMate(current_state) or current_depth == self.depthMax:
            return self.heuristica(current_state, True)
        
        # Iniciamos búsqueda
        best_successor = None
        utility = float('-inf')

        self.newBoardSim(current_state)
        #self.chess.boardSim.print_board()

        for next_state in self.getListNextStatesB(current_state):
            
            _, next_state_utility = self.minValue(next_state, current_depth + 1)
            
            if next_state_utility > utility:
                utility = next_state_utility
                best_successor = next_state

        return best_successor, utility
    

    def minimaxGame(self, depthWhite, depthBlack):
        
        current_state = self.getCurrentState()

        self.newBoardSim(current_state)

        while True:

            # Next white move
            white_next_state, _ = self.maxValueWhite(current_state, depthWhite)

            break

            # Movemos al nuevo estado de las blancas
            movement = self.getMovement(current_state, white_next_state)
            self.chess.move(movement[0], movement[1])

            # Asignamos el movimeinto del blanco como nuevo estado actual
            currentState = white_next_state

            if self.isBlackInCheckMate(current_state):
                return True
            
            # Ejecutamos minimax para las negras
            black_next_state = self.minimaxBlack(currentState, depthBlack)

            # Movemos al nuevo estado de las negras
            movement = self.getMovement(currentState, black_next_state)
            self.chess.move(movement[0], movement[1])

            # Asignamos el movimiento de las negrsa como nuevo estado actual
            currentState = black_next_state

            if self.isWhiteInCheckMate(currentState):
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
    #aichess.minimaxGame(4,4)
  # Add code to save results and continue with other exercises
    algorithms = ['MiniMax', 'Alpha-Beta Prunning', 'ExpectMiniMax']

    print('Que algoritmo quieres probar')
    for i, a in enumerate(algorithms):
        print(f'\t{i + 1}. {a}')

    option = -1

    while option < 1 or option > len(algorithms):
        option = int(input('> '))

    print("printing board")
    aichess.chess.boardSim.print_board()

    if option == 1:
        ganador = aichess.minimaxGame(4,4)

        if ganador:
            print(f"Ha ganado blanco")
        else:
            print(f"Ha ganado negro")

    #elif option == 2:
        #ganador = aichess.alphaBetaPoda(4,4)
        #print(f"Ha ganado {ganador}")

    #elif option == 3:
        #ganador = aichess.expectimax(4,4)
        #print(f"Ha ganado {ganador}")


  

