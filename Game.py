from __future__ import print_function
from SharedMemLogic import Board
import numpy as np
import sys
sys.path.append('..')

class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(0,0)]
    __previousTurn = 0
    __oldBoard = Board(8)
    __Changes = []
    def __init__(self,n):
        self.n = n
        # for i in range(self.n):
        #     for j in range(self.n):
        #         self.__Changes[i][j] = [0]

        self.__Changes = [0]*self.n
        for i in range(self.n):
            self.__Changes[i] = [0]*self.n


    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        b = Board.init_with_oldboard(self.__oldBoard)
        return np.array(b.pieces)

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """

        return (self.n,self.n)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        #this should be the outputs + the endTurn signal
        return self.n*self.n*9+1 #je kunt eventueel nog de de hoeveelheid toevoegen door nog een factor 256 toe te voegen

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """


        # deze actie moet uit de source code van lucrasoft gehaald worden
        b = init_with_oldboard(self.__oldBoard)
        b.pieces = np.copy(board)


        if self.__previousTurn == -player and player == 1:
            b._upgrade_board(self.__Changes)
            self.__oldBoard.pieces = np.copy(b)
            #clear changes after they get used in upgrade board
            for i in range(self.n):
                for j in range(self.n):
                    self.__Changes[i][j] = [0]
        self.__previousTurn = player


        #convert action to coordinates
        (x,y,z) = getCoords(action);
        (dx,dy) = __directions[z]


        if action[-1]: #== self.n*self.n*9   #als laatste element waar is moet je naar de beurt van de tegenstander.
            #define hier wat je move gaat zijn. gebruik hier action. of iets van input en output board. move zou misschien ook een array kunnen zijn van actions ofzo
            #move = Move((x,y),(x+dx,y+dy),0)
            #b.execute_move(move,player)     #in logic maken
            return (b.pieces, -player)
        else:
            #define hier wat je move gaat zijn. gebruik hier action. of iets van input en output board. move zou misschien ook een array kunnen zijn van actions ofzo
            move = Move((x,y),(x+dx,y+dy),0)
            if not(x == x+dx and y == y+y+dy):
                self.__Changes[x][y] = 1
                self.__Changes[x+dx][y+cy] = 1
                # __Changes.append((x,y))

            b.execute_move(move,player)     #in logic maken
            return (b.pieces, player)



    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        valids = [0] * self.getActionSize()
        b = Board.init_with_oldboard(self.__oldBoard)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)      #in logic maken         Move(origin,destination,amount)
        if len(legalMoves)==0:
            valids[-1]=1 #dit is het laatste element. min getal telt vanaf achter ipv voor
            return np.array(valids)
        for move in legalMoves:
            ((x1,y1), (x2,y2),amount) = move
            z = getIndexByDirection(getDirection((x1,y1),(x2,y2)))
            valids[getIndex(x1,y1,z)]=1         #klopt dit?
        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost.
        """
        b = Board.init_with_oldboard(self.__oldBoard)
        b.pieces = np.copy(board)

        player1 = 0
        player2 = 0
        for i in range(self.n):
            for j in range(self.n):
                if b.pieces[i][j] > 0:
                    player1 += abs(int(b.pieces[i][j]))
                elif b.pieces[i][j] < 0:
                    player2 += abs(int(b.pieces[i][j]))

        if player1 == 0:
            return -1
        if player2 == 0:
            return 1
        return 0

        # #dit niet zo doen maar aan de hand van de punten
        # if b.has_legal_moves(player):
        #     return 0
        # if b.has_legal_moves(-player):
        #     return 0
        # if b.countDiff(player) > 0:
        #     return 1
        # else:
        #     return -1

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """

        return player * board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.tostring()

def display(board):
    n = board.shape[0]

    for y in range(n):
        print (y,"|",end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|",end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1: print("b ",end="")
            elif piece == 1: print("W ",end="")
            else:
                if x==n:
                    print("-",end="")
                else:
                    print("- ",end="")
        print("|")

    print("   -----------------------")

def getCoords(index):
    z = index / (self.n*self.n)
    index -= z * self.n * self.n
    x = index / self.n
    y = index % self.n
    return(x,y,z)

def getIndex(x,y,z):
    return z * self.n * self.n + y * self.n + x

def getDirection(origin, destination):
    (x1,y1) = origin
    (x2,y2) = destination
    dx = x1-x2
    dy = y1-y2
    return (dx,dy)

def getIndexByDirection(dx,dy):
    for i in len(__directions):
        if(self.__directions[i]==(dx,dy)):
            return i
    return -1