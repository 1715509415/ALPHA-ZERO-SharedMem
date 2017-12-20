class Board():
    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(0,0)]


    def __init__(self, n):
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n

        # Set up the initial 2 spaces
        self.pieces[0][0] = 1           #left most top
        self.pieces[int(self.n/2)-1][int(self.n)-1] = -1 #right most bottom

    def __getitem__(self, index): 
        return self.pieces[index]


    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for player 1, -1 for player 2
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    moves.update(newmoves)
        return list(moves)

    def has_legal_moves(self, color):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    if len(newmoves)>0:
                        return True
        return False

    def get_moves_for_square(self, square):
        """Returns all the legal moves that use the given square as a base.
        That is, if the given square is (3,4) and it contains a black piece,
        and (3,5) and (3,6) contain white pieces, and (3,7) is empty, one
        of the returned moves is (3,7) because everything from there to (3,4)
        is flipped.
        """
        (x,y) = square

        # determine the color of the piece.
        color = _get_color(self,square)
        if color==0:
            return None

        # search all possible directions.
        moves = []
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                # print(square,move,direction)
                moves.append(move)

        # return the generated move list
        return moves

    def execute_move(self, move, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        #Much like move generation, start at the new piece's square and
        #follow it on all 8 directions to look for a piece allowing flipping.

        # Add the piece to the empty square.
        # print(move)
        # flips = [flip for direction in self.__directions
        #               for flip in self._get_flips(move, direction, color)]
        # assert len(list(flips))>0
        # for x, y in flips:
        #     #print(self[x][y],color)
        #     self[x][y] = color

    def _discover_move(self, origin, direction):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        (x,y) = origin
        color = _get_color(self,(x,y))
        flips = []
        #output should be origin,destination,amount
        for a, b in Board._increment_move(origin, direction, self.n):
            if self[a][b] > 254:
                return None
            else:
                if x == a && y == b:
                    return Move(origin,(a,b),0)
                else:
                    #hier even defineren hoeveel je wilt overschrijven.
                    return Move(origin,(a,b),int(self[x][y])*0.7)
            # if self[x][y] == 0:
            #     if flips:
            #         # print("Found", x,y)
            #         return (x, y)
            #     else:
            #         return None
            # elif self[x][y] == color:
            #     return None
            # elif self[x][y] == -color:
            #     # print("Flip",x,y)
            #     flips.append((x, y))

    def _upgrade_board(self, oldBoard):
        for y in range(self.n):
            for x in range(self.n):
                # ik weet niet zeker of dit kan dalijk gaat de bot express worden overschijven dat ze hetzelfde blijven zodat de waarde groeit wanneer dit eigenlijk niet mag
                if self[x][y]==oldBoard.pieces[x][y]:
                    #increment
                    value = self[x][y]
                    if value >= 128:
                        self[x][y] += 8
                        if self[x][y] > 255:
                            self[x][y] = 255
                    if value >= 64:
                        self[x][y] += 7
                    elif value >= 32:
                        self[x][y] += 6
                    elif value >= 16:
                        self[x][y] += 5
                    elif value >= 8:
                        self[x][y] += 4
                    elif value >= 4:
                        self[x][y] += 3
                    elif value >= 2:
                        self[x][y] += 2
                    elif value >= 1:
                        self[x][y] += 1

    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        #move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)): 
        #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move=list(map(sum,zip(move,direction)))
            #move = (move[0]+direction[0],move[1]+direction[1])

    def _get_color(self, square):
        (x,y) = square
        if self[x][y] > 0:
            return 1
        if self[x][y] < 0:
            return -1
        if self[x][y] == 0:
            return 0

class Move:
    def __init__(origin, destination, amount):
        self.origin = origin
        self.destination = destination
        self.amount = amount