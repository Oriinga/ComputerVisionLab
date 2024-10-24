from classes import *

puzzle = Puzzle(MATCH_IMGS)
corner_piece = puzzle.pieces[3]

# Start BFS by adding in the bottom left corner piece
queue = []
queue.append(corner_piece)
corner_piece.insert()
corner_piece.inserted = True

# TODO: Rest of BFS
