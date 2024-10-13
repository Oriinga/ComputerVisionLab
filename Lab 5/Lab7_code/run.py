from classes import *

puzzle = Puzzle(MATCH_IMGS)
corner_piece = puzzle.pieces[3]

# Start BFS by adding in the bottom left corner piece
queue = []
queue.append(corner_piece)
corner_piece.insert()
corner_piece.inserted = True

# Loop through self.edge_list of the corner piece and find the two flat edges (lets call them first_edge
# and second_edge where second_edge is anti-clockwise of first_edge). first_edge.point2
# should be the same coordinates as second_edge.point1

# edges are in anti clock wise, so flat 1 flat 2, the mod operator is to make the index a rolling index (0-1 = 3)

first_edge:Edge = None
second_edge:Edge = None
for i in range(4):
    if(corner_piece.edge_list[i].is_flat == True):
        if(corner_piece.edge_list[(i - 1) % 4].is_flat == True):
            first_edge = corner_piece.edge_list[(i - 1) % 4]
            second_edge = corner_piece.edge_list[i]
        elif(corner_piece.edge_list[(i + 1) % 4].is_flat == True):
            first_edge = corner_piece.edge_list[i]
            second_edge = corner_piece.edge_list[(i + 1) % 4]
        else:
            raise Exception("Not a corner piece") 
if(set(first_edge.point2) != set(second_edge.point1) ):
    raise Exception("Flat edges dont share corner point")

# add first edge point2

# I dont know if i miss-understand, col,row major means that the bottom left is at 0,699, or is it 0,799?
piece_height = abs(first_edge.point2[0] - first_edge.point1[0])
piece_width = abs(second_edge.point2[1] - first_edge.point2[1])

pts_src = [
    first_edge.point2[::-1].tolist(),
    first_edge.point1[::-1].tolist(),
    second_edge.point2[::-1].tolist()
]
pts_dst = [
    [0,canvas.shape[0]-1],  # Bottom-left (scaled proportionally)
    [0,799 - piece_height],  # Along left edge
    [second_edge.point2[0].item(),canvas.shape[0]-1]  # Along bottom edge
]

pts_src = np.array(pts_src,dtype=np.float32)
pts_dst = np.array(pts_dst,dtype=np.float32)

M = cv2.getAffineTransform(pts_src,pts_dst)
corner_piece.dst = cv2.warpAffine(corner_piece.image,M,(700,800))
mask_warped = cv2.warpAffine(corner_piece.mask,M,(700,800))
plt.imshow(corner_piece.dst)
plt.show()