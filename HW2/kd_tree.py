from sklearn.neighbors import KDTree
import numpy as np
import matplotlib.pyplot as plt

points = list()

with open('points.txt', 'r') as f:
    for line in f:
        points.append(list(line.strip('\n').split(' ')))

# To integer list
points = np.asarray(points, dtype=int)
points = points.tolist()


def buildKDTree(points, depth=0, k=2):
    n = len(points)

    if n <= 0:
        return None

    axis = depth % k

    #sorted by axis
    sorted_points = sorted(points, key = lambda points: points[axis])

    return{
        'point': sorted_points[int(n/2)],
        'left' : buildKDTree(sorted_points[:int(n/2)], depth + 1),
        'right': buildKDTree(sorted_points[int(n/2)+1 :], depth + 1)
    }

def plotKDTree(Tree, max_x, max_y ,min_x, min_y, branch, parent_node, depth=0, k=2):
    '''
    Tree:
    the subtree with the root of current node
        max_x: the maximum x value when constructing the line
        max_y: the maximum y value when constructing the line
        min_x: the minimum x value when constructing the line
        min_y: the minimum y value when constructing the line
        branch: if the current node is the leftchild of its parent_node return true
                if the current node is the rightchild return false
        parent_node is the parent node of the current node
        depth represent the depth of the current node

    '''
    cur_node = Tree['point']
    left_subtree  = Tree['left']
    right_subtree = Tree['right']
    axis = depth % k

    #draw a vertical splitting line
    if axis == 0:
        if parent_node is not None:
            if branch:
                max_y = parent_node[1]
            else:
                min_y = parent_node[1]
        plt.plot([cur_node[0], cur_node[0]], [min_y, max_y], linestyle='-', color = 'red')


    elif axis == 1:
        if parent_node is not None:
            if branch:
                max_x = parent_node[0]
            else:
                min_x = parent_node[0]
        plt.plot([min_x, max_x], [cur_node[1], cur_node[1]], linestyle='-', color = 'blue')
    plt.plot(cur_node[0], cur_node[1], 'ko')

    if left_subtree is not None:
        branch = True
        plotKDTree(left_subtree, max_x, max_y, min_x, min_y, branch, cur_node, depth+1)
    if right_subtree is not None:
        branch = False
        plotKDTree(right_subtree, max_x, max_y, min_x, min_y, branch, cur_node, depth+1)


#
##
########
##########################################################################################
sorted_x = sorted(points, key = lambda point: point[0])
sorted_y = sorted(points, key = lambda point: point[1])
max_val = 10
min_val = 0
max_x = max(sorted_x[-1][0], max_val)
min_x = min(sorted_x[0][0], min_val)
max_y = max(sorted_y[-1][1], max_val)
min_y = min(sorted_y[0][1], min_val)
delta = 2
#Construct the tree
kd_tree = buildKDTree(points)

#Draw the tree
plt.figure("K-d Tree", figsize=(10., 10.))
plt.axis([min_x-delta, max_x+delta, min_y-delta, max_x+delta])
plt.xticks([i for i in range(min_x-delta, max_x+delta, 1)])
plt.yticks([i for i in range(min_x-delta, max_y+delta, 1)])
plotKDTree(kd_tree, max_x, max_y, min_x, min_y, branch=None, parent_node=None)

plt.title('K-D Tree')
plt.show()
