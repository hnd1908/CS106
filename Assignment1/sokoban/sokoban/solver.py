import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     

    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    #In ra trạng thái khởi đầu
    print("Start state:", startState)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            # In trạng thái kết thúc
            print("End state:", node[-1])
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    #Lấy vị trí ban đầu của hộp từ gameState
    beginBox = PosOfBoxes(gameState)
    #Lấy vị trí ban đầu của nhân vật Sokoban từ gameState
    beginPlayer = PosOfPlayer(gameState)
    #Lưu trạng thái khởi đầu
    startState = (beginPlayer, beginBox)
    #In ra trạng thái khởi đầu
    print("Start state:", startState)
    # Hàng đợi chứa các trạng thái để duyệt
    frontier = collections.deque([[startState]])
    # Tập hợp lưu trữ các trạng thái đã được duyệt, đảm bảo thuật toán không chạy lại các trạng thái đã được duyệt
    exploredSet = set()
    # Hàng đợi chứa các hành động mà nhân vật có thể thực hiện tương ứng với mỗi trạng thái trong hàng chờ frontier
    actions = collections.deque([[0]])
    # Danh sách tạm thời chứa hành động
    temp = []
    while frontier: # Trong khi hàng đợi vẫn còn trạng thái để duyệt
        node = frontier.popleft() # Lấy ra và xóa trạng thái (phần tử đầu tiên của hàng đợi) ở đầu bên trái ngoài cùng của hàng đợi
        node_action = actions.popleft() # Lấy ra và xóa hành động (phần tử đầu tiên của hàng đợi) ở đầu bên trái ngoài cùng của hàng đợi
        if isEndState(node[-1][-1]):  # node[-1][-1] đề cập tới vị trí của hộp trong trạng thái cuối cùng. Dòng lệnh kiểm tra xem trạng thái hiện tại của hộp (cụ thể là node[-1][-1]) có phải là trạng thái kết thúc của trò chơi không.
            temp += node_action[1:] # Thêm tất cả các hành động ứng với trạng thái hiện tại (node_action) vào temp
            # In trạng thái kết thúc
            print("End state:", node[-1])
            break # Thoát khỏi vòng lặp
        if node[-1] not in exploredSet: # Kiểm tra xem trạng thái hiện tại đã được duyệt chưa
            exploredSet.add(node[-1]) # Nếu chưa, thêm trạng thái đó vào danh sách các trạng thái đã được duyệt
            for action in legalActions(node[-1][0], node[-1][1]): # Duyệt qua tất cả các hành động hợp lệ của nhân vật đối với trạng thái hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Cập nhật trạng thái mới của nhân vật và các hộp dựa trên hành động
                if isFailed(newPosBox):# Nếu tình trạng mới của hộp là trạng thái thất bại (bị mắc kẹt) thì bỏ qua
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)]) # Thêm trạng thái mới của nhân vật và các hộp vào hàng đợi frontier
                actions.append(node_action + [action[-1]]) # Thêm hành động mới vào hàng đợi actions
    return temp # Trả về danh sách các hành động để đạt được trạng thái kết thúc của trò chơi

def cost(actions):
    """A cost function"""
    action_str=''.join(map(str, actions))
    return len([x for x in action_str if x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    # Lấy vị trí ban đầu của các hộp từ gameState
    beginBox = PosOfBoxes(gameState)
    # Lấy vị trí ban đầu của người chơi từ gameState
    beginPlayer = PosOfPlayer(gameState)
    startState = (beginPlayer, beginBox)  # Lưu trạng thái khởi đầu
    #In ra trạng thái khởi đầu
    print("Start state:", startState)
    frontier = PriorityQueue()  # Hàng đợi ưu tiên chứa các trạng thái để duyệt
    frontier.push([startState], 0)  # Thêm trạng thái ban đầu vào hàng đợi với chi phí 0
    exploredSet = set()  # Tập hợp lưu trữ các trạng thái đã được duyệt
    actions = PriorityQueue()  # Hàng đợi ưu tiên chứa các hành động
    actions.push([0], 0)  # Thêm hành động ban đầu vào hàng đợi với chi phí 0
    temp = []  # Danh sách tạm thời chứa hành động
    while not frontier.isEmpty():  # Trong khi hàng đợi ưu tiên vẫn còn trạng thái để duyệt
        node = frontier.pop()  # Lấy ra trạng thái có chi phí thấp nhất từ hàng đợi ưu tiên
        node_action = actions.pop()  # Lấy ra hành động tương ứng
        if isEndState(node[-1][-1]):  # Kiểm tra xem trạng thái hiện tại của hộp có phải là trạng thái kết thúc của trò chơi không
            temp += node_action[1:]  # Thêm tất cả các hành động ứng với trạng thái hiện tại vào danh sách tạm thời
            # In trạng thái kết thúc
            print("End state:", node[-1])
            break  # Kết thúc vòng lặp
        if node[-1] not in exploredSet:  # Kiểm tra xem trạng thái hiện tại đã được duyệt chưa
            exploredSet.add(node[-1])  # Nếu chưa, thêm trạng thái đó vào tập hợp các trạng thái đã được duyệt
            for action in legalActions(node[-1][0], node[-1][1]):  # Duyệt qua tất cả các hành động hợp lệ của nhân vật đối với trạng thái hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)  # Cập nhật trạng thái mới của nhân vật và các hộp dựa trên hành động
                if isFailed(newPosBox):  # Nếu trạng thái mới của hộp là trạng thái thất bại (bị mắc kẹt) thì bỏ qua
                    continue 
                # Thêm trạng thái mới của nhân vật và các hộp vào hàng đợi
                frontier.push(node + [(newPosPlayer, newPosBox)], cost(node_action[1:]))
                # Thêm hành động mới vào hàng đợi ưu tiên với chi phí là thông số ưu tiên
                actions.push(node_action + [action[-1]], cost(node_action[1:]))
    return temp  # Trả về danh sách các hành động để đạt được trạng thái kết thúc của trò chơi

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':        
        result = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    print("Actions:",len(result))
    return result
