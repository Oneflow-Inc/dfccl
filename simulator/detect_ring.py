# 这段代码是一个用于检测图中环的Python脚本，采用深度优先搜索（DFS）算法。其基本原理和执行过程如下：

# 图的表示：该脚本使用字典graph来表示图。每个键是一个顶点，与之关联的值是一个列表，包含与该顶点相邻的顶点。例如，"a": ["b", "c"]意味着顶点a与顶点b和c相连。

# DFS算法：深度优先搜索（DFS）是一种用于遍历或搜索树或图的算法。该算法从一个顶点开始，探索尽可能深的分支，直到找不到未访问的相邻顶点为止，然后回溯。

# 检测环：在DFS过程中，如果访问到一个已经访问过的顶点，且该顶点在当前的访问路径上（即在trace列表中），那么就检测到一个环。trace列表维护着当前路径上的顶点。

# 实现细节：

# vis列表用于存储访问过的顶点。
# trace列表用于跟踪当前DFS路径上的顶点。
# dfs函数用于执行深度优先搜索。它接受一个顶点v作为参数。
# 如果顶点v已在vis中，表示已访问过：
# 如果v也在trace中，表示找到一个环，打印出环的路径。
# 否则，返回，因为不需要再探索从此顶点出发的路径。
# 将v加入vis和trace。
# 对于顶点v的每一个相邻顶点，递归调用dfs函数。
# 完成对所有相邻顶点的探索后，从trace中移除顶点v。

import pprint


PRINT_RING = True
PRINT_DEBUG = False
PRINT_PERFORMANCE = False

class RingDetector:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.vis = []
        self.trace = []
        self.find_ring = False

    def dfs(self, graph, v):
        if PRINT_DEBUG:
            print(f"in dfs, v: {v}, self.vis: {self.vis}, trace: {self.trace}, v in self.vis: {v in self.vis}, v in self.trace: {v in self.trace}", flush=True)
        if v in self.vis:
            if v in self.trace:
                self.find_ring = True
                if PRINT_RING:
                    v_index = self.trace.index(v)
                    print("Ring found:")
                    for i in range(v_index, len(self.trace)):
                        print(self.trace[i] + ' ', end='')
                    print(v)
                    # print("\n")
                return
            return

        self.vis.append(v)
        self.trace.append(v)
        for vs in graph[v]:
            self.dfs(graph, vs)
            if self.find_ring:
                return
        self.trace.pop()

    def detect_ring(self, graph) -> bool:
        if PRINT_DEBUG:
            print("in detect_ring", flush=True)
            pprint.pprint(graph)
        
        if PRINT_PERFORMANCE:
            max_neighbors = 0
            for v in graph.keys():
                if len(graph[v]) > max_neighbors:
                    max_neighbors = len(graph[v])
            print(f"#nodes: {len(graph.keys())} max_neighbors: {max_neighbors}", flush=True)
        

        for v in graph.keys():
            self.dfs(graph, v)
            if self.find_ring:
                return True
            
        if PRINT_PERFORMANCE:
            print(f"no ring, return", flush=True)
        return False
