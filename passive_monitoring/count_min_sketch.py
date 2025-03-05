class CountMinSketch:
    def __init__(self, width, depth, seed):
        self.width = width
        self.depth = depth
        self.tables = [[0 for _ in range(width)] for _ in range(depth)]
        self.hash_seeds = [seed + i for i in range(depth)]
    
    def hash_(self, key, i):
        return hash(str(key) + str(self.hash_seeds[i])) % self.width
    
    def update(self, key, count=1):
        for i in range(self.depth):
            idx = self.hash_(key, i)
            self.tables[i][idx] += count
    
    def estimate_count(self, key):
        return min(self.tables[i][self.hash_(key, i)] for i in range(self.depth))
