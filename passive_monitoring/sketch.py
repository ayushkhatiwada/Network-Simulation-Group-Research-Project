class Sketch:
    """
    At the moment the sketch keeps track of cumulative delay and the packet count on a switch
    """
    def __init__(self, width, depth, seed):
        self.width = width
        self.depth = depth
        self.tables = [[{'count': 0, 'delay_sum': 0.0} for _ in range(width)] for _ in range(depth)] # A 2d table where each cell is a dict
        self.hash_seeds = [seed + i for i in range(depth)]

    def hash_(self, key, i):
        return hash(str(key) + str(self.hash_seeds[i])) % self.width

    def update(self, key, delay):
        for i in range(self.depth):
            idx = self.hash_(key, i)
            self.tables[i][idx]['count'] += 1
            self.tables[i][idx]['delay_sum'] += delay 

    def estimate_delay(self, key):
        estimates = []
        for i in range(self.depth):
            idx = self.hash_(key, i)
            cell = self.tables[i][idx]
            if cell['count'] > 0:
                avg_delay = cell['delay_sum'] / cell['count']
                estimates.append(avg_delay)
            else:
                estimates.append(0.0) # if there are no updates for the key
        return min(estimates)