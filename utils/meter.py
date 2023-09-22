class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeterList(object):

    def __init__(self):
        self.val = []
        self.avg = []
        self.sum = []
        self.count = 0

    def reset(self):
        self.val = []
        self.avg = []
        self.sum = []
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if not self.sum:
            self.sum = [v*n for v in val]
        else:
            self.sum = [s + v * n for s,v in  zip(self.sum, val)]
        self.count += n
        self.avg = [s/self.count for s in self.sum]
