

class Biggest:
    
    A = 10
    def __init__(self):
        print(self.A)
        pass

class Lesser(Biggest):
    
    def __init__(self):
        print(self.A)


class Greater(Biggest):
    
    def __init__(self):
        print(self.A)
        
