class parentt:
    def __init__(self, name):
        self.name = name
    
    def namex(self):
        self.x = 5
        # print(self.name)
        # return None

class childd(parentt):
    def __init__(self, name, First):
        super().__init__(name)
        self.first = First


childdd = childd('van der Wee', 'Tom')
childdd.namex()
print(childdd.x)