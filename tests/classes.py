

class Alpha:
    def __init__(self,val):
        self.v = val

    def __add__(self,beta):
        return self.v+beta.v

    def __str__(self):
        return str(self.v)



a = Alpha(5)
b = Alpha(10)
a+=b
print(a)
print(b)
