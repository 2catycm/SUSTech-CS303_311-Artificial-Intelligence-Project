class A():
    def __init__(self):
        self.b = B()
        self.b.b()
A()
class B():
    def b(self):
        print("Hello from b")
A()