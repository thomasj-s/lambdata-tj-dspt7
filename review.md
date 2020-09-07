# Code Review

Nice package, I was initially impressed with how you thought about utility and transferred the skills you learned from
the previous unit. You will need more practice with classes moving forward, since form before function, here's what
a typical class syntax looks like 

    class Key:
        def __init__(self, key):
        self.key = key


    class Door(key):
        def __init__(slef, key):
            # Inherits previous class's variable
            super()__init__(key=key)


        def function1(self):
            print(key)
            
    
### TODO 1 Write a class in the global scope
In line 48, the BaseSet class is in the basic_classifier() function, which is something I haven't seen before.
Functions are usually nested under classes. 

### TODO 2 Have multiple methods under one class

There are 3 global functions which does it job, but if we are trying to program in an object oriented manner, these 
three functions will have to be nested under classes if possible. 