

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) / 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# print quicksort([3, 6, 8, 10, 1, 2, 1])

# x = 3
# print type(x) # Prints "<type 'int'>"
# print x       # Prints "3"
# print x + 1   # Addition; prints "4"
# print x - 1   # Subtraction; prints "2"
# print x * 2   # Multiplication; prints "6"
# print x ** 2  # Exponentiation; prints "9"
# x += 1
# print x  # Prints "4"
# x *= 2
# print x  # Prints "8"
# y = 2.5
# print type(y) # Prints "<type 'float'>"
# print y, y + 1, y * 2, y ** 2 # Prints "2.5 3.5 5.0 6.25"

# t = True
# f = False
# print type(t) # Prints "<type 'bool'>"
# print t and f # Logical AND; prints "False"
# print t or f  # Logical OR; prints "True"
# print not t   # Logical NOT; prints "False"
# print t != f

# hello = 'hello'   # String literals can use single quotes
# world = "world"   # or double quotes; it does not matter.
# print hello       # Prints "hello"
# print len(hello)  # String length; prints "5"
# hw = hello + ' ' + world  # String concatenation
# print hw  # prints "hello world"
# hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
# print hw12  # prints "hello world 12"
#
# s = "hello"
# print s.capitalize()  # Capitalize a string; prints "Hello"
# print s.upper()       # Convert a string to uppercase; prints "HELLO"
# print s.rjust(7)      # Right-justify a string, padding with spaces; prints "  hello"
# print s.center(7)     # Center a string, padding with spaces; prints " hello "
# print s.replace('l', '(ell)')  # Replace all instances of one substring with another;
#                                # prints "he(ell)(ell)o"
# print '  world '.strip()  # Strip leading and trailing whitespace; prints "world"

# xs = [3, 1, 2]   # Create a list
# print xs, xs[2]  # Prints "[3, 1, 2] 2"
# print xs[-1]     # Negative indices count from the end of the list; prints "2"
# xs[2] = 'foo'    # Lists can contain elements of different types
# print xs         # Prints "[3, 1, 'foo']"
# xs.append('bar') # Add a new element to the end of the list
# print xs         # Prints "[3, 1, 'foo', 'bar']"
# x = xs.pop()     # Remove and return the last element of the list
# print x, xs      # Prints "bar [3, 1, 'foo']"

# nums = range(5)    # range is a built-in function that creates a list of integers
# print nums         # Prints "[0, 1, 2, 3, 4]"
# print nums[2:4]    # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
# print nums[2:]     # Get a slice from index 2 to the end; prints "[2, 3, 4]"
# print nums[:2]     # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
# print nums[:]      # Get a slice of the whole list; prints ["0, 1, 2, 3, 4]"
# print nums[:-1]    # Slice indices can be negative; prints ["0, 1, 2, 3]"
# nums[2:4] = [8, 9] # Assign a new sublist to a slice
# print nums         # Prints "[0, 1, 8, 9, 4]"

# animals = ['cat', 'dog', 'monkey']
# for animal in animals:
#     print animal

# animals = ['cat', 'dog', 'monkey']
# for idx, animal in enumerate(animals):
#     print '#%d: %s' % (idx + 1, animal)

# nums = [0, 1, 2, 3, 4]
# squares = []
# for x in nums:
#     squares.append(x ** 2)
# print squares

# nums = [0, 1, 2, 3, 4]
# squares = [x ** 2 for x in nums]
# print squares

# nums = [0, 1, 2, 3, 4]
# even_squares = [x ** 2 for x in nums if x % 2 != 0]
# print even_squares

# d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
# print d['cat']       # Get an entry from a dictionary; prints "cute"
# print 'cat' in d     # Check if a dictionary has a given key; prints "True"
# d['fish'] = 'wet'    # Set an entry in a dictionary
# print d['fish']      # Prints "wet"
# # print d['monkey']  # KeyError: 'monkey' not a key of d
# print d.get('monkey', 'N/A')  # Get an element with a default; prints "N/A"
# print d.get('fish', 'N/A')    # Get an element with a default; prints "wet"
# del d['fish']        # Remove an element from a dictionary
# print d.get('fish', 'N/A')

# d = {'person': 2, 'cat': 4, 'spider': 8}
# for animal in d:
#     legs = d[animal]
#     print 'A %s has %d legs' % (animal, legs)

# d = {'person': 2, 'cat': 4, 'spider': 8}
# for animal, legs in d.iteritems():
#     print 'A %s has %d legs' % (animal, legs)

# nums = [0, 1, 2, 3, 4]
# even_num_to_square = {x: x / 4 for x in nums if x % 2 == 0}
# print even_num_to_square

# animals = {'cat', 'dog'}
# print 'cat' in animals   # Check if an element is in a set; prints "True"
# print 'fish' in animals  # prints "False"
# animals.add('fish')      # Add an element to a set
# print 'fish' in animals  # Prints "True"
# print len(animals)       # Number of elements in a set; prints "3"
# animals.add('cat')       # Adding an element that is already in the set does nothing
# print len(animals)       # Prints "3"
# animals.remove('cat')    # Remove an element from a set
# print len(animals)

animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal)

