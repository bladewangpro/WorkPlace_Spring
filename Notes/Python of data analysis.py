#random number generated
import numpy as np
data = {i : np.random.randn() for i in range(7)}
data
#ipython is more readable than the normal python
'''Ipython shell
we can use TAB key to complete the sentence in your browser.
this shell is a cosmetically different version of python
'''
'''Jupyter notebook
b = [1,2,3]
b?
print?
'''
def add_numbers(a, b):
	return a + b
#using ? shows us the information about this function
#using ?? will show the source code of this function asap
#? has a final usage, which is for searching the IPython namespace in a manner similar to the Unix
#we can run any file as Python program inside IPython session
%run ipython_script_test.py
#if a python file need certain parameters, then sys.argv
#if u wish to give a script access to variables already defined in the interactive IPython namespace, %run -i instead of plain %run
%load ipython_script_test.py
#in Jupyter notebook, imports a script into a code cell
%paste
#it will takes whatever test in the clipboard and executes it as a single block in the cell
%cpaste
#it gives you a special prompt for pasting code into
---------------------------------Magic Command
import numpy as np
a = np.random.randn()
%timeit np.dot(a,a)   # We will get the times it executed
#many of magic command has options, we can use the ? to check
%debug?
%pwd
foo = %pwd
#data visualization and other interface libraries like matplotlib
type(a)

a = 5
isinstance(a, int)
a = 5; b = 4.5
isinstance(a, (int, float))
isinstance(b, (int, float))

#show the arributes and methods
a = 'foo'
a.<Press TAB>
#also
getattr(a, 'split')

-----------------------------------Duck typing "If it walks like a duck, quacks like a duck, then it's a duck"
#sometimes, you may not care about the type of an object but rather than whether is has certain method or behavior.
def isiterable(obj):
	try:
		iter(obj)
		return True
	except TypeError:
		return False
isiterable('a string')
isiterable('[1, 2, 3]')
isiterable(5)

if not isinstance(x, list) and isiterable(x):
	x = list(x)

import some_module #some_module.py
result = some_module.f(5)
pi = some_module.PI 

from some_module import f,g,PI
result = g(5, PI)

import some_module as sm
from some_module import PI as pi, g as gf
r1 = sm.f(pi)
r2 = gf(6, pi)

#Most of the binary math operations and comparisons are as you might expect:
a is b  #True if a and b reference the same Python object
a is not b #True if a and b reference different Python objects

#strings and tuples are immutable
#Scalar type in Python
str, None, bytes, float, bool, int
c = """
This is a longer string that
spans multiple lines
"""
c.count('\n') # 3

a = 'this is a string'
b = a.replace('string', 'longer string')

#casting
a = 5.6
s = str(a)
s = 'python'
list(s)
#'' need \, "" don't need \\
s = r'this\\has\\no\\special\\characters'

a = 'apple'
b = 'pig'
a + b

template = '{0:.2f} {1:s} are worth US${2:d}'
'''
{0:.2f} means to format the first argument as a floating-point number with two
decimal places.
{1:s} means to format the second argument as a string.
{2:d} means to format the third argument as an exact integer.
'''
template.format(4.5560, 'Argentine Pesos', 1)

val_utf8 = val.encode('utf-8')
val_utf8.decode('utf-8')

bytes_val = b'this is bytes'
decoded = bytes_val.decode('utf8')
True and True
False or True

s = '3.14159'
fval = float(s)
int(fval)
bool(fval)
bool(0)

a = None
a is None
b = 5
b is not None

from datetime import datetime, date, time
dt = datetime(2011, 10, 29, 20, 30, 21)
dt.day
dt.minute

dt.date()
dt.time()
dt.strftime('%m/%d/%Y %H:%M')
datetime.strptime('20091031', '%Y%m%d')
dt.replace(minute = 0, second = 0)
dt2 = datetime(2011, 11, 15, 22, 30)
delta = dt2 - dt
delta
type(delta)
dt + delta


if a < b or c > d:
	print('Made it')
4 > 3 > 2 > 1

for value in collection:
	# do something with value
for i in range(4):
	for j in range(4):
		if j > i:
			break
		print((i, j))


for a, b, c in iterator:
# do something

x = 256
total = 0
while x > 0:
	if total > 500:
		break
	total += x
	x = x //2

if x < 0:
	print('negative!')
elif x == 0:
# TODO: put something smart here
	pass
else:
	print('positive!')

list(range(10))
list(range(0, 20, 2))
list(range(5, 0, -1))

seq = [1, 2, 3, 4]
for i in range(len(seq)):
	val = seq[i]

sum = 0
for i in range(100000):
# % is the modulo operator
	if i % 3 == 0 or i % 5 == 0:
		sum += i
#value = true-expr if condition else false-expr
x = 5
'Non-negative' if x >= 0 else 'Negative'


tup = 4, 5, 6 #tuple
nested_tup = (4, 5, 6), (7, 8)
tuple([4, 0, 2]) #convert to the tuple
tup = tuple('string')

#While the objects stored in a tuple may be mutable themselves, once the tuple is created it’s not possible to modify which object is stored in each slot:
tup = tuple(['foo', [1, 2], True])
tup[2] = False # running error
#If an object inside a tuple is mutable, such as a list, you can modify it in-place:
tup[1].append(3)
#You can concatenate tuples using the + operator to produce longer tuples:
(4, None, 'foo') + (6, 0) + ('bar',)
('foo', 'bar') * 4

-----------------unpacking tuples
tup = (4, 5, 6)
a, b, c = tup

tup = 4, 5, (6, 7)
a, b, (c, d) = tup

#swap
b, a = a, b

seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for a, b, c in seq:
	print('a={0}, b={1}, c={2}'.format(a, b, c))

values = 1, 2, 3, 4, 5
a , b, *rest = values
a, b # tuple
rest #list

a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)

----------------------------list
tup = ('foo', 'bar', 'baz')
b_list = list(tup)

gen = range(10)
list(gen)

b_list.append('dwarf')
b_list.insert(1, 'red')
b_list.pop(2)

b_list.append('foo')
b_list.remove('foo')

'dwarf' in b_list
'dwarf' not in b_list

[4, None, 'foo'] + [7, 8, (2, 3)]
x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])


everything = []
for chunk in list_of_lists:
	everything.extend(chunk)
#is faster than the concatenative alternative:
everything = []
for chunk in list_of_lists:
	everything = everything + chunk  #concatenating is time-consuming

a = [7, 2, 5, 1, 3]
a.sort()
b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len)
'''
The built-in bisect module implements binary search and insertion into a sorted list.
bisect.bisect finds the location where an element should be inserted to keep it sor‐
ted, while bisect.insort actually inserts the element into that location:
'''
import bisect
c = [1, 2, 2, 2, 3, 4, 7]
bisect.bisect(c, 2)  #4
bisect.bisect(c, 5)  #6
bisect.insort(c, 6)  #c [1, 2, 2, 2, 3, 4, 6, 7]


-------------------------------slicing
seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[1:5]
seq[3:4] = [6, 3]
seq[:5]
seq[3:]
seq[-4:]
seq[-6:-2]
seq[::2]  #[7, 3, 3, 6, 1]
seq[::-1] #reversing a list or tuple



i = 0
for value in collection:
# do something with value
	i += 1
#equal to 
for i, value in enumerate(collection):
# do something with value

some_list = ['foo', 'bar', 'baz']
mapping = {}
for i, v in enumerate(some_list):
	mapping[v] = i
mapping

sorted([7, 1, 2, 6, 0, 3, 2])
sorted('horse race')

seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zipped = zip(seq1, seq2)
list(zipped)
seq3 = [False, True]
list(zip(seq1, seq2, seq3)) #only zip the first two items

for i, (a, b) in enumerate(zip(seq1, seq2)):
	print('{0}: {1}, {2}'.format(i, a, b))

pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'),('Schilling', 'Curt')]
first_names, last_names = zip(*pitchers)
first_names
last_names

list(reversed(range(10)))

---------------------------------dictionary
'b' in d1
del d1[5]
ret = d1.pop('dummy') #ret is the value who is being deleted, and d1 is the dictionary after the modification

list(d1.keys())
list(d1.values())
d1.update({'b' : 'foo', 'c' : 12}) #You can merge one dict into another using the update method

#Creating dicts from sequences
mapping = {}
for key, value in zip(key_list, value_list):
	mapping[key] = value

mapping = dict(zip(range(5), reversed(range(5))))

--------------------setting the default value
if key in some_dict:
	value = some_dict[key]
else:
	value = default_value
#equals to
value = some_dict.get(key, default_value)

words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}

for word in words:
	letter = word[0]
	if letter not in by_letter:
		by_letter[letter] = [word]
	else:
		by_letter.append(word)

for word in words:
	letter = word[0]
	by_letter.setdefault(letter, []).append(word)

from collections import defaultdict
by_letter = defaultdict(list)
for word in words:
	by_letter[word[0]].append(word)

--------------------hashability
hash('string')
hash(1, 2, (2, 3))
hash(1, 2, [2, 3])  #this one will fail, cuz it has list and will change

d = {}
d[tuple([1, 2, 3])] = 5  #To use a list as a key, one option is to convert it to a tuple, which can be hashed as long as its elements also can

--------------------------------------------------set
set([2, 2, 2, 1, 3, 3])
{2, 2, 2, 1, 3, 3}

a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}

a.union(b) #which is equvalent to a | b
a.intersection(b)  #which is equvalent to a & b

c = a.copy()
c |= b
d = a.copy()
d &= b 

[expr for val in collection if condition]
#equals to 
result = []
for val in collection:
	if condition:
		result.append(expr)

strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
[x.upper() for x in strings if len(x) > 2]

unique_lengths = {len(x) for x in strings}

set(map(len, strings))

loc_mapping = {val : index for index, val in enumerate(strings)}

all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'], ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]
names_of_interest = []
for names in all_data:
	enough_es = [name for name in names if name.count('e') >= 2]
	names_of_interest.extend(enough_es)

some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
flattened = [x for tup in some_tuples for x in tup]

#Keep in mind that the order of the for expressions would be the same if you wrote a nested for loop instead of a list comprehension:
flattened = []
for tup in some_tuples:
	for x in tup:
		flattened.append(x)

return_value = f()
def f():
	a = 5
	b = 6
	c = 7
	return {'a' : a, 'b' : b, 'c' : c}



states = ['     Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda', 'southcarolina##', 'West virginia?']

import re
def clean_strings(strings):
	result = []
	for value in strings:
		value = value.strip()
		value = re.sub('[!#?]', '', value)
		value = value.title()
		result.append(value)
	return result

def remove_punctuation(value):
	return re.sub('[!#?]', '', value)
clean_ops = [str.strip, remove_punctuation, str.title]

def clean_strings(strings, ops):
	result = []
	for value in strings:
		for function in ops:
			value = function(value)
		result.append(value)
	return result;
clean_strings(states, clean_ops)

#we can use function as arguments to other functions
for x in map(remove_punctuation, states):
	print(x)

---------------------------------------------------------------------------------------------------lambda
strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
strings.sort(key=lambda x: len(set(list(x))))
strings # ['aaaa', 'foo', 'abab', 'bar', 'card']

def apply_to_list(some_list, f):
	return [f(x) for x in some_list]
ints = [4, 0, 1, 5, 6]
apply_to_list(ints, lambda x: x * 2)

-----------------------------------------------------Curring
'''
Currying is computer science jargon (named after the mathematician Haskell Curry) that means deriving new functions from existing ones by partial argument application.
'''
def add_numbers(x, y):
	return x + y
add_five = lambda x: add_numbers(x, 5)
#equals to
from functools import partial
add_five = partial(add_numbers, 5)


---------------------------------------------------------generators
some_dict = {'a':1, 'b':2, 'c':3}
for key in some_dict:
	print(key)
#When you write for key in some_dict , the Python interpreter first attempts to create an iterator out of some_dict :
dict_iterator = iter(some_dict)
list(dict_iterator)

#To create a generator, use the yield keyword instead of return in a function
def squares(n=10):
	print('Generating squares from 1 to {0}'.format(n ** 2))
	for i in range(1, n+1):
		yield i ** 2
gen = squares()
for x in gen:
	print(x, end='   ')

-----------------------------------------------Generator expressions
gen = (x ** 2 for x in range(100)) #gen is a generator object
#is completely equivalent to the following:
def _make_gen():
	for x in range(100):
		yield x ** 2
gen = _make_gen()

#generator expressions can be used instead of list comprehensions as function arguments in many cases.
sum(x ** 2 for x in range(100))
dict((i, i **2) for i in range(5))

#The standard library itertools module has a collection of generators for many common data algorithm.
#groupby takes any sequence and a function, grouping consecutive elements in the sequence by return value of the function.
import itertools
first_letter = lambda x: x[0]
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
for letter, names in itertools.groupby(names, first_letter):
	print(letter, list(names))

----------------------------------------------address error
'''
we wanted a version of float that fails gracefully, returning the input arguments
'''
def attempt_float(x):
	try:
		return float(x)
	except:
		return x
#if you wanna suppress ValueError, TYpeError might indicate a legitimate bug in your program
def attemp_float(x):
	try:
		return float(x)
	except ValueError:
		return x

def attemp_float(x):
	try:
		return float(x)
	except (TypeError, ValueError):
		return x

'''
In some cases, you may not want to suppress an exception, but you want some code
to be executed regardless of whether the code in the try block succeeds or not. To do
this, use finally
'''
f = open(path, 'w')

try:
	write_to_file(f)
finally:
	f.close()

#block succeed using else:
f = open(path, 'w')
try:
	write_to_file(f)
except:
	print('Fails')
else:
	print('Succeed')
finally:
	f.close()

-----------------------------------------------------------------Files and the Operating Sytem
#Obsolute or relative path
path = 'examples/segismudo.txt'
f = open(path)
#by default,, the file is opened in read-only mode 'r'. We can then treat the file handle f like a list and iterate over the line 
for line in f:
	pass
#The lines come out of the file with the end-of-line (EOL) markers intact., so you will often see code to get an EOL-free list of lines in a file like:
lines = [x.rstrip() for x in open(path)]
#need to close the opened file
f.close()

#other way to make it easier to clean up open files
with open(path) as f:
	lines = [x.rstrip() for x in f]
#this will automatically close the file 

'''
If we had typed f = open(path, 'w') , a new file at examples/segismundo.txt would
have been created (be careful!), overwriting any one in its place. There is also the 'x'
file mode, which creates a writable file but fails if the file path already exists. See
Table 3-3 for a list of all valid file read/write modes.
'''
#read return a certain number of characters from the file.
f = open(path)
f.read(10)
f2.open(path, 'rb')#binary mode
f2.read(10)
#The read method advance the file handle's pos by the number of bytes read. tell gives you the current position
f.tell()    #11
f2.tell()   #10
#check the default coding function
import sys
sys.getdefaultencoding()    #'utf-8'
#seek changes the file position to the indicated byte in the file
f.seek(3)  #out 3
f.read(1)  #read the fourth position

f.close()
f2.close()
'''
Read-only mode                                                                                                                                                           r
Write-only mode; creates a new file (erasing the data for any file with the same name)       w
Write-only mode; creates a new file, but fails if the file path already exists                                  x
Append to existing file (create the file if it does not already exist)                                                    a
Read and write                                                                                                                                                               r+
Add to mode for binary files (i.e., 'rb' or 'wb' )                                                                                                b
Text mode for files (automatically decoding bytes to Unicode). This is the default if not specified. Add t to other
modes to use this (i.e., 'rt' or 'xt' )                                                                                                                          t
'''

with open('tmp.txt', 'w') as handle:
	handle.writelines(x for x in open(path) if len(x) > 1)
with open('tmp.txt') as f:
	lines = f.readlines()
lines

'''
read([size])                 Return data from file as a string, with optional size argument indicating the number of bytes to read
readlines([size])       Return list of lines in the file, with optional size argument
write(str)                     Write passed string to file
writelines(strings)  Write passed sequence of strings to the file
close()                           Close the handle
flush()                            Flush the internal I/O buffer to disk
seek(pos)                     Move to indicated file position (integer)
tell()                               Return current file position as integer
closed                            True if the file is closed
'''

with open(path) as f:
	chars = f.read(10)
chars

with open(path, 'rb') as f:
	data = f.read(10)
data
data.decode('utf8')
data[:4].decode('utf8')# error Depending on the text encoding, you may be able to decode the bytes to a str object yourself, but only if each of the encoded Unicode characters is fully formed:

'''
Text mode, combined with the encoding option of open , provides a convenient way
to convert from one Unicode encoding to another:
'''
sink_path = 'sink.txt'
with open(path) as source:
	with open(sink_path, 'xt', encoding = 'iso-8859-1') as sink:
		sink.write(source.read())
with open(sink_path, encoding='iso-8859-1') as f:
	print(f.read(10))


#Beware using seek when opening files in any mode other than binary. If the file position falls in the middle of the bytes defining a Unicode character, then subsequent reads will result in an error:
f = open(path)
f.read(5)
f.seek(4)
f.read(1) #error









----------------------------------------NumPy Basics
'''
NumPy's array objects is the lingua franca for data exchange.
-ndarray, an efficient multidimensional array providing fast array-oriented arithmetic operations and flexible broadcasting capabilities.
-Mathematical function for fast operations on entire arrays of data without having to write loops
-Tools for reading/writing array data to disk and working with memory-mapped files
-Linear algebra, random number generation, and Fourier transform capabilities
-A C API for connecting NumPy with libraries written in C, C++, or FORTRAN

The reason why NumPy is so efficient on processing large array of data.
-NumPy internally stores data in a contiguous block of memory, independent of other built-in Python objects.
-NumPy 's alg is written by C, which can operate on this memory without any type-checking or other overhead and less memory usage
-NumPy operations perform complex computation on entire arrays without the need for Python for loops.
'''
#time testing
import numpy as np
my_arr = np.arange(1000000)
my_list = list(range(1000000))
%time for _ in range(10): my_arr2 = my_arr * 2
%time for _ in range(10): my_list2 = [x * 2 for x in my_list]
#the final result shows NumPy-based alg are 10-100 times faster than pure Python counterpartgs and use sigificantly less memory.

import numpy as np
#generate some random data
data = np.random.randn(2, 3)
data

data * 10
data + data

##The numpy namespace is large and contains a number of functions whose names conflict with built-in Python functions (like min and max ).
#An ndarray is a generic multidimensional container for homogeneous data, Every array has a shape, a tuple indicating the size of each dimension and dtype, an object describing the data type of the array

data.shape
data.dtype

#The easiest way to create an array is to use the array function, this accepts any sequence-like object (including other arrays) and produces a new NumPy array containing the passed data.
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1

#Nested sequences, will be converted into multidimensional array
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2

arr2.ndim   #we use this attribute to certify the dimension of this array
arr2.shape

#Unless explicitly specified, np.array tries to infer a good data type for the array that it creates.
#******************This data type is stored in a special dtype metadata object.
arr1.dtype
arr2.dtype


np.zeros(10)   #one dimension
np.zeros((3, 6)) # row 3 colomn 6
np.empty((2, 3, 2)) #******************************It is not safe to assume that np.empty will return an array of all zeros. In some cases, it may return unintialized "garbage" values

np.arange(15) # arange is an array-valued version of the built-in Python range function
''' Array creation functions
array                            Convert input data (list, tuple, array, or other sequence type) to an ndarray either by inferring a dtype or explicitly specifying a dtype; copies the input data by default            
asarray                        Convert input to ndarray, but do not copy if the input is already an ndarray
arange                         Like the built-in range but returns an ndarray instead of a list
ones, ones_like       Produce an array of all 1s with the given shape and dtype; ones_like takes another array and produces a ones array of the same shape and dtype
zeros, zeros+_like  Like ones and ones_like but producing arrays of 0s instead
empty, empty_like Create new arrays by allocating new memory, but do not populate with any values like ones and zeros
full, full_like               Produce an array of the given shape and dtype with all values set to the indicated “fill value”; full_like takes another array and produces a filled array of the same shape and dtype
eye, identity               Create a square N × N identity matrix (1s on the diagonal and 0s elsewhere)
'''
arr1 = np.array([1, 2, 3], dtype = np.float64)
arr2 = np.array([1, 2, 3], dtype = np.int32)
arr1.dtype
arr2.dtype

'''ndarrays 's Data Type
The numerical dtypes are named the same way: a type name, like float or int , followed by a number indicating the number of bits per element.
NumPy data Types table is rather rarely for using, so...
'''
-------------------------casting
arr = np.array([1, 2, 3, 4, 5])
arr.dtype
float_arr = arr.astype(np.float64)
float_arr.dtype

arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
int_arr = arr.astype(np.int32)

numeric_strings = np.array(['1.25', '-9.6', '42'], dtype = np.string_)
numeric_strings.astype(float)
#********************It is important to be cautious when using the numpy.string_ type, as string data in NumPy is fixed size and may truncate input without warning. Pandas has more intuitive out-of-the-box behaviour on non-numeric data.


int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
int_array.astype(dtype(calibers))

empty_uint32 = np.empty(8, dtype='u4')   #u4 is the shorthand of uint32
empty_uint32

arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr
arr * arr
arr - arr
1/arr
arr ** 0.5
#Comparisons between arrays of the same size yield boolean arrays:
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
arr2
arr2 > arr #this will produce a boolean matrix

---------------------------------------------------------------------------------------------broadcasting
'''
Operations between differently sized arrays is called broadcasting
'''
-------------------------------------------------------------------------------------------------------------Basic Indexing and Slicking 
arr = np.arange(10)
arr
arr[5]
arr[5:8]
arr[5:8] = 12
arr

#*****************the most important distinction from Python's built-in lists is that array slices are views on the original array. This means that the data is not copied.
arr_slice  = arr[5:8]
arr_slice
arr_slice[1]  = 12334
arr

#*************The 'bare' slice[:] will assign to all values in an array
arr[:] = 64
arr

#********************if you need a copy of a slice of an ndarray instead of a view
arr[5:8].copy()#very important

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]    #print out the third row.
arr2d[0][2]
#is equivalent with 
arr2d[0, 2]

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d
arr3d[0]
old_value = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_value
arr3d

#indexing with slices
arr
array([0, 1, 2, 3, 4, 64, 64, 64, 8, 9])
arr[1:6]   #output is array([1, 2, 3, 4, 64]), left including except right side
arr2d   #output is as following 
array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])

arr2d[:2]
'''output is 
array([[1, 2, 3],
[4, 5, 6]])'''

#you can pass multiple slices just like you can pass multiple indexes
arr2d[:2, 1:]
'''output
array([[2, 3], [5, 6]])
'''
#I can select the second row but only the first two columns like so
arr2d[1, :2]    #output array([4, 5])
array[:2, 2]     #output array([3, 6])
arr2d[:, :1]       #output  array([[1], [4], [7]])
arr2d[:2, 1:] = 0    #output array([[1, 0, 0], [4, 0, 0], [7, 8, 9]])

----------------------------------------Boolean Indexing
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
names == 'Bob'
#out :array([ True, False, False, True, False, False, False], dtype=bool)
data[names == 'Bob']
#out array([[ 0.0929, 0.2817, 0.769, 1.2464],[ 1.669 , -0.4386, -0.5397, 0.477 ]])
#The above boolean array must be of the same length as the array it's indexing

data[name == 'Bob', 2:]
#out array([[ 0.769 , 1.2464], [-0.5397, 0.477 ]])
data[names == 'Bob', 3]
#out array([ 1.2464, 0.477 ])
names != 'Bob'
data[~(names == 'Bob')]

#the operator ~ can be useful when u wanna invert a general condition
cond = names == 'Bob'
data[~cond]

mask = (names == 'Bob') | (names == 'Will')    #it can use & as well
mask
data[mask]

#to set all of negative values in data to 0
data[data < 0] = 0
data

#setting whole rows or columns using a one-dimentional boolean array is also easy:
data[name!='Joe'] = 7
data

---------------------------------------------Fancy indexing
#fancy indexing is a term adopted by NumPy to describe indexing using integer arrays
arr = np.empty((8, 4))
for i in range(8):
	arr[i] = i

arr[[4, 3, 0, 6]]
'''result
array([
[ 4., 4., 4.,4.],
[ 3., 3., 3.,3.],
[ 0., 0., 0.,0.],
[ 6., 6., 6.,6.]])
'''
arr[[-3, -5, -7]]

arr = np.arange(32).reshape((8, 4))
arr
arr[[1, 5, 7, 2], [0, 3, 1, 2]]#out array([ 4, 23, 29, 10])

#kind of different function to do that
arr[[1, 5, 7, 2]][:[0, 3, 1, 2]]
'''result
array([[ 4, 7, 5, 6],
[20, 23, 21, 22],
[28, 31, 29, 30],
[ 8, 11, 9, 10]])'''

---------------------------------------------transposing arrays and swapping axes
arr = np.arange(15).reshape((3, 5))
arr
arr.T   #transposing the array
'''result
array([[ 0, 5,10],
[ 1, 6,11],
[ 2, 7,12],
[ 3, 8,13],
[ 4, 9,14]])
'''

arr = np.random.randn(6, 3)
arr 
np.dot(arr.T, arr) #to computing the inner matrix product us np.dot

arr = np.arange(16).reshape((2, 2, 4))
arr
'''Output
array([[[ 0, 1, 2, 3],
[ 4, 5, 6, 7]],
[[ 8, 9, 10, 11],
[12, 13, 14, 15]]])
'''
arr.transpose((1, 0, 2))
'''output
array([[[ 0, 1, 2, 3],
[ 8, 9, 10, 11]],
[[ 4, 5, 6, 7],
[12, 13, 14, 15]]])
'''

arr.swapaxes(1, 2)
'''output 
array([[[ 0, 4],
[ 1, 5],
[ 2, 6],
[ 3, 7]],
[[ 8, 12],
[ 9, 13],
[10, 14],
[11, 15]]])
'''
-------------------------------Universal function

arr = np.arange(10)
arr
np.sqrt(arr)
np.exp(arr)

x = np.random.rand(8)
y = np.random.rand(8)
x
y 
np.maximum(x, y)   #find max value and compose to a new matrix

arr = np.random.randn(7) * 5

remainder, whole_part = np.modf(arr)
remainder  #digital part
whole_part #the integer part

#Ufunction accept out argument that allows them to operate in-place on arrays:
arr
np.sqrt(arr)
np.sqrt(arr, arr)

points = np.arange(-5, 5, 0.01) #1000 equally spaced points
xs, ys = np.meshgrid(points, points)
ys

#using natplotlib to create visualizations of this two-dimentional array
import matplotlib.pyplot as plt 
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^ 2}$ for a grid of values")

---------------------------------------------------------------------Expressing conditional logic as array operations
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

result = [(x if c else y)
					for x, y, z in zip(xarr, yarr, cond)]
result
'''The issue which will occur in above
1. when it is encountered with the large arrays, it won't be very fast.(Interpreted Python code)
2. it will not work with multimensional arrays
'''
result = np.where(cond, xarr, yarr)
result
#The second and third arguments to np.where don't need to be arrays, one or both of them can be scalars.
arr = np.random.randn(4, 4)
arr 
arr > 0
np.where(arr > 0, 2, -2)
#or u can combined scalars and arrays when using np.
np.where(arr > 0, 2, arr) #set only positive value to 2

-----------------------------------Mathematical and Statistical method
arr = np.random.randn(5, 4)
arr 
arr.mean()
np.mean(arr)
arr.sum()
arr.mean(axis=1)   #axis is y,,, this will count the value on the x axis
arr.sum(axis = 0)
#arr.mean(1) means “compute mean across the columns” where arr.sum(0)means “compute sum down the rows.”

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr.cumsum() #out array([ 0, 1, 3, 6, 10, 15, 21, 28])

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

#0 vertical, 1 horizontal
arr.cumsum(axis = 0)
arr.cumprod(axis = 1)


















