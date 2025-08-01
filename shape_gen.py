import math

'''
# shape_butt
radius = 1000
a = []
with open('shape.txt', 'w') as f:
    for i in range(0, int(1.6*radius + 1)):
        r = math.sqrt(radius**2 - (i - radius)**2)
        a.append(r)
        f.write(f"{r}\n")
    for r in a[::-1]:
        f.write(f"{r}\n")
'''

# shape rod
radius = 150
length = 3000
a = []
with open('shape.txt', 'w') as f:
    for i in range(0, radius):
        a.append(math.sqrt(radius**2 - (i - radius)**2))
    for i in range(radius, length + radius):
        a.append(radius)
    for i in range(0, radius+1):
        a.append(math.sqrt(radius**2 - i**2))

    for r in a:
        f.write(f"{r}\n")