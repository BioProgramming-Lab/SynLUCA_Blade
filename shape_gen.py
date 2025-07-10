import math

radius = 1000
a = []
with open('shape.txt', 'w') as f:
    for i in range(0, int(1.6*radius + 1)):
        r = math.sqrt(radius**2 - (i - radius)**2)
        a.append(r)
        f.write(f"{r}\n")
    for r in a[::-1]:
        f.write(f"{r}\n")   