import toml

data = {'t': [0, 1, 2, 3, 4, 5, 6, 7],
        'y': [0, 1, 8, 27, 64, 65, 66, 66]}

with open("config.toml", "w") as f:
    toml.dump(data, f)

'''data = toml.load("config.toml")
print(data)'''