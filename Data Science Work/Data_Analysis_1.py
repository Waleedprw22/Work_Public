import numpy as np
import requests

url = (
"https://raw.githubusercontent.com/changyaochen/MECE4520/"
"master/lectures/week-1/random_numbers.txt"
)

response = requests.get(url)
values = [int(x.strip()) for x in response.text.split("\n") if len(x) > 0]

##print(values)
count = 0
for i in range(0,len(values)):
    for j in range(1,len(values)):
        if values[i] + values[j] == 5000:
            count = count + 1
    values[i] = 0


print(count)