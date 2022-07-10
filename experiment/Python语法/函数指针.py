a = list(range(10))


def fun(a, b, c):
    return a + b + c


a.sort(key=lambda x: fun(1, 2, x))  # 对的
# a.sort(key=fun(1, 2))  # 错的
print(a)



def a(x):
    return 1

# (a+1)(1)   # 错

print((lambda x: a(x) + 1)(1))   #对

