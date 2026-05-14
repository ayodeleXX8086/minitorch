from minitorch import Scalar

if __name__ == "__main__":
    # a = Scalar(1)
    # b = a*a*a
    # b.backward()
    # print(b.derivative)
    # print(a.derivative)
    c=Scalar(1)
    d = c*c
    d.backward()
    print(c.derivative)
    print(d.derivative)