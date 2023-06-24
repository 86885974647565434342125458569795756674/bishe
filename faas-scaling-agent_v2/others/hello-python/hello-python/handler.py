def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    val = 0
    for i in range(1000):
        val += i
    print('1+2+3+...+999=', val)
    return req


