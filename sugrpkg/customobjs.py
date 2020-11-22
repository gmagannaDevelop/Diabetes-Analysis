class objdict(dict):
    """
    Instantiate a dictionary that allows accesing values
    with object notation (as if they were attributes):

    ex:
        x.foo = 5
    instead of
        x['foo'] = 5

    The best part is that both ways work !

    Ideal for working with TOML files.

    Original code snippet found here :
    https://goodcode.io/articles/python-dict-object/
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


##
