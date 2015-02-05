from lib2to3.pytree import Leaf, Node
from symbol import *
import sys
from lib2to3.pgen2 import driver
from lib2to3 import pygram, pytree
import re


def walk_class(tree):
    pass

def walk_func(tree):
    #print repr(child)
    func = tree.children
    func_name = func[1].value
    func_args = None
    for fa in func[2].children:
        if fa.type == 334:
            func_args = fa
            break


    args = []
    if func_args is not None:
        func_args = func_args.children
        #print repr(func_args)
        i = 0
        while i < len(func_args):
            if func_args[i].type == 1:
                arg = [func_args[i].value]
                if i<len(func_args)-2 and func_args[i+1].type == 22:
                    if isinstance(func_args[i+2], Leaf):
                        #atom
                        arg.append(func_args[i+2].value)
                    else:
                        #compound
                        arg.append(str(func_args[i+2]))
                    i+=3
                else:
                    i+=2
                args.append(arg)
            else:
                i+=1

    #print '---'
    #print args
    #print '---'

    body = None
    for fa in func:
        if fa.type == 321: #suite
            body = fa
            break

    #print "found body"
    #print repr(body)[:300]
    padding = ''
    first = -1
    for i, b in enumerate(body.children):
        #print i, b.type
        #print repr(b)[:50]
        if b.type < 255:
            if b.type == 5: # padding
                padding += b.value
            first = i
        else:
            break

    #print "colon at", first
    #print "padding '%s'" % padding
    #if colon > -1:
    leaves = [
        Leaf(4, '\n'),
        Leaf(5, padding),
        Leaf(3, '"""'),
        Leaf(4, '\n'),
        Leaf(5, padding),
        Leaf(3, 'Missing documentation'),
        Leaf(4, '\n'),
        Leaf(5, padding),
    ]

    if len(args) > 0:
        if not (len(args) == 1 and (args[0][0] == 'self' or args[0][0] == 'cls')):
            leaves.extend([
                Leaf(4, '\n'),
                Leaf(5, padding),
                Leaf(3, 'Parameters'),
                Leaf(4, '\n'),
                Leaf(5, padding),
                Leaf(3, '----------'),
                Leaf(4, '\n'),
                Leaf(5, padding),
            ])

            for arg in args:
                if arg[0] == 'self' or arg[0] == 'cls':
                    continue
                p_def = ''
                pd = None
                if len(arg)==2:
                    pd = arg[1]
                    p_def = ' (default %s)' % arg[1]

                t = 'Type'
                if pd:
                    if pd == 'True' or pd == 'False':
                        t = 'bool'
                    if (pd[0] == "'" or pd[0] == '"') and pd[-1] == pd[0]:
                        t = 'str'
                    if re.match(r'[+\-0-9]+', pd):
                        t = 'int'
                    elif re.match(r'[+\-0-9e\.]+', pd):
                        t = 'float'

                p = '%s : %s' % (arg[0], t)
                if len(arg)==2:
                    p+=', optional'
                leaves.extend([
                    Leaf(3, p),
                    Leaf(4, '\n'),
                    Leaf(5, padding),
                    Leaf(5, '    '),
                    Leaf(3, 'Description'+p_def),
                    Leaf(4, '\n'),
                    Leaf(5, padding),
                ])

    leaves.extend([
        Leaf(4, '\n'),
        Leaf(5, padding),
        Leaf(3, 'Returns'),
        Leaf(4, '\n'),
        Leaf(5, padding),
        Leaf(3, '-------'),
        Leaf(4, '\n'),
        Leaf(5, padding),
        Leaf(3, 'Value : Type'),
        Leaf(4, '\n'),
        Leaf(5, padding),
        Leaf(5, '    '),
        Leaf(3, 'Description'),
        Leaf(4, '\n'),
        Leaf(5, padding),
        Leaf(3, '"""'),
    ])

    n = Node(313, leaves) #simple_stmt
    body.children.insert(0, n)
    #print "inserted"

def walk_tree(tree):
    for child in tree.children:
        #print child.type
        #print repr(child)[:60]
        if child.type == 266: #classdef
            #print "classdef"
            walk_tree(child)
            walk_class(child)
        if child.type == 274: #decorated
            walk_tree(child)
        if child.type == 321: #suite
            #print "suite"
            walk_tree(child)
        if child.type == 292: #funcdef
            #print "funcdef"
            walk_func(child)


# To reconstruct a file from AST:
# print str(tree)
# 313 - existing docstring
# 292 - funcdef
# 266 - class definion
# 321 - suite (class, function content)


import glob
import os
path = sys.argv[1]
for fn in glob.glob(os.path.join(path, '*.py')):
    print 'processing %s' % fn
    print 'reading...'
    with open(fn, 'r') as f:
        contents = f.read()

    print 'parsing...'
    drv = driver.Driver(pygram.python_grammar, pytree.convert)
    tree = drv.parse_string(contents, True)
    walk_tree(tree)

    print 'writing ...'
    with open(fn, 'w') as f:
        f.write(str(tree))

