import torch
from computational_graph import CG



def graph_1() -> CG:
    g = CG()

    ################################################################
    # TODO
    g.input("x", id = "x", forward=1, backward=16)
    g.input("y", id = "y", forward = 5, backward = 3)
    g.input("z", id = "z", forward = 2, backward = -.41)

    g.add_node("x/2", id = "x1", input_nodes=["x"], forward = .5, backward = 2)
    g.add_node("ln(x/2)", id = "x2", input_nodes=["x1"], forward = -.69, backward = 1)
    g.add_node("x*y", id = "x3", input_nodes=["x", "y"], forward = 5, backward = 3)
    g.add_node("3(x*y)", id = "x4", input_nodes=["x3"], forward = 15, backward = 1)
    g.add_node("sin(z)", id = "x5", input_nodes=["z"], forward = .9, backward = 1)
    g.add_node("sin(z) + 3(x*y)", id = "x6", input_nodes=["x4", "x5"], forward = .9, backward = 1)
    
    g.output("ln(x/2) + 3xy + sin(z)", id = "x7", input_nodes=["x2", "x6"], forward=15.21, backward=1)
    ################################################################

    return g



def graph_2() -> CG:
    g = CG()

    ################################################################
    # TODO
   
    g.input("x", id = "x", forward = 2, backward = 0)
    g.input("y", id = "y", forward = 3, backward = 0)
    g.input("z", id = "z", forward = 4, backward = .76)

    g.add_node("x*y", id = "x1", input_nodes=["x", "y"], forward = 6, backward = 0)
    g.add_node("z+x*y", id = "x2", input_nodes=["x1", "z"], forward = 10, backward = 0)
    g.add_node("tanh()", id = "x3", input_nodes=["x2"], forward = 1, backward = -.65)
    g.add_node("cos(z)", id = "x4", input_nodes=["z"], forward = -.65, backward = 1)
    
    g.output("*", id = "x5", input_nodes=["x3", "x4"], forward = -.65, backward = 1)

    ################################################################

    return g



def graph_3() -> CG:
    g = CG()

    ################################################################
    # TODO
    
    g.input("V", id = "V", forward = torch.tensor([[1, .5, .2]]), backward = torch.tensor([[.57, .66, .75]]))
    g.input("W", id = "W", forward = torch.tensor([[.1, .05], [.25, .15], [.1, .4]]), backward = torch.tensor([[.25, .5], [.11, .22], [.04, .08]]))
    g.input("x", id = "x", forward = torch.tensor([[1], [2]]), backward = torch.tensor([[0.056], [.046]]))
    g.input("b", id = "b", forward = torch.tensor([[.1], [.15], [.2]]), backward = torch.tensor([[.25], [.11], [.04]]))

    g.add_node("Wx", id = "x1",input_nodes= ["W", "x"], forward = torch.tensor([[.2], [.55], [.2]]), backward = torch.tensor([[.25], [.11], [.04]]))
    g.add_node("Wx+b", id = "x2",input_nodes= ["x1", "b"], forward = torch.tensor([[.3], [.7], [1.1]]), backward = torch.tensor([[.25], [.11], [.04]]))
    g.add_node("sigma(Wx+b)", id = "x3",input_nodes= ["x2"], forward = torch.tensor([[.57], [.66], [.75]]), backward = torch.tensor([[1], [.5], [.2]]))

    g.output("v*sigma(Wx+b)", id = "x4",input_nodes= ["x3", "V"], forward = torch.tensor([1.05]), backward = torch.tensor([1]))


    ################################################################

    return g



def graph_4() -> CG:
    g = CG()

    ################################################################
    # TODO
    g.input("x", id = "x", forward = torch.tensor([[1], [2]]), backward = torch.tensor([[-.45], [-.18]]))
    g.input("W", id = "W", forward = torch.tensor([[.5, .25], [.2, -.4]]), backward = torch.tensor([[-.9, 0], [-1.8, 0]]))
    g.input("V", id = "V", forward = torch.tensor([[3], [2]]), backward = torch.tensor([[-.27], [0]]))
    g.input("y", id = "y", forward = torch.tensor([3]), backward = torch.tensor([.3]))
    
    g.add_node("xT", id = "x1", input_nodes=["x"], forward = torch.tensor([[1, 2]]), backward = torch.tensor([[-.45, -.18]]))
    g.add_node("xTW", id = "x2", input_nodes=["x1", "W"], forward = torch.tensor([[.9, -.55]]), backward = torch.tensor([[-.9, 0]]))
    g.add_node("ReLU(xTW)", id = "x3", input_nodes=["x2"], forward = torch.tensor([[.9, 0]]), backward = torch.tensor([[-.9, -.6]]))
    g.add_node("ReLU(xTW)*V", id = "x4", input_nodes=["x3", "V"], forward = torch.tensor([2.7]), backward = torch.tensor([-.3]))
    g.add_node("y-ReLU(xTW)*V", id = "x5", input_nodes=["x4", "y"], forward = torch.tensor([.3]), backward = torch.tensor([.3]))

    g.output("out", id = "x6", input_nodes=["x5"], forward = torch.tensor([.0045]), backward = torch.tensor([1]))

    ################################################################

    return g
