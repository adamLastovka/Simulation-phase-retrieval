import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython.display import  display, HTML
import time


def draw_graph(start, watch=[]):
    from graphviz import Digraph

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    graph = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    assert (hasattr(start, "grad_fn"))
    if start.grad_fn is not None:
        _draw_graph(start.grad_fn, graph, watch=watch)

    size_per_element = 0.15
    min_size = 12

    # Get the approximate number of nodes and edges
    num_rows = len(graph.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    graph.graph_attr.update(size=size_str)
    graph.render(filename='net_graph.jpg')


def _draw_graph(var, graph, watch=[], seen=[], indent="", pobj=None):
    ''' recursive function going through the hierarchical graph printing off
    what we need to see what autograd is doing.'''
    from rich import print

    if hasattr(var, "next_functions"):
        for fun in var.next_functions:
            joy = fun[0]
            if joy is not None:
                if joy not in seen:
                    label = str(type(joy)).replace(
                        "class", "").replace("'", "").replace(" ", "")
                    label_graph = label
                    colour_graph = ""
                    seen.append(joy)

                    if hasattr(joy, 'variable'):
                        happy = joy.variable
                        if happy.is_leaf:
                            label += " \U0001F343"
                            colour_graph = "green"

                            for (name, obj) in watch:
                                if obj is happy:
                                    label += " \U000023E9 " + \
                                             "[b][u][color=#FF00FF]" + name + \
                                             "[/color][/u][/b]"
                                    label_graph += name

                                    colour_graph = "blue"
                                    break

                                vv = [str(obj.shape[x]) for x in range(len(obj.shape))]
                                label += " [["
                                label += ', '.join(vv)
                                label += "]]"
                                label += " " + str(happy.var())

                    graph.node(str(joy), label_graph, fillcolor=colour_graph)
                    print(indent + label)
                    _draw_graph(joy, graph, watch, seen, indent + ".", joy)
                    if pobj is not None:
                        graph.edge(str(pobj), str(joy))

class ElementwiseLinear(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(ElementwiseLinear, self).__init__()

        # w is the learnable weight of this layer module
        self.w = nn.Parameter(torch.ones(input_size), requires_grad=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # simple elementwise multiplication
        return self.w * x


class Net(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        # self.L1 = ElementwiseLinear(image_size)
        self.weights = nn.Parameter(torch.ones(image_size), requires_grad=True)

    def forward(self, x):
        # x = self.L1.forward(x)
        return x * self.weights

def step(net, sim, current_mask, target_intensity, loss_func, optimizer, device):
    net.train()

    updated_mask = net(current_mask.detach())

    sim.embed_mask(updated_mask[0])
    sim.propagate_forward()
    image_intensity = sim.extract_intensity()
    # image_intensity = torch.log(image_intensity)
    image_intensity = image_intensity/torch.max(image_intensity)  # scale/normalize?

    # fig,ax = plt.subplots(1,2)
    # ax[0].matshow(image_intensity.detach().numpy())
    # ax[1].matshow(target_intensity[0].detach().numpy())
    # plt.show()

    loss = loss_func(image_intensity, target_intensity[0])

    optimizer.zero_grad()
    loss.backward(create_graph=True)  # retain_graph=True
    optimizer.step()

    # draw_graph(loss)

    return loss.detach(), updated_mask.detach(), image_intensity.detach()


def L4Norm(output,target):
    loss = torch.mean((output-target)**4)
    return loss

def train_net(sim, input_mask, target_intensity, config):
    input_mask = transforms.functional.to_tensor(input_mask)
    target_intensity = transforms.functional.to_tensor(target_intensity)

    # instantiate net
    net = Net((1024, 1280))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_func = L4Norm  #nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config["momentum"])

    train_losses = []
    mask_buffer = []
    output_buffer = []
    for iteration in range(config["num_iterations"]):  # loop over the dataset multiple times
        start = time.time()
        train_loss, updated_mask, updated_output = step(net, sim, input_mask, target_intensity, loss_func, optimizer, device)
        end = time.time()
        print(f'[Epoch:{iteration + 1}] loss: {train_loss:.5f} time:{end-start:.2f}')

        train_losses.append(train_loss)  # Log losses

        if iteration % 10 == 0:  # Log masks
            mask_buffer.append(updated_mask)
            output_buffer.append(updated_output)
    print('Finished Training')

    plot_result(input_mask, output_buffer[0], updated_mask, updated_output)

    # plot_optim_progression(mask_buffer, output_buffer)

    return net, train_losses


def plot_training_losses(train_losses):
    plt.figure(7)
    plt.plot(train_losses, '-bo')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(linestyle='--', linewidth=1)
    plt.show()

def plot_result(initial_mask, initial_output, optimized_mask, optimized_output):
    fig1, ax1 = plt.subplots(1, 2, num=1)
    plt.suptitle("Optimization Results - mask")
    ax1[0].matshow(initial_mask[0], cmap='jet')
    ax1[1].matshow(optimized_mask[0], cmap='jet')
    ax1[0].set_xlabel("Initial")
    ax1[1].set_xlabel("Optimized")
    for axi in ax1:
        axi.set_xticks([])
        axi.set_yticks([])
    # plt.tight_layout(pad=0.2)

    time.sleep(0.2)

    fig2, ax2 = plt.subplots(1, 2, num=2)
    plt.suptitle("Optimization Results - Intensity")
    ax2[0].matshow(initial_output, cmap='jet')
    ax2[1].matshow(optimized_output, cmap='jet')
    ax2[0].set_xlabel("Initial")
    ax2[1].set_xlabel("Optimized")
    for axi in ax2:
        axi.set_xticks([])
        axi.set_yticks([])
    # plt.tight_layout(pad=0.2)

    mask_correction = optimized_mask[0] - initial_mask[0]
    plt.matshow(mask_correction)
    plt.title('Mask changes')
    plt.xlabel(f"Max:{torch.max(mask_correction).numpy():.2e} Min:{torch.min(mask_correction).numpy():.2e}")

    intensity_change = optimized_output - initial_output
    plt.matshow(intensity_change)
    plt.title('Intensity changes')
    plt.xlabel(f"Max:{torch.max(intensity_change).numpy():.2e} Min:{torch.min(intensity_change).numpy():.2e}"
               f"\nAvg:{torch.mean(intensity_change).numpy():.2e}")

def plot_optim_progression(mask_buffer, output_buffer):
    fig = plt.figure(5)
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(img.numpy(),(1,2,0)), animated=True)] for img in mask_buffer]
    ani1 = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    fig = plt.figure(6)
    plt.axis("off")
    ims = [[plt.imshow(img.numpy(), animated=True)] for img in output_buffer]
    ani2 = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # display(HTML(ani1.to_jshtml()))
    # display(HTML(ani2.to_jshtml()))
