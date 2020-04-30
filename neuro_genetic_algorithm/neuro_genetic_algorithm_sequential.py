import numpy as np
import matplotlib.pyplot as plt

from pySimulator.simulators import Simulator
from pySimulator.nodes import InputTrain, LIF
from pySimulator.connections import Synapse
from pySimulator.networks import Network
from pySimulator.detectors import Raster, Multimeter


def seq_onemax(seq_len, name='e0'):
    """Creates a rone-max evaluation ensemble and returns lists of neurons and internal
    connections. The first two elements in the neuron list are the input neurons
    of the [0] top and the [1] bottom lanes. The last three elements in the list are the
    output neurons of the [-3] top, [-2] bottom lanes and [-1] swap y/n evaluation.
    Assumes LIF default parameters
    m=0., V_init=0., V_reset=0., V_min=0., thr=0.99, amplitude=1.,
    I_e=0., refrac_time=0, noise=0. in the pySimulator.

    :param seq_len: (int) length of the sequence excluding lead bit, used for random crossover point
    :param name: (string) name used as a prefix for the neuron names
    :return: (list, list) neurons, connections
    """
    neurons = []
    connections = []

    # Define ensemble neurons
    neurons.append(LIF(name='%s_in_t' % name))                                          # 0
    neurons.append(LIF(name='%s_in_b' % name))                                          # 1
    neurons.append(LIF(thr=1.99, amplitude=seq_len, refrac_time=seq_len, name='%s_act' % name))  # 2
    thr = seq_len + 0.99
    neurons.append(LIF(m=1., thr=thr, V_min=-seq_len, name='%s_acc' % name))            # 3
    amp = 2 * seq_len + 1
    neurons.append(LIF(amplitude=amp, name='%s_reset' % name))                          # 4
    neurons.append(LIF(name='%s_out_t' % name))                                         # 5
    neurons.append(LIF(name='%s_out_b' % name))                                         # 6
    neurons.append(LIF(name='%s_out_s' % name))                                         # 7

    # Define ensemble connections
    d = seq_len + 2
    connections.append(Synapse(neurons[0], neurons[-3], w=1., d=d))         # in top - out top
    connections.append(Synapse(neurons[1], neurons[-2], w=1., d=d))         # in bottom - out bottom
    connections.append(Synapse(neurons[0], neurons[2], w=1., d=1))          # in top - activation
    connections.append(Synapse(neurons[1], neurons[2], w=1., d=1))          # in bottom - activation
    connections.append(Synapse(neurons[0], neurons[3], w=-1., d=1))         # in top - accumulator
    connections.append(Synapse(neurons[1], neurons[3], w=1., d=1))          # in bottom - accumulator
    connections.append(Synapse(neurons[2], neurons[3], w=1., d=seq_len))    # activation - accumulator
    d = seq_len + 1
    connections.append(Synapse(neurons[2], neurons[4], w=1., d=d))          # activation - reset
    connections.append(Synapse(neurons[3], neurons[-1], w=1., d=1))         # accumulator - out score
    connections.append(Synapse(neurons[4], neurons[3], w=1., d=1))          # reset - accumulator
    connections.append(Synapse(neurons[4], neurons[-1], w=-1., d=2))        # reset - out score

    return neurons, connections


def seq_bubblesort(seq_len, name='s0'):
    """Creates a bubblesort ensemble and returns lists of neurons and internal
    connections. The first three elements in the neuron list are the input neurons
    of the [0] top, [1] bottom lanes and [2] score switch y/n. The last two elements in the list are the
    output neurons of the [-2] top and [-1] bottom lanes. Assumes LIF default parameters
    m=0., V_init=0., V_reset=0., V_min=0., thr=0.99, amplitude=1.,
    I_e=0., refrac_time=0, noise=0. in the pySimulator.

    :param seq_len: (int) length of the sequence excluding lead bit, used for random crossover point
    :param name: (string) name used as a prefix for the neuron names
    :return: (list, list) neurons, connections
    """
    neurons = []
    connections = []

    # Define ensemble neurons (t=top, b=bottom, i=identity, s=switch)
    neurons.append(LIF(name='%s_in_t' % name))                              # 0
    neurons.append(LIF(name='%s_in_b' % name))                              # 1
    neurons.append(LIF(refrac_time=seq_len, name='%s_in_s_gca' % name))     # 2
    neurons.append(LIF(name='%s_gate_it' % name))                           # 3
    neurons.append(LIF(thr=1.99, name='%s_gate_st' % name))                 # 4
    neurons.append(LIF(name='%s_gate_ib' % name))                           # 5
    neurons.append(LIF(thr=1.99, name='%s_gate_sb' % name))                 # 6
    neurons.append(LIF(name='%s_gate_control' % name))                      # 7
    neurons.append(LIF(name='%s_out_t' % name))                             # 8
    neurons.append(LIF(name='%s_out_b' % name))                             # 9

    # Define in/out and lane gate connections (identity and switch)
    connections.append(Synapse(neurons[0], neurons[3], w=1., d=2))      # in top - idn gate top
    connections.append(Synapse(neurons[3], neurons[-2], w=1., d=1))     # idn gate top - out top
    connections.append(Synapse(neurons[0], neurons[4], w=1., d=2))      # in top - switch gate top
    connections.append(Synapse(neurons[4], neurons[-1], w=1., d=1))     # switch gate top - out bottom
    connections.append(Synapse(neurons[1], neurons[5], w=1., d=2))      # in bottom - idn gate bottom
    connections.append(Synapse(neurons[5], neurons[-1], w=1., d=1))     # idn gate bottom - out bottom
    connections.append(Synapse(neurons[1], neurons[6], w=1., d=2))      # in bottom - switch gate bottom
    connections.append(Synapse(neurons[6], neurons[-2], w=1., d=1))     # switch gate bottom - out top

    # Define gate control connections
    connections.append(Synapse(neurons[7], neurons[3], w=-1., d=1))     # gate con - gate idn top
    connections.append(Synapse(neurons[7], neurons[5], w=-1., d=1))     # gate con - gate idn bottom
    connections.append(Synapse(neurons[7], neurons[4], w=1., d=1))      # gate con - gate swi top
    connections.append(Synapse(neurons[7], neurons[6], w=1., d=1))      # gate con - gate swi bottom
    connections.append(Synapse(neurons[7], neurons[7], w=1., d=1))      # gate con recurrent

    # Define score input / gate control activation connections
    connections.append(Synapse(neurons[2], neurons[7], w=1., d=1))      # in switch/gca - gate control a
    d = seq_len + 2
    connections.append(Synapse(neurons[2], neurons[7], w=-1., d=d))     # in switch/gca - gate control b

    return neurons, connections


def seq_crossover(seq_len, name='c0'):
    """Creates a random-point crossover ensemble and returns lists of neurons and internal
    connections. The first two elements in the neuron list are the input neurons
    of the [0] top and the [1] bottom lanes. The last two elements in the list are the
    output neurons of the [-2] top and [-1] bottom lanes. Assumes LIF default parameters
    m=0., V_init=0., V_reset=0., V_min=0., thr=0.99, amplitude=1.,
    I_e=0., refrac_time=0, noise=0. in the pySimulator.

    :param seq_len: (int) length of the sequence excluding lead bit, used for random crossover point
    :param name: (string) name used as a prefix for the neuron names
    :return: (list, list) neurons, connections
    """
    neurons = []
    connections = []

    # Define ensemble neurons (t=top, b=bottom, i=identity, s=switch)
    neurons.append(LIF(name='%s_in_t' % name))                                          # 0
    neurons.append(LIF(name='%s_in_b' % name))                                          # 1
    neurons.append(LIF(thr=1.99, name='%s_gate_it' % name))                             # 2
    neurons.append(LIF(thr=1.99, name='%s_gate_st' % name))                             # 3
    neurons.append(LIF(thr=1.99, name='%s_gate_ib' % name))                             # 4
    neurons.append(LIF(thr=1.99, name='%s_gate_sb' % name))                             # 5
    neurons.append(LIF(name='%s_gate_control_i' % name))                                # 6
    neurons.append(LIF(name='%s_gate_control_s' % name))                                # 7
    neurons.append(LIF(thr=1.99, refrac_time=seq_len, name='%s_control_act' % name))    # 8
    p = 1 / seq_len
    neurons.append(LIF(thr=1.99 - p, noise=1., name='%s_stoch' % name))                 # 9
    neurons.append(LIF(name='%s_stoch_control' % name))                                 # 10
    neurons.append(LIF(name='%s_out_t' % name))                                         # 11
    neurons.append(LIF(name='%s_out_b' % name))                                         # 12

    # Define in/out and lane gate connections (identity and switch)
    connections.append(Synapse(neurons[0], neurons[2], w=1., d=3))      # in top - idn gate top
    connections.append(Synapse(neurons[2], neurons[-2], w=1., d=1))     # idn gate top - out top
    connections.append(Synapse(neurons[0], neurons[3], w=1., d=3))      # in top - switch gate top
    connections.append(Synapse(neurons[3], neurons[-1], w=1., d=1))     # switch gate top - out bottom
    connections.append(Synapse(neurons[1], neurons[4], w=1., d=3))      # in bottom - idn gate bottom
    connections.append(Synapse(neurons[4], neurons[-1], w=1., d=1))     # idn gate bottom - out bottom
    connections.append(Synapse(neurons[1], neurons[5], w=1., d=3))      # in bottom - switch gate bottom
    connections.append(Synapse(neurons[5], neurons[-2], w=1., d=1))     # switch gate bottom - out top

    # Define identity gate control connections
    connections.append(Synapse(neurons[6], neurons[2], w=1., d=1))      # gate con idn - gate idn top
    connections.append(Synapse(neurons[6], neurons[4], w=1., d=1))      # gate con idn - gate idn bottom
    connections.append(Synapse(neurons[6], neurons[6], w=1., d=1))      # gate con idn recurrent

    # Define switch gate control connections
    connections.append(Synapse(neurons[7], neurons[3], w=1., d=1))      # gate con swi - gate swi top
    connections.append(Synapse(neurons[7], neurons[5], w=1., d=1))      # gate con swi - gate swi bottom
    connections.append(Synapse(neurons[7], neurons[7], w=1., d=1))      # gate con swi recurrent

    # Define gate control activation connections
    connections.append(Synapse(neurons[0], neurons[8], w=1., d=1))      # in top - gate control act
    connections.append(Synapse(neurons[1], neurons[8], w=1., d=1))      # in bottom - gate control act
    connections.append(Synapse(neurons[8], neurons[6], w=1., d=1))      # gate con act - gate con idn a
    d = seq_len + 2
    connections.append(Synapse(neurons[8], neurons[6], w=-1., d=d))     # gate con act - gate con idn b
    connections.append(Synapse(neurons[8], neurons[7], w=-1., d=d))     # gate con act - gate con swi
    connections.append(Synapse(neurons[8], neurons[9], w=1., d=1))      # gate con act - stochastic
    connections.append(Synapse(neurons[8], neurons[10], w=1., d=1))     # gate con act - stoch con a
    d = seq_len
    connections.append(Synapse(neurons[8], neurons[10], w=-1., d=d))    # gate con act - stoch con b

    # Define stochastic neuron connections
    connections.append(Synapse(neurons[9], neurons[6], w=-1., d=1))     # stoch - gate con idn
    connections.append(Synapse(neurons[9], neurons[7], w=1., d=1))      # stoch - gate con swi
    connections.append(Synapse(neurons[9], neurons[10], w=-1., d=1))    # stoch - stoch control

    # Define stochastic control neuron connections
    connections.append(Synapse(neurons[10], neurons[9], w=1., d=1))     # stoch control - stoch
    connections.append(Synapse(neurons[10], neurons[10], w=1., d=1))    # stoch control recurrent

    return neurons, connections


def seq_mutation(seq_len, p, name='m0'):
    """Creates a mutation ensemble and returns lists of neurons and internal
    connections. The first element in the neuron list is the input neuron and
    the last element in the list is the output neuron. Assumes LIF default
    parameters m=0., V_init=0., V_reset=0., V_min=0., thr=0.99, amplitude=1.,
    I_e=0., refrac_time=0, noise=0. in the pySimulator.

    :param seq_len: (int) length of the sequence excluding lead bit
    :param p: (float) probability applied to the mutation of each bit
    :param name: (string) name used as a prefix for the neuron names
    :return: (list, list) neurons, connections
    """
    neurons = []
    connections = []

    # Define ensemble neurons
    neurons.append(LIF(name='%s_in' % name))
    neurons.append(LIF(thr=1.99 - p, noise=1., name='%s_stoch1' % name))
    neurons.append(LIF(thr=1.99 - p, noise=1., name='%s_stoch2' % name))
    neurons.append(LIF(name='%s_control' % name))
    neurons.append(LIF(refrac_time=seq_len, name='%s_control_act' % name))
    neurons.append(LIF(name='%s_out' % name))

    # Define ensemble connections
    connections.append(Synapse(neurons[0], neurons[1], w=1., d=3))
    connections.append(Synapse(neurons[0], neurons[2], w=-1., d=3))
    connections.append(Synapse(neurons[0], neurons[4], w=1., d=1))
    connections.append(Synapse(neurons[0], neurons[5], w=1., d=4))
    connections.append(Synapse(neurons[1], neurons[5], w=-1., d=1))
    connections.append(Synapse(neurons[2], neurons[5], w=1., d=1))
    connections.append(Synapse(neurons[3], neurons[2], w=1., d=1))
    connections.append(Synapse(neurons[3], neurons[3], w=1., d=1))
    connections.append(Synapse(neurons[4], neurons[1], w=-1., d=2))
    connections.append(Synapse(neurons[4], neurons[2], w=-1., d=2))
    connections.append(Synapse(neurons[4], neurons[3], w=1., d=1))
    connections.append(Synapse(neurons[4], neurons[3], w=-1., d=seq_len + 2))

    return neurons, connections


def stack_evaluation(n_chromosomes, seq_len):
    """Creates multiple ensembles. Returns the neurons as a list of lists.
    Each ensemble remains a separate list in the list of lists to allow for better indexing
    when making connections to the next layer of ensembles (e.g. bubblesort).
    :param n_chromosomes: (int) the population size. Must be even.
    :param seq_len: (int) length of the sequence excluding lead bit
    :return: (list, list) neurons as list of lists, connections as list
    """
    assert n_chromosomes % 2 == 0, "n_chromosomes must be even, got %d" % n_chromosomes
    neurons = []
    connections = []
    for i in range(n_chromosomes // 2):
        n, c = seq_onemax(seq_len, name='e%d' % i)
        neurons.append(n)
        connections.extend(c)
    return neurons, connections


def stack_bubblesort(n_chromosomes, seq_len):
    """Creates multiple ensembles. Returns the neurons as a list of lists.
    Each ensemble remains a separate list in the list of lists to allow for better indexing
    when making connections to the next layer of ensembles (e.g. crossover).
    :param n_chromosomes: (int) the population size. Must be even.
    :param seq_len: (int) length of the sequence excluding lead bit
    :return: (list, list) neurons as list of lists, connections as list
    """
    assert n_chromosomes % 2 == 0, "n_chromosomes must be even, got %d" % n_chromosomes
    neurons = []
    connections = []
    for i in range(n_chromosomes // 2):
        n, c = seq_bubblesort(seq_len, name='s%d' % i)
        neurons.append(n)
        connections.extend(c)
    return neurons, connections


def stack_crossover(n_chromosomes, seq_len):
    """Creates multiple ensembles. Returns the neurons as a list of lists.
    Each ensemble remains a separate list in the list of lists to allow for better indexing
    when making connections to the next layer of ensembles (e.g. mutation).
    :param n_chromosomes: (int) the population size. Must be even.
    :param seq_len: (int) length of the sequence excluding lead bit
    :return: (list, list) neurons as list of lists, connections as list
    """
    assert n_chromosomes % 2 == 0, "n_chromosomes must be even, got %d" % n_chromosomes
    neurons = []
    connections = []
    for i in range(n_chromosomes // 2):
        n, c = seq_crossover(seq_len, name='c%d' % i)
        neurons.append(n)
        connections.extend(c)
    return neurons, connections


def stack_mutation(n_chromosomes, seq_len, base_p, scaling_mutation=False):
    """Creates multiple ensembles. Returns the neurons as a list of lists.
    Each ensemble remains a separate list in the list of lists to allow for better indexing
    when making connections to the next layer of ensembles (e.g. evaluation).
    :param n_chromosomes: (int) the population size. Must be even.
    :param seq_len: (int) length of the sequence excluding lead bit
    :param base_p: (float) mutation probability for each bit
    :param scaling_mutation: (boolean) scales the mutation probability over the fitness hierarchy: [0, base_p]
    :return: (list, list) neurons as list of lists, connections as list
    """
    neurons = []
    connections = []
    for i in range(n_chromosomes):
        if scaling_mutation:
            p = base_p * i / (n_chromosomes - 1)  # linear range [0, base_p] over fitness hierarchy
        else:
            p = base_p
        n, c = seq_mutation(seq_len, p, name='m%d' % i)
        neurons.append(n)
        connections.extend(c)
    return neurons, connections


def connect_inputs_eval(nodes_input, neurons_eval):
    """Connects a list of input nodes to a list of lists of ensemble neurons.
    :param nodes_input: (list) list of input nodes
    :param neurons_eval: (list) list of lists of ensemble neurons
    :return: (list) connections
    """
    assert len(neurons_eval) >= 1, "number of ensembles must be >= 1, got n=%d" % len(neurons_eval)
    assert len(nodes_input) == 2 * len(neurons_eval),\
        "number of ensembles must match 2:1, got n_inputs=%d, n_eval=%d" % \
        (len(nodes_input), len(neurons_eval))
    connections = []
    # Create regular connections
    for i in range(len(neurons_eval)):
        connections.append(Synapse(nodes_input[i * 2], neurons_eval[i][0], w=1., d=1))
        connections.append(Synapse(nodes_input[i * 2 + 1], neurons_eval[i][1], w=1., d=1))
    return connections


def connect_eval_sort(neurons_eval, neurons_sort):
    """Connects a list of lists of ensemble neurons to another list of lists of ensemble neurons.
    :param neurons_eval: (list) list of lists of ensemble neurons
    :param neurons_sort: (list) list of lists of ensemble neurons
    :return: (list) connections
    """
    assert len(neurons_eval) >= 1, "number of ensembles must be >= 1, got n=%d" % len(neurons_eval)
    assert len(neurons_eval) == len(neurons_sort),\
        "number of ensembles must match 1:1, got n_eval=%d, n_sort=%d" % \
        (len(neurons_eval), len(neurons_sort))
    connections = []
    # Create regular connections
    for i in range(len(neurons_eval)):
        connections.append(Synapse(neurons_eval[i][-3], neurons_sort[i][0], w=1., d=1))
        connections.append(Synapse(neurons_eval[i][-2], neurons_sort[i][1], w=1., d=1))
        connections.append(Synapse(neurons_eval[i][-1], neurons_sort[i][2], w=1., d=1))
    return connections


def connect_sort_cross(neurons_sort, neurons_cross):
    """Connects a list of lists of ensemble neurons to another list of lists of ensemble neurons.
    :param neurons_sort: (list) list of lists of ensemble neurons
    :param neurons_cross: (list) list of lists of ensemble neurons
    :return: (list) connections
    """
    assert len(neurons_sort) >= 3, "number of ensembles must be >= 3, got n=%d" % len(neurons_sort)
    assert len(neurons_sort) == len(neurons_cross),\
        "number of ensembles must match 1:1, got n_sort=%d, n_cross=%d" % \
        (len(neurons_sort), len(neurons_cross))
    connections = []
    # Create irregular connections
    connections.append(Synapse(neurons_sort[0][-2], neurons_cross[0][0], w=1., d=1))    # top 1 to top 1
    connections.append(Synapse(neurons_sort[0][-1], neurons_cross[1][0], w=1., d=1))    # top 2 to top 3
    connections.append(Synapse(neurons_sort[0][-2], neurons_cross[-1][1], w=1., d=1))   # top 1 ot bottom -1
    connections.append(Synapse(neurons_sort[-1][-2], neurons_cross[-2][1], w=1., d=1))  # bottom -2 to bottom -3
    # Create regular connections
    for i in range(1, len(neurons_sort) - 1):
        connections.append(Synapse(neurons_sort[i][-2], neurons_cross[i - 1][1], w=1., d=1))  # up one lane pair
        connections.append(Synapse(neurons_sort[i][-1], neurons_cross[i + 1][0], w=1., d=1))  # down one lane pair
    return connections


def connect_cross_mutation(neurons_cross, neurons_mutation):
    """Connects a list of lists of ensemble neurons to another list of lists of ensemble neurons.
    :param neurons_cross: (list) list of lists of ensemble neurons
    :param neurons_mutation: (list) list of lists of ensemble neurons
    :return: (list) connections
    """
    assert len(neurons_cross) >= 1, "number of ensembles must be >= 1, got n=%d" % len(neurons_cross)
    assert len(neurons_cross) * 2 == len(neurons_mutation),\
        "number of ensembles must match 1:2, got n_cross=%d, n_mutation=%d" % \
        (len(neurons_cross), len(neurons_mutation))
    connections = []
    # Create regular connections
    for i in range(len(neurons_cross)):
        connections.append(Synapse(neurons_cross[i][-2], neurons_mutation[i * 2][0], w=1., d=1))
        connections.append(Synapse(neurons_cross[i][-1], neurons_mutation[i * 2 + 1][0], w=1., d=1))
    return connections


def connect_mutation_eval(neurons_mutation, neurons_eval, delay=1):
    """Connects a list of lists of ensemble neurons to another list of lists of ensemble neurons.
    :param neurons_mutation: (list) list of lists of ensemble neurons
    :param neurons_eval: (list) list of lists of ensemble neurons
    :param delay: (int) optional delay to visibly separate generations in the raster plot
    :return: (list) connections
    """
    assert len(neurons_mutation) >= 1, "number of ensembles must be >= 1, got n=%d" % len(neurons_mutation)
    assert len(neurons_mutation) == 2 * len(neurons_eval),\
        "number of ensembles must match 2:1, got n_mutation=%d, n_eval=%d" % \
        (len(neurons_mutation), len(neurons_eval))
    connections = []
    # Create regular connections
    for i in range(len(neurons_eval)):
        connections.append(Synapse(neurons_mutation[i * 2][-1], neurons_eval[i][0], w=1., d=delay))
        connections.append(Synapse(neurons_mutation[i * 2 + 1][-1], neurons_eval[i][1], w=1., d=delay))
    return connections


def flatten_list(list):
    """Flattens a list of lists.
    :param list: (list) list of lists
    :return: (list) flattened list
    """
    return [item for sublist in list for item in sublist]


def create_network(n_chromosomes, seq_len, zeros_init=False, scaling_mutation=False):
    """Creates the network to run a genetic algorithm with the defined parameters.
    Returns lists of nodes and connections.
    :param n_chromosomes: (int) number of chromosomes >= 6
    :param seq_len: (int) sequence length of each chromosome excluding the lead bit
    :param zeros_init: (boolean) initialize chromosomes with zeros, otherwise random
    :param scaling_mutation: (boolean) scale mutation rate based on fitness
    :return: (lists...) inputs, neurons, outputs, sort_out, connections
    """
    assert n_chromosomes >= 6

    inputs = []
    neurons = []
    outputs = []        # only output neurons of ensembles for raster plotting
    sort_out = []       # only output neurons of sort ensembles to show generations
    connections = []    # connections within and between ensembles

    # Initialize chromosomes
    for i in range(n_chromosomes):
        if zeros_init:
            chromosome = [0] * seq_len  # zeros chromosome
        else:
            chromosome = list(np.random.randint(0, 2, size=seq_len))  # random chromosome
        inputs.append(InputTrain([1] + chromosome, loop=False, name='i%d' % i))

    # Define ensemble layers (neurons, connections)
    n_eval, c_eval = stack_evaluation(n_chromosomes, seq_len)
    n_sort, c_sort = stack_bubblesort(n_chromosomes, seq_len)
    n_cros, c_cros = stack_crossover(n_chromosomes, seq_len)
    n_muta, c_muta = stack_mutation(n_chromosomes, seq_len, base_p=0.5 / seq_len,
                                    scaling_mutation=scaling_mutation)
    neurons.extend(flatten_list(n_eval))
    neurons.extend(flatten_list(n_sort))
    neurons.extend(flatten_list(n_cros))
    neurons.extend(flatten_list(n_muta))
    connections.extend(c_eval)
    connections.extend(c_sort)
    connections.extend(c_cros)
    connections.extend(c_muta)

    # Connect inputs and ensembles to form full generational loop
    connections.extend(connect_inputs_eval(inputs, n_eval))
    connections.extend(connect_eval_sort(n_eval, n_sort))
    connections.extend(connect_sort_cross(n_sort, n_cros))
    connections.extend(connect_cross_mutation(n_cros, n_muta))
    connections.extend(connect_mutation_eval(n_muta, n_eval))

    # Collect output neurons for plotting
    for sublist in n_eval:
        outputs.extend(sublist[-3:])
    for sublist in n_sort:
        outputs.extend(sublist[-2:])
    for sublist in n_cros:
        outputs.extend(sublist[-2:])
    for sublist in n_muta:
        outputs.append(sublist[-1])
    for sublist in n_sort:
        sort_out.extend(sublist[-2:])

    return inputs, neurons, outputs, sort_out, connections


# Define parameters
n_chromosomes = 16  # minimum 6
seq_len = 8  # excludes lead bit
n_steps = 1000
zeros_init = True
scaling_mutation = False

# Define the network (nodes, connections)
inputs, neurons, outputs, sort_out, connections = create_network(n_chromosomes, seq_len,
                                                                 zeros_init=zeros_init,
                                                                 scaling_mutation=scaling_mutation)
net = Network(inputs + neurons, connections)
print('inputs:     ', len(inputs))
print('neurons:    ', len(neurons))
print('connections:', len(connections))

# Create recording devices
##raster = Raster(inputs + neurons)
##raster = Raster(inputs + outputs)
##raster = Raster(neurons)
raster = Raster(sort_out)

# Create and run simulator
sim = Simulator(net, [raster])
sim.run(n_steps)

# Plot recordings
plt.rcParams["font.family"] = 'Courier New'
plt.figure()
raster.plot()
plt.show()

# Plot solution quality over time (requires raster on only sort outputs)
scores_average = []
scores_top = []
spikes = raster.spikes.T
lead_steps = np.where(np.all(spikes == True, axis=0))[0]  # column indices of lead bits
# Remove columns where all chromosomes spike, which are false positives for lead steps
false_positives = []
last_lead = 0
for k in range(1, len(lead_steps)):
    if lead_steps[k] < last_lead + seq_len + 1:
        false_positives.append(k)
    else:
        last_lead = lead_steps[k]
lead_steps = np.delete(lead_steps, false_positives)
##print("lead_steps:", lead_steps)
# Gather scores
for i in range(len(lead_steps) - 1):
    gen_begin = lead_steps[i] + 1       # column index of first bit in generation
    gen_end = gen_begin + seq_len - 1   # column index of last bit in generation
    scores_average.append(spikes[:, gen_begin:gen_end + 1].sum() / n_chromosomes)
    scores_top.append(spikes[0, gen_begin:gen_end + 1].sum())
# Plot solution quality
plt.figure()
plt.hlines(seq_len, xmin=0, xmax=len(lead_steps) - 2, linestyles='--')  # maximum score in onemax
plt.hlines(0.875 * seq_len, xmin=0, xmax=len(lead_steps) - 2, linestyles=':')  # accepted score
plt.plot(scores_average, label='avg')
plt.plot(scores_top, label='top')
plt.xlabel('generations')
plt.ylabel('onemax score')
plt.legend()
plt.show()


'''
# fixed seq_len=16, n_chromosomes=[8, 16, 32, 64, 128]
neurons = [172, 344, 688, 1376, 2752]
connections = [352, 704, 1408, 2816, 5632]
gen_time = [33, 33, 33, 33, 33]
spikes_mean = [1022, 2052, 4100, 8177, 16308]
spikes_std = [73, 81, 126, 168, 226]

# fixed n_chromosomes=16, seq_len=[8, 16, 32, 64, 128]
neurons = [344, 344, 344, 344, 344]
connections = [704, 704, 704, 704, 704]
gen_time = [25, 33, 49, 81, 145]
spikes_mean = [1143, 2043, 3848, 7412, 14462]
spikes_std = [61, 75, 110, 203, 318]

np.save('seq_spikes_std_fixed_nchrms.npy', np.array([61, 75, 110, 203, 318]))
'''
