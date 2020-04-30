import numpy as np
import matplotlib.pyplot as plt

from pySimulator.simulators import Simulator
from pySimulator.nodes import InputTrain, LIF
from pySimulator.connections import Synapse
from pySimulator.networks import Network
from pySimulator.detectors import Raster, Multimeter



chrl = 8
chromosome_amount = 8 #must be even number
geneamount = chrl * chromosome_amount

allconnections = []
allneurons = []

#define random input
inputneurons = []
input_spikes = list(np.random.randint(2,size = geneamount))
for i in range(geneamount):    
    inputneurons.append(InputTrain([input_spikes[i]],loop=False))

chromosomes = []    
for i in range(geneamount):
    chromosomes.append(LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=.99, amplitude=1., I_e=0., noise=0.))
for i in range(geneamount):
    allconnections.append(Synapse(inputneurons[i],chromosomes[i],w=1. ,d=1))

#make lead bit
leadbit_input = InputTrain([1],loop=False)
leadbit = LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=.99, amplitude=1., I_e=0., noise=0.)
allconnections.append(Synapse(leadbit_input,leadbit,w=1,d=1))
allneurons = allneurons + [leadbit_input] + [leadbit] + chromosomes + inputneurons

#placeholder for eval_bubble function
def eval_bubble(chrl, eval_chromosomes, leadbit):
    connections = []
    eval_output= []
    
    chromosome1 = eval_chromosomes[:chrl]
    chromosome2 = eval_chromosomes[chrl:]
    
    identity_gates1 = []
    identity_gates2 = []
    cross_gates1 = []
    cross_gates2 = []

    output_neurons1 = []
    output_neurons2 = []
    
    # Define decider neuron (active when lower gene is preferred)
    # (when chromosomes are equal then decider is not activated)
    Decider = LIF(m=0., V_reset=0., V_min=0., thr=0.99, amplitude=1., noise=0, name='Decider')      # Decider neuron
    
    # Creation of identity gates and cross gates for both neurons stored in the respective gate_neurons lists.
    for i in range(0, chrl):
        identity_gates1.append(
        LIF(m=0., V_reset=0., V_min=0, thr=0.99, amplitude=1., noise=0, name='chrom1_identity'+str(i))
        )
        cross_gates1.append(
        LIF(m=0., V_reset=0., V_min=0., thr=1.99, amplitude=1., noise=0, name='chrom1_cross'+str(i))
        )
        
        identity_gates2.append(
        LIF(m=0., V_reset=0., V_min=0, thr=0.99, amplitude=1., noise=0, name='chrom2_identity'+str(i))
        )
        cross_gates2.append(
        LIF(m=0., V_reset=0., V_min=0., thr=1.99, amplitude=1., noise=0, name='chrom2_cross'+str(i))
        )
        
    # Creation of neurons for every output gene for both chromosomes and saved to the proper list.
    for i in range(0, chrl):
        output_neurons1.append(
        LIF(m=0., V_reset=0., V_min=0., thr=0.99, amplitude=1., noise=0, name='chrom1_out'+str(i))
        )
        output_neurons2.append(
        LIF(m=0., V_reset=0., V_min=0., thr=0.99, amplitude=1., noise=0, name='chrom2_out'+str(i))
        )
    
    
    for i in range(0, chrl):
        # Connections between all genes from both chromosomes to the Decider are created.

        connections.append(Synapse(chromosome1[i], Decider, w=-1, d=1))             # Chromosome 1 has negative weights
        connections.append(Synapse(chromosome2[i], Decider, w=1, d=1))              # Chromosome 2 has positive weights

        # Connections between the input genes and their gates are added.
        connections.append(Synapse(chromosome1[i], identity_gates1[i], w=1, d=2))
        connections.append(Synapse(chromosome1[i], cross_gates1[i], w=1, d=2))

        connections.append(Synapse(chromosome2[i], identity_gates2[i], w=1, d=2))
        connections.append(Synapse(chromosome2[i], cross_gates2[i], w=1, d=2))

        # Connections between the decider and the gates are added.
        connections.append(Synapse(Decider, identity_gates1[i], w=-1, d=1))       # Identity gates have negative weights
        connections.append(Synapse(Decider, identity_gates2[i], w=-1, d=1))
    
        connections.append(Synapse(Decider, cross_gates1[i], w=1, d=1))           # Cross gates have positive weights
        connections.append(Synapse(Decider, cross_gates2[i], w=1, d=1))

        # Connections between the gates and the output neurons are added.
        connections.append(Synapse(identity_gates1[i], output_neurons1[i], w=1, d=1))
        connections.append(Synapse(identity_gates2[i], output_neurons2[i], w=1, d=1))

        connections.append(Synapse(cross_gates1[i], output_neurons2[i], w=1, d=1))
        connections.append(Synapse(cross_gates2[i], output_neurons1[i], w=1, d=1))
    
    
    
    
    eval_lead = LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=.99, amplitude=1., I_e=0., noise=0.)
    connections.append(Synapse(leadbit,eval_lead,w=1,d=3))
    
    neurons = [eval_lead]+[Decider]+identity_gates1+identity_gates2+cross_gates1+cross_gates2+output_neurons1+output_neurons2

    eval_output = output_neurons1+output_neurons2
        
    return neurons, connections, eval_lead, eval_output
    
def crossover(chrl, cross_chromosomes,leadbit):
    connections = []
    #parralel crossover with random point
    output_leadbit = LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=.99, amplitude=1., I_e=0., noise=0.)
    connections.append(Synapse(leadbit, output_leadbit,w=1,d=5))
    par_cross_input = LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=.99, amplitude=1., I_e=0., noise=0.)
    input_with_prob = []
    gateopeners = []
    n = chrl-2
    prob = 0
    if (n > 0):
        prob = 1/n
    for i in range(chrl*2):
        connections.append(Synapse(cross_chromosomes[i],par_cross_input,w=1,d=1))
    for i in range(n):
        input_with_prob.append((LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=2-prob, amplitude=1., I_e=0., noise=1.)))
    for i in range(n):
        connections.append(Synapse(par_cross_input,input_with_prob[i],w=1.,d=1))
    for i in range(n):
        gateopeners.append(LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=.99, amplitude=1., I_e=0., noise=0.))
    for i in range(n):
        for j in range(n):
            if (j >= i):
                connections.append(Synapse(input_with_prob[i],gateopeners[j],w=1.,d=1))
    iden_gate_crossover = []
    cross_gate_crossover = []
    output_crossover = []
    #define output neurons
    for i in range(n*2):
        iden_gate_crossover.append(LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=.99, amplitude=1., I_e=0., noise=0.))
        cross_gate_crossover.append(LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=1.95, amplitude=1., I_e=0., noise=0.))

    for i in range(chrl*2):
        output_crossover.append(LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=.99, amplitude=1., I_e=0., noise=0.))
    for i in range(chrl*2):
        if (i == 0 or i == chrl):
            connections.append(Synapse(cross_chromosomes[i],output_crossover[i],w=1.,d=5))
        elif(i == chrl-1):
            connections.append(Synapse(cross_chromosomes[i],output_crossover[chrl*2-1],w=1.,d=5))
        elif(i == chrl*2-1):
            connections.append(Synapse(cross_chromosomes[i],output_crossover[chrl-1],w=1.,d=5))

    for i in range(n):
        j = i + 1
        connections.append(Synapse(cross_chromosomes[j],iden_gate_crossover[i],w=1.,d=4))
        connections.append(Synapse(cross_chromosomes[j],cross_gate_crossover[i],w=1.,d=4))
        connections.append(Synapse(iden_gate_crossover[i],output_crossover[j],w=1.,d=1))
        connections.append(Synapse(cross_gate_crossover[i],output_crossover[j + chrl],w=1.,d=1))
    for i in range(n):
        j = i + chrl + 1
        connections.append(Synapse(cross_chromosomes[j],iden_gate_crossover[i+n],w=1.,d=4))
        connections.append(Synapse(cross_chromosomes[j],cross_gate_crossover[i+n],w=1.,d=4))
        connections.append(Synapse(iden_gate_crossover[i+n],output_crossover[j],w=1.,d=1))
        connections.append(Synapse(cross_gate_crossover[i+n],output_crossover[j - chrl],w=1.,d=1))    
    for i in range(n):
        connections.append(Synapse(gateopeners[i],iden_gate_crossover[i],w=-1,d=1))
        connections.append(Synapse(gateopeners[i],iden_gate_crossover[i+n],w=-1.,d=1))
        connections.append(Synapse(gateopeners[i],cross_gate_crossover[i],w=1.,d=1))
        connections.append(Synapse(gateopeners[i],cross_gate_crossover[i+n],w=1.,d=1))
        
    neurons = input_with_prob + [par_cross_input] + gateopeners + iden_gate_crossover + cross_gate_crossover + output_crossover + [output_leadbit]
    return neurons, connections, output_leadbit, output_crossover

def mutation(chrl, mut_chromosomes,leadbit1):
    n = len(mut_chromosomes)
    mutconnections = []
    topneurons = []
    bottomneurons = []
    output_neurons = []
    mut_leadbit = LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=.99, amplitude=1., I_e=0., noise=0.)
    mutconnections.append(Synapse(leadbit1,mut_leadbit,w=1,d=2))
    prob = 1/chrl
    for i in range(n):
        topneurons.append(LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=1.99-prob, amplitude=1., I_e=0., noise=1.))
        bottomneurons.append(LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=1.99-prob, amplitude=1., I_e=0., noise=1.))
        output_neurons.append(LIF(m=0., V_init=0., V_reset=0., V_min=0., thr=.99, amplitude=1., I_e=0., noise=0.))
    for i in range(n):
        mutconnections.append(Synapse(leadbit1,topneurons[i],w=1,d=1))
        mutconnections.append(Synapse(mut_chromosomes[i],topneurons[i],w=-1,d=1))
        mutconnections.append(Synapse(mut_chromosomes[i],bottomneurons[i],w=1,d=1))
        mutconnections.append(Synapse(mut_chromosomes[i],output_neurons[i],w=1,d=2))
        mutconnections.append(Synapse(topneurons[i],output_neurons[i],w=1,d=1))
        mutconnections.append(Synapse(bottomneurons[i],output_neurons[i],w=-1,d=1))
        
    mutation_neurons = topneurons + bottomneurons + output_neurons + [mut_leadbit]
    return mutconnections, mutation_neurons, mut_leadbit, output_neurons

# Define the network (nodes, connections)
def makeNet(chrl, chromosome_amount, input_chrom, input_lead):
    net_connections = []
    net_neurons = []
    
    #crossover 01 02 34 56 78 .. ..
    outputs_cross = []
    crossover_leadbit = []
    for i in range(int(chromosome_amount/2)):
        if (i==0):
            neurons, connections, cross_leadbit, output_crossover = crossover(chrl,input_chrom[:chrl*2],input_lead)
            crossover_leadbit.append(cross_leadbit)
            net_neurons.extend(neurons)
            net_connections.extend(connections)
            outputs_cross.extend(output_crossover)
        elif (i==1):    
            chromslice = input_chrom[:chrl] + input_chrom[chrl*2:chrl*3]
            neurons, connections, _ , output_crossover = crossover(chrl,chromslice,input_lead)    
            net_neurons.extend(neurons)
            net_connections.extend(connections)
            outputs_cross.extend(output_crossover)
        else:
            begin = chrl * (i * 2 - 1)
            end = begin + chrl * 2
            neurons, connections, _ , output_crossover = crossover(chrl,input_chrom[begin:end],input_lead) 
            net_neurons.extend(neurons)
            net_connections.extend(connections)
            outputs_cross.extend(output_crossover)
    #mutation
    outputs_mutation = []
    mutation_leadbit = []
    for i in range(int(chromosome_amount/2)):
        if (i == 0):
            connections, neurons, mut_leadbit, output_neurons = mutation(chrl, outputs_cross[0:chrl*2], crossover_leadbit[0])
            net_neurons.extend(neurons)
            net_connections.extend(connections)
            mutation_leadbit.append(mut_leadbit)
            outputs_mutation.extend(output_neurons)
        else:
            begin = i * chrl
            end = begin + chrl*2
            connections, neurons, _ , output_neurons = mutation(chrl, outputs_cross[begin:end], crossover_leadbit[0])
            net_neurons.extend(neurons)
            net_connections.extend(connections)
            outputs_mutation.extend(output_neurons)
    #eval        
    outputs_eval = []
    eval_leadbit = []
    for i in range(int(chromosome_amount/2)):
        if (i == 0):
            neurons, connections, eval_lead, eval_output = eval_bubble(chrl, outputs_mutation[0:chrl*2], mutation_leadbit[0])
            net_neurons.extend(neurons)
            net_connections.extend(connections)
            eval_leadbit.append(eval_lead)
            outputs_eval.extend(eval_output)
        else:
            begin = i * chrl
            end = begin + chrl*2
            neurons, connections, _ , eval_output = eval_bubble(chrl, outputs_mutation[begin:end], mutation_leadbit[0])
            net_neurons.extend(neurons)
            net_connections.extend(connections)
            outputs_eval.extend(eval_output)
    
    #connect eval to input
    net_connections.append(Synapse(eval_leadbit[0],input_lead,w=1,d=1))
    for i in range(chromosome_amount):
        if (i == 0):
            for j in range(chrl):
                print(str(j) + "  " + str(j))
                net_connections.append(Synapse(outputs_eval[j],input_chrom[j],w=1,d=1)) 
        elif (i == chromosome_amount-1):
            for j in range(chrl):
                print(str(i*chrl+j) + "  " + str(i*chrl+j))
                net_connections.append(Synapse(outputs_eval[i*chrl+j],input_chrom[i*chrl+j],w=1,d=1))
        elif (i % 2) == 0:
            for j in range(chrl):
                begin = i * chrl + j
                end = begin - chrl 
                print(str(begin) + "  " + str(end))
                net_connections.append(Synapse(outputs_eval[begin],input_chrom[end],w=1,d=1))
        else:
            for j in range(chrl):
                begin = i * chrl + j
                end = begin + chrl 
                print(str(begin) + "  " + str(end))
                net_connections.append(Synapse(outputs_eval[begin],input_chrom[end],w=1,d=1))
    
    return net_neurons, net_connections, outputs_cross, crossover_leadbit, outputs_mutation, mutation_leadbit, outputs_eval, eval_leadbit

    
net_neurons, net_connections, outputs_cross,crossover_leadbit, outputs_mutation, mutation_leadbit, outputs_eval, eval_leadbit = makeNet(chrl,chromosome_amount,chromosomes,leadbit)
print(len(outputs_cross))
allneurons = allneurons + net_neurons
allconnections = allconnections + net_connections

net = Network(allneurons, allconnections)

# Create recording devices
rasterchr = Raster(chromosomes)
raster = Raster(outputs_cross)
raster2 = Raster(crossover_leadbit)
raster3 = Raster(outputs_mutation)
raster4 = Raster(mutation_leadbit)
raster5 = Raster(outputs_eval)
raster6 = Raster(eval_leadbit)
# Create and run simulator
sim = Simulator(net, [rasterchr, raster, raster2, raster3, raster4,raster5, raster6])
sim.run(100)

# Plot recordings
rasterchr.plot()
plt.show()
raster.plot()
#print(raster.spikes.transpose())
plt.show()
raster2.plot()
plt.show()
raster3.plot()
plt.show()
raster4.plot()
plt.show()
raster5.plot()
plt.show()
raster6.plot()
plt.show()
