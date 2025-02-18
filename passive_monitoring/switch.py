# model the switch to be able to inspect each packet as it passes through
# extract relevant fields like timestamps, flow identifiers without modifying the packet or sending any additional traffic
# store arrival times for tcp flows, record the arrival of the initiating packet and then match the corresponding response and find delays accordingly
# use a sketch for scalability
# get the average, variance and percentiles for the network path
# model the switch as a match-action paradigm
# switch needs to have:
# logic for packet inspection, timestamp extraction, flow matching, and sketch updates

#overall network/graph: generate synthetic traffic, model end to end delays
# use aggregated delay on a particular graph to be able to understand its performance (leads into making traffic decisions)

# implement a sketch 
class Sketch:
    de