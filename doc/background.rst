.. _section-mathematical-background:

Mathematical background
=======================

The following sections describe some of the underlying mathematics of methods
used in `hdnet`.


.. _section-mathematical-background-hopfield:

Hopfield Networks
-----------------

Hopfield network

    Hopfield Networks are a form of associative memory(structurally similar to
    recurrent Neural Networks). They are often considered to model human memory.
    A hopfield Network consists of nodes as its structural units, if there are
    *n* nodes.Each node has a connection to every other node. The connections
    are described with the help of a symmetric weight matrix **J**, which
    has all diagonal-entries set zero.

    .. figure:: /figures/Hopfield-net-vector.svg.png
        :width: 33%
        :align: center

       Hopfield Network. (Source: Wikipedia)

    The individual units/nodes in these networks are updated asynchronously on
    a random basis, making them similar to biological systems.

    Like in Neural Networks, each unit is updated by re-calculating its activation
    value(binary: 0 or 1), by taking the dot-product between the activations of
    all the nodes in the network with the row-vector from the Weight matrix,
    corresponding to the unit which we want to update, and passing it to an
    activation function, which maybe taken as the Heaviside function(H), which
    takes the value 1, if the value is greater than a threshold, and 0 otherwise.

    .. figure:: /figures/single_node_update.png
        :width: 33%
        :align: center

       Asynchronous single-node updates.



    It is provable, that after finitely many iterations of the above procedure,
    a stable state is reached wherein, any update won't change any of the
    activations.

    It's also provable that a Hopfield Network with *n* nodes can accurately store
    *n* binary patterns, 1 pattern per neuron.

    Proving the former statement, invokes a very important definition for
    Bi-directional Associative Memory(BAM) i.e. Energy. The Energy of a given
    overall state of a BAM is defined as:

    .. figure:: /figures/energy.png
        :width: 33%
        :align: center

       The Energy Function.

    The theta-part of the above expression denotes the different thresholds of the
    Heaviside function for different units/nodes.
    Without a rigorous proof, it's intuitively obvious that as the state of the
    Hopfield Network approaches stability, the energy tends to decrease.

    Therefore, Training the network essentially means decreasing the Energy of
    the network.


    Many techniques are available for this: Outer Product Learning Rule(OPR),
    Perceptron Learning Rule(PER) and the most efficient one, Minimum
    Probability Rule(MPF), which we shall be using in HDNet.

    You may learn more about Hopfield Networks from :
    http://page.mi.fu-berlin.de/rojas/neural/chapter/K13.pdf
    (Chapter on Hopfield Networks from Raul Rojas's Neural Networks: A Systematic
    Introduction)


Minimum probability flow (MPF)

    This method of minimizing the MPF Objective function in order to get an hopfield
    Network to efficiently converge to a stable state was first proposed in this
    paper : https://arxiv.org/abs/1204.2916 (co-authored by one of HDNet's creator
    Christopher Hillar).

   First let's define the neighbourhood of a particular state(denoted by **x**, the
   vector containing activations of all nodes!), as all the binary vectors which
   are Hamming Distance 1 away from that particular state i.e.(exactly having
   exactly one bit/node different than x). Let this be denoted by *N(x)*.

   Then the MPF Objective Function is defined as:

   .. figure:: /figures/objective_func.png
       :width: 33%
       :align: center

      The MPF Objective Function.

   The Learning rule for our Neural Network is henceforth defined as moving our
   parameters (**J**, theta) by a small amount in the direction of steepest
   descent of the MPF Objective function(similar to Gradient Descent!).

   As it does make it vulnerable to converge to local minima, We say that
   this Learning rule is local.

   As we could see from <100 patterns, MPF performs remarkably better than other
   learning rules.

   .. figure:: /figures/learning_curves.png
       :width: 33%
       :align: center

      Comparing MPF with other learning rules.

.. _section-mathematical-background-dichotomized:

Dichotomized Gaussian
---------------------

Dichotomy means divided. The Gaussian distribution closely models many natural
phenomenon, even when the signal/data is discrete and not continuous. Hence, in
order to generate a discrete signal, we use Dichotomized Gaussian, whereby we
divide the gaussian into many small portions, and give its values only at some
points, this maybe better understood by the following diagram:

.. figure:: /figures/dichomgauss.png
    :width: 33%
    :align: center

   Visualizing Dichotomized Gaussians.
