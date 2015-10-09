"""The Columnar Machine software.

Notes
-----

The classes conform to the pylearn2 API, and should be used through
the provided pylearn2 YAML configuration files.

Two pylearn2 configuration files are included:

ae.yaml
    Configuration file for the autoencoder

cm.yaml
    Configuration file for the Columnar Machine.

The module should be called as a Python script:

$ python2 cm.py

The script figures out the necessary parameters (e.g., the number of
groups), substitutes them into the configuration files, and runs the
CM algorithm with the autoencoder pretraining step on all the datasets
named in the ``datasets`` variable.
"""

import numpy as np
import theano.tensor as T
import theano

from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin

import pickle

def shrinkage_func(V, theta):
    """The shrinkage function.

    Parameters
    ----------
    V : numpy array
        The input matrix.
    theta : numpy array
        The vector of thresholds.

    Returns
    -------
    Theano expression
        Expression to compute the shrinkage function.
    """
    return T.sgn(V) * (abs(V) - theta) * (abs(V) > theta)

class L2Cost(DefaultDataSpecsMixin, Cost):
    """L2 cost function for the CM."""

    supervised = True

    def expr(self, model, data, **kwargs):
        """Implements the L2 cost function.

        Parameters
        ----------
        model : CMModel
            The model to compute the cost function for.
        data : pylearn2 dataset
            The data the cost function is computed on.

        Returns
        -------
        Theano expression
            The cost function.
        """
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data
        outputs = model.fprop(inputs)
        loss = ((outputs - targets) ** 2).sum(axis=1).mean()/2
        return loss

class CMModel(Model):
    """Model for the Columnar Machine.

    Parameters
    ----------
    nvis : int
        The number of visible units.
    nhid: int
        The number of hidden units.
    num_S : int
        The number of hidden layers, 0 or 1.
    init_W : filename of a pylearn2 model
        The model to load the initial weight matrix for W from. If None, W is initialized randomly.

    Attributes
    ----------
    nvis : int
        The number of visible units.
    nhid: int
        The number of hidden units.
    num_S : int
        The number of hidden layers, 0 or 1.
    W : Theano shared variable
        The weight matrix for the input layer.
    S : Theano shared variable
        The weight matrix for the hidden layer.
    theta : Theano shared variable
        The vector of thresholds for the shrinkage function.
    _params : list of Theano shared variables
        The parameters to optimize.
    input_space : pylearn2.space.VectorSpace
        The space of the inputs.
    output_space : pylearn2.space.VectorSpace
        The space of the outputs.
    """

    def __init__(self, nvis, nhid, num_S=0, init_W=None):
        super(CMModel, self).__init__()

        self.nvis = nvis
        self.nhid = nhid
        self.num_S = num_S
        assert num_S in {0, 1}, "Currently only num_S == 0 or num_S == 1 is supported!"

        if init_W:
            model = pickle.load(open(init_W, "rb"))
            W = model.W.get_value()
            self.W = sharedX(W)
        else:
            self.W = sharedX(np.random.uniform(-1e-3, 1e-3, (nhid, nvis)))

        self.S = sharedX(np.random.uniform(-1e-3, 1e-3, (nhid, nhid)))
        self.theta = sharedX(np.zeros(nhid))

        if self.num_S > 0:
            self._params = [self.W, self.S, self.theta]
        else:
            self._params = [self.W, self.theta]

        self.input_space = VectorSpace(dim=nvis)
        self.output_space = VectorSpace(dim=nhid)

    def fprop(self, x):
        """Produces the expression of the forward propagation.

        Parameters
        ----------
        x : Theano variable
            The input variable.

        Returns
        -------
        Theano expression
            Expression to compute the forward propagation step.
        """
        B = T.dot(x, self.W.T)
        out = shrinkage_func(B, self.theta)
        for i in range(self.num_S):
            out = shrinkage_func(B + T.dot(out, self.S.T), self.theta)
        return out

    def get_weights(self):
        return self.W.get_value()

class AECost(DefaultDataSpecsMixin, Cost):
    """Cost function for the autoencoder."""

    supervised = True

    def expr(self, model, data, **kwargs):
        """Implements the cost function for the autoencoder.

        Parameters
        ----------
        model : AEModel
            The model for the autoencoder.
        data : pylearn2 dataset
            The data the cost function is computed on.

        Returns
        -------
        Theano expression
            The loss function.
        """
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data
        outputs = model.fprop(inputs)
        loss = ((outputs - inputs) ** 2).sum(axis=1).mean()/2
        return loss

class AEModel(Model):
    """Model for the autoencoder.

    Parameters
    ----------
    nvis : int
        The number of visible units.
    nhid: int
        The number of hidden units.
    
    Attributes
    ----------
    nvis : int
        The number of visible units.
    nhid: int
        The number of hidden units.
    W : Theano shared variable
        The weight matrix for the input layer.
    W_prime : Theano shared variable
        The weight matrix tied with W.
    theta : Theano shared variable
        The vector of thresholds for W.
    theta_prime : Theano shared variable
        The vector of thresholds for W_prime.
    _params : list of Theano shared variables
        The parameters to optimize.
    input_space : pylearn2.space.VectorSpace
        The vector space of the inputs.
    output_space : pylearn2.space.VectorSpace
        The vector space of the outputs.
    """

    def __init__(self, nvis, nhid):
        super(AEModel, self).__init__()

        self.nvis = nvis
        self.nhid = nhid

        self.W = sharedX(np.random.uniform(-1e-3, 1e-3, (nhid, nvis)), name="W")
        self.W_prime = self.W.T
        self.theta = sharedX(np.zeros(nhid))
        self.theta_prime = sharedX(np.zeros(nvis))

        self._params = [self.W, self.theta, self.theta_prime]

        self.input_space = VectorSpace(dim=nvis)
        self.output_space = VectorSpace(dim=nhid)

    def encode(self, x):
        """Implements the encoding phase for the autoencoder."""
        return shrinkage_func(T.dot(x, self.W.T), self.theta)

    def decode(self, x):
        """Implements the decoding phase for the autoencoder."""
        return shrinkage_func(T.dot(x, self.W_prime.T), self.theta_prime)

    def fprop(self, x):
        """Produces the expression of the forward propagation."""
        return self.decode(self.encode(x))

from pylearn2.config import yaml_parse
import os

def create_params(dataset):
    """Determines some parameters for a dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset

    Returns
    -------
    dict
        A dictionary with the computed parameters.
    """
    f = np.load(os.path.join(dataset, "test.npz"))
    nvis = f["X"].shape[1]
    nhid = f["G"].shape[1]
    return {"dataset": dataset, "nvis": nvis, "nhid": nhid}

if __name__ == "__main__":
    """This loop trains the autoencoder and the Columnar Machine for the
    datasets in the list ``datasets``. The currently filled in values
    are the names of the datasets in the paper.
    """
    datasets = ["random", "random4", "sport", "football"]
    ae_yaml = open("ae.yaml").read()
    cm_yaml = open("cm.yaml").read()
    for dataset in datasets:
        params = create_params(dataset)
        ae_conf = ae_yaml % params
        train = yaml_parse.load(ae_conf)
        train.main_loop()
        params["num_S"] = 0
        cm_conf = cm_yaml % params
        train = yaml_parse.load(cm_conf)
        train.main_loop()
        params["num_S"] = 1
        cm_conf = cm_yaml % params
        train = yaml_parse.load(cm_conf)
        train.main_loop()
