from .bernoulli import Bernoulli
from .discrete_arm import DiscreteArm


mapping_ARM_TYPE = {
    "B": Bernoulli, "Bernoulli": Bernoulli,
    "D": DiscreteArm, "Discrete": DiscreteArm
}