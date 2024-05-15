#*******************************************************************************
# Imports and Setup
#*******************************************************************************
# packages
import torch

# nflows imports
from nflows.nn.nets.resnet import ResidualNet
from nflows.transforms.autoregressive import \
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.coupling import \
    PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms import LULinear
from nflows.transforms.permutations import ReversePermutation


#*******************************************************************************
# Function Definitions
#*******************************************************************************
def create_alternating_binary_mask(features, even=True):
    '''
    Create a binary mask for coupling layers. This code is inspired by the 
    nflows package (https://github.com/bayesiains/nflows).
    Args:
        features:   number of features
    Returns:
        mask:       alternating binary mask
    '''
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def create_linear_transform(args):
    '''
    Create a linear transformation, which can be stacked as part of a flow. This
    code is inspired by the nflows package 
    (https://github.com/bayesiains/nflows).
    Args:
        args:   dictionary of flow hyperparameters
    Returns:
        a list of flow transformations
    '''
    if args['linear'] == 'permutation':
        return [ReversePermutation(features=args['features'])]
    elif args['linear'] == 'lu':
        return [
            ReversePermutation(features=args['features']),
            LULinear(args['features'], identity_init=True)
        ]
    else:
        raise ValueError
    

def create_base_transform(i, args):
    '''
    Create a neural spline base transformation, which can be stacked as part of 
    a flow. This code is inspired by the nflows package 
    (https://github.com/bayesiains/nflows).
    Args:
        i:      index of flow block
        args:   dictionary of flow hyperparameters
    Returns:
        a list of flow transformations
    '''
    if args['base'] == 'rq-ar':
        return [MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=args['features'],
            hidden_features=args['hidden_features'],
            context_features=args['context_features'],
            num_bins=args['num_bins'],
            tails='linear',
            tail_bound=args['tail_bound'],
            dropout_probability=args['dropout_probability'],
            use_batch_norm=args['use_batch_norm']
        )]
    elif args['base'] == 'rq-c':
        return [PiecewiseRationalQuadraticCouplingTransform(
            mask=create_alternating_binary_mask(args['features'], even=(i%2==0)),
            transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args['hidden_features'],
                context_features=args['context_features'],
                dropout_probability=args['dropout_probability'],
                use_batch_norm=args['use_batch_norm']
            ),
            num_bins=args['num_bins'],
            tails='linear',
            tail_bound=args['tail_bound'],
        )]
    else:
        raise ValueError
    

def create_transform(args):
    '''
    Create a flow transformation by alternating linear and base layers. This 
    code is inspired by the nflows package 
    (https://github.com/bayesiains/nflows).
    Args:
        args:       dictionary of flow hyperparameters
    Returns:
        transform:  a CompositeTransform object representing the flow transform
    '''
    transform_list = []
    for i in range(args['num_flow_steps']):
        transform_list.extend(create_linear_transform(args))
        transform_list.extend(create_base_transform(i, args))

    transform_list.extend(create_linear_transform(args))
    transform = CompositeTransform(transform_list)
    return transform