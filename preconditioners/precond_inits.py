from preconditioners.diagonal import Diagonal
from preconditioners.nystrom import Nystrom
from preconditioners.sassn import SASSN
from preconditioners.lessn import LESSN
from preconditioners.ssn import SSN

def init_preconditioner(precond_type, model, rho, rank):
    if precond_type == 'diagonal':
        return Diagonal(model, rho)
    elif precond_type == 'nystrom':
        return Nystrom(model, rho, rank)
    elif precond_type == 'sassn':
        return SASSN(model, rho, rank)
    elif precond_type == 'lessn':
        return LESSN(model, rho, rank)
    elif precond_type == 'ssn':
        return SSN(model, rho)
    else:
        raise ValueError(f"We do not support the following preconditioner type: {precond_type}")