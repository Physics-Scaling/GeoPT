from models import Transolver


def get_model(args):
    model_dict = {
        'Transolver': Transolver,
    }
    return model_dict[args.model].Model(args)
