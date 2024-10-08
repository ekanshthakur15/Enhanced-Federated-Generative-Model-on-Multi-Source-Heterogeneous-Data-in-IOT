import torch
from pytorch_pretrained_gans import make_gan


class ClientGAN:
    def __init__(self, gan_type='biggan', model_name='biggan-deep-256'):
        self.generator = make_gan(gan_type=gan_type, model_name=model_name)
    
    def get_model_params(self):
        return self.generator.state_dict()
    
    def set_model_params(self, params):
        self.generator.load_state_dict(params)
    
    def generate_data(self, z, y):
        return self.generator(z=z, y=y)

def fed_avg(models):
    """FedAvg function to average weights from different clients."""
    avg_model = models[0].get_model_params()
    for k in avg_model.keys():
        for i in range(1, len(models)):
            avg_model[k] += models[i].get_model_params()[k]
        avg_model[k] = avg_model[k] / len(models)
    
    return avg_model
