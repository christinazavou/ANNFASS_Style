import models.resunet as resunet
import models.res16unet as res16unet
import models.resunet_ae as resunet_ae
import models.hrnet as hrnet
import models.hrnet_ae as hrnet_ae
import models.res16unet_ae as res16unet_ae
import models.resunet_style_cls as resunet_style_cls
import models.res16unet_style_cls as res16unet_style_cls
import models.res16unet_style_cls_fromps as res16unet_style_cls_from_ps
import models.hrnet_style_cls as hrnet_style_cls
import models.hrnet_style_cls_fromps as hrnet_style_cls_from_ps

# from models.trilateral_crf import TrilateralCRF
from models.conditional_random_fields import BilateralCRF, TrilateralCRF

MODELS = []


def add_models(module):
  MODELS.extend([getattr(module, a) for a in dir(module) if 'Net' in a])


add_models(resunet)
add_models(res16unet)
add_models(hrnet)
add_models(resunet_ae)
add_models(hrnet_ae)
add_models(res16unet_ae)
add_models(resunet_style_cls)
add_models(res16unet_style_cls)
add_models(res16unet_style_cls_from_ps)
add_models(hrnet_style_cls)
add_models(hrnet_style_cls_from_ps)


WRAPPERS = [BilateralCRF, TrilateralCRF]


def get_models():
  '''Returns a tuple of sample models.'''
  return MODELS


def get_wrappers():
  return WRAPPERS


def load_model(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_models = get_models()
  mdict = {model.__name__: model for model in all_models}
  if name not in mdict:
    print('Invalid model index. Options are:')
    # Display a list of valid model names
    for model in all_models:
      print('\t* {}'.format(model.__name__))
    return None
  NetClass = mdict[name]

  return NetClass


def load_wrapper(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_wrappers = get_wrappers()
  mdict = {wrapper.__name__: wrapper for wrapper in all_wrappers}
  if name not in mdict:
    print('Invalid wrapper index. Options are:')
    # Display a list of valid model names
    for wrapper in all_wrappers:
      print('\t* {}'.format(wrapper.__name__))
    return None
  WrapperClass = mdict[name]

  return WrapperClass
