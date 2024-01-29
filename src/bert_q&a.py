# -*- coding: utf-8 -*-
!pip install tf-models-official==2.2
!pip install tensorflow==2.2

import tensorflow as tf
tf.__version__

import tensorflow_hub as hub
from official.nlp.bert.tokenization import FullTokenizer
from official.nlp.bert.input_pipeline import create_squad_dataset
from official.nlp.data.squad_lib import generate_tf_record_from_json_file
#from official.nlp.bert.data.squad_lib import generate_tf_record_from_json_file
from official.nlp.data.squad_lib import generate_tf_record_from_json_file
from official.nlp import optimization
from official.nlp.data.squad_lib import read_squad_examples
from official.nlp.data.squad_lib import FeatureWriter
from official.nlp.data.squad_lib import convert_examples_to_features
from official.nlp.data.squad_lib import write_predictions

import numpy as np
import math
import random
import time
import json
import collections
import os
from google.colab import drive