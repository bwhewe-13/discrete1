"""
The driver for the neutron transport equation
"""

from discrete1.construction import Construct

# import numpy as np
import json
import datetime

# Converting user input to dictionary
user_input = json.load(open('../data/input_script_source.json','r'))

# Setting a unique timestamp for each of the problems run
unique_timestamp = datetime.datetime.now().strftime("%H%M%S-%d%b%Y")


Construct(user_input).run()

