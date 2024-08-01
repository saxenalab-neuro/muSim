import config
import numpy as np
import pickle 
from xml.etree import ElementTree as ET
from SAC import kinematics_preprocessing_specs 

### PARAMETERS ###
parser = config.config_parser()
args = parser.parse_args()

#Parse the musculo xml
tree = ET.parse(args.musculoskeletal_model_path)
root = tree.getroot()

#Load the kinematics to determine the number of targets
with open(args.kinematics_path + '/kinematics.pkl', 'rb') as f:
            kin = pickle.load(f)	#[num_conditions][n_targets, num_coords, timepoints]

kinematics_train = kin['train']
num_conds = len(kinematics_train)

#Randomly sample a condition
cond_sampled = np.random.randint(0, num_conds)
num_targets = kinematics_train[cond_sampled].shape[0]

for child in root:

	if child.tag == 'worldbody':
		for current_target in range(num_targets):
			body_to_add = f'<body name="target{current_target}" pos="0.1 0.1 0.85"> \n'
			body_to_add += '<geom size="0.01 0.01 0.01" type="sphere"/> \n'

			if kinematics_preprocessing_specs.xyz_target[current_target][0]:
				body_to_add += f'<joint axis="1 0 0" name="box:x{current_target}" type="slide" limited="false" range="-6.28319  6.28319"></joint> \n'
			
			if kinematics_preprocessing_specs.xyz_target[current_target][1]:
				body_to_add += f'<joint axis="0 0 1" name="box:y{current_target}" type="slide" limited="false" range="-6.28319  6.28319"></joint> \n'

			if kinematics_preprocessing_specs.xyz_target[current_target][2]:
				body_to_add += f'<joint axis="0 1 0" name="box:z{current_target}" type="slide" limited="false" range="-6.28319  6.28319"></joint> \n'

			body_to_add += '</body>'

			child.append(ET.fromstring(body_to_add))


#Now save the updated musculoskeletal model with target bodies added
tree.write(args.musculoskeletal_model_path[:-len('musculoskeletal_model.xml')] + 'musculo_targets.xml')
