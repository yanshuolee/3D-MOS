import xml.etree.ElementTree as ET
import numpy as np
import random
import os
from itertools import product

## config ##
SIZE_X = 12
SIZE_Y = 12
SIZE_Z = 12
N_TARGETS = 6 # MAX
ANGLE = 60
N_EPISODES = 100 # n_samples
SAVE_FOLDER = "/home/rsl/ysl/sim/worlds/multiuav/env-12/target"
BASE_WORLD_FILE = "/home/rsl/ysl/sim/worlds/multiuav-base.world"
## config ##

pos_population = list(product([i for i in range(1, SIZE_X)], 
                      [i for i in range(1, SIZE_Y)], 
                      [i for i in range(1, SIZE_Z-1)],
                      [i for i in range(0, 180, ANGLE)]
                      ))

def get_new_element(tag_id):
    disk = ET.Element('include')
    pos = pos_population.pop(random.randint(0,len(pos_population)-1))

    # randomly generate
    (x, y, z, theta) = pos

    driver = ET.SubElement(disk, 'pose')
    driver.text = '{} {} {} 1.5708 0 {}'.format(x, y, z, np.deg2rad(theta))
    uri = ET.SubElement(disk, 'uri')
    uri.text = 'model://april_tag{}'.format(tag_id)

    return disk


if __name__ == '__main__':

    for i in range(N_EPISODES):
        xml_tree = ET.parse(BASE_WORLD_FILE)
        devices_element = xml_tree.find('world')
        for tag_id in range(N_TARGETS):
            new_element = get_new_element(tag_id)
            devices_element.append(new_element)

        new_xml_tree_string = ET.tostring(xml_tree.getroot())
        with open(os.path.join(SAVE_FOLDER, "world-{}.xml".format(i)), "wb") as f:
            f.write(new_xml_tree_string)