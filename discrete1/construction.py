""" 
Converting the user-defined input into the correct format
    - Construct the mapping vector
    - Set up the cross sections
    - Set up boundary conditions
""" 

class Construct:

    def __init__(self,dictionary):
        self.cells = dictionary['spatial']
        self.groups = dictionary['energy']
        self.map_item = dictionary['mapping'].copy()
        self.materials = dictionary['materials'].copy()
        self.boundaries = dictionary['boundaries'].copy()

    def run(self):
        Construct.set_mapping(self)
        Construct.set_cross_sections(self)

    def set_mapping(self):
        print(self.map_item)
        print(self.cells)
        ...

    def set_cross_sections(self):
        print('Here')
        ...

    def set_boundaries(self):
        ...


# class Chemistry:
