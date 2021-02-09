"""
How to Return the Number Density of Different Materials
Current Materials Available
    - Uranium Hydride (enriched or depleted Uranium)
    - High Density Polyethylene
    - Stainless Steel 440
Pu239 and Pu240 Need to be looked at
"""

import json, re
import pkg_resources

# Used for determining path to dictionaries
DATA_PATH = pkg_resources.resource_filename('discrete1','data/')

class NumberDensity:
    """ Main class for running to create number density
    Returns a dictionary of elements with number densities """
    __allowed = ("enrich")
    __compounds = ("UH3","HDPE","SS440","U")

    def __init__(self,compound,**kwargs):
        assert (compound in self.__class__.__compounds), "Compound not allowed, available: UH3, HDPE, SS440, U"
        self.compound = compound
        # Kwargs
        self.enrich = 0.0; 
        # Make sure user inputs right kwargs
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: enrich, energy" 
            setattr(self, key, value)


    def run(self):
        if self.compound == 'UH3':
            return UH3(self.enrich).number_density()
        
        elif self.compound == 'SS440':
            return SS440().number_density()

        elif self.compound == 'HDPE':
            return HDPE().number_density()

        elif self.compound == 'Pu':
            return Pu(self.enrich).number_density()

        elif self.compound == 'U':
            return U(self.enrich).number_density()


class HDPE:
    __isotopes = ['cnat','h1']

    def __init__(self):
        return None

    # def percentage(self):
    #     return None

    def molar_mass(self):
        """ Needed if not already in .json file """
        # Add natural carbon
        # hdpe_molar = _Constants.compound_density['C'][0]
        # Add hydrogen
        # hdpe_molar += _Constants.compound_density['H'][0] * 3
        # return hdpe_molar
        return _Constants.compound_density['HDPE'][0]

    def number_density(self):
        density_list = {}
        hdpe_molar = HDPE.molar_mass(self)
        rho = _Constants.compound_density['HDPE'][1]
        # Add cnat
        density_list['cnat'] = (rho * _Constants.avagadro) / hdpe_molar * _Constants.barn
        # Add H1
        density_list['h1'] = (rho * _Constants.avagadro) / hdpe_molar * 3 * _Constants.barn
        
        return density_list 


class UH3:
    __isotopes = ['u235','u238','h1']

    def __init__(self,enrich=0.0):
        self.enrich = enrich

    def molar_mass(self):
        # Add enriched UH3
        uh3_molar = self.enrich * _Constants.compound_density['U235'][0]
        # Add depleted UH3
        uh3_molar += (1 - self.enrich) * _Constants.compound_density['U238'][0]
        # Add Hydrogen
        uh3_molar += _Constants.compound_density['H'][0] * 3

        return uh3_molar

    def number_density(self):
        density_list = {}
        uh3_molar = UH3.molar_mass(self)
        rho = _Constants.compound_density['UH3'][1]
        library = _Constants.compound_density.copy()
        # Add Enriched U235
        density_list['u235'] = (self.enrich * rho * _Constants.avagadro) / library['U235'][0] * \
            (self.enrich * library['U235'][0] + (1 - self.enrich) * library['U238'][0]) / uh3_molar * _Constants.barn
        # Add U238
        density_list['u238'] = ((1 - self.enrich) * rho * _Constants.avagadro) / library['U238'][0] * \
            (self.enrich * library['U235'][0] + (1 - self.enrich) * library['U238'][0]) / uh3_molar * _Constants.barn
        # Add H1
        density_list['h1'] = (rho * _Constants.avagadro) / uh3_molar * 3 * _Constants.barn

        return density_list


class U:
    __isotopes = ['u235','u238']

    def __init__(self,enrich=0.0):
        self.enrich = enrich

    def molar_mass(self):
        # Add U-235
        u_molar = self.enrich * _Constants.compound_density['U235'][0]
        # Add U-238
        u_molar += (1 - self.enrich) * _Constants.compound_density['U238'][0]

        return u_molar

    def number_density(self):
        density_list = {}
        u_molar = U.molar_mass(self)
        rho = _Constants.compound_density['U'][1]
        library = _Constants.compound_density.copy()
        # Add Enriched U235
        density_list['u235'] = (self.enrich * rho * _Constants.avagadro) / library['U235'][0] * \
            (self.enrich * library['U235'][0] + (1 - self.enrich) * library['U238'][0]) / u_molar * _Constants.barn
        # Add U238
        density_list['u238'] = ((1 - self.enrich) * rho * _Constants.avagadro) / library['U238'][0] * \
            (self.enrich * library['U235'][0] + (1 - self.enrich) * library['U238'][0]) / u_molar * _Constants.barn
        
        return density_list

class SS440:
    __isotopes = ['fe54','fe56','fe57','cr50','cr52','cr53','cr54','si28','si29',
            'si30','mn55','cnat']
    __elements = ['fe','cr','si','mn','cnat']

    def __init__(self):
        return None

    def percentage(self):
        """ Returns dictionary of percentage of each element in SS440 """
        # Percentage of Materials in SS440
        self.fe_percent = 0.79; self.cr_percent = 0.18; 
        self.cnat_percent = 0.01; 
        self.si_percent = 0.01; self.mn_percent = 0.01; 
        # Percent of Each Element
        percent_dictionary = {}
        percent = [self.fe_percent,self.cr_percent,self.si_percent,self.mn_percent,self.cnat_percent]
        for ii in range(len(self.__class__.__elements)):
            percent_dictionary[self.__class__.__elements[ii]] = percent[ii]
        return percent_dictionary

    def molar_mass(self):
        # Call to get percentages 
        _ = SS440.percentage(self)
        # Calculate Individual Molar Mass
        fe_molar = _Tools.natural_compound('fe',self.__class__.__isotopes)
        cr_molar = _Tools.natural_compound('cr',self.__class__.__isotopes)
        cnat_molar = _Tools.natural_compound('cnat',self.__class__.__isotopes)
        si_molar = _Tools.natural_compound('si',self.__class__.__isotopes)
        mn_molar = _Tools.natural_compound('mn',self.__class__.__isotopes)
        # Combine for Molar Mass of SS440
        ss440_molar = 1/(self.fe_percent/fe_molar + self.cr_percent/cr_molar + 
            self.cnat_percent/cnat_molar + self.si_percent/si_molar + self.mn_percent/mn_molar)
        return ss440_molar

    def number_density(self):
        density_list = {}
        # Get elements of isotopes
        elements = [re.sub(r'[0-9]','',ii) for ii in self.__class__.__isotopes]
        # Get Percentage dictionary
        percent = SS440.percentage(self)
        # Density of SS440
        rho = _Constants.compound_density['SS440'][1]
        for iso,ele in zip(self.__class__.__isotopes,elements):
            density_list[iso] = (_Constants.isotope_abundance[iso][0]*percent[ele]*rho*_Constants.avagadro)/ \
                _Constants.isotope_abundance[iso][1]*_Constants.barn 
        return density_list


class Pu240:
    __isotopes = ['pu240']

    def __init__(self):
        return None

    # def percentage(self):
    #     return None


class Pu239:
    __isotopes = ['pu239','pu240']

    def __init__(self,enrich=0.0):
        self.enrich = enrich

    # def percentage(self):
    #     return None

    def molar_mass(self):
        # Add Pu239
        pu_molar = (1 - self.enrich) * _Constants.compound_density['Pu239'][0]
        # Add Pu240
        pu_molar += self.enrich * _Constants.compound_density['Pu240'][0]

        return pu_molar

    def number_density(self):
        density_list = {}
        pu_molar = Pu239.molar_mass(self)
        rho = _Constants.compound_density['UH3'][1]
        library = _Constants.compound_density.copy()
        # Add Enriched U235
        density_list['u235'] = (self.enrich * rho * _Constants.avagadro) / library['U235'][0] * \
            (self.enrich * library['U235'][0] + (1 - self.enrich) * library['U238'][0]) / uh3_molar * _Constants.barn
        # Add U238
        density_list['u238'] = ((1 - self.enrich) * rho * _Constants.avagadro) / library['U238'][0] * \
            (self.enrich * library['U235'][0] + (1 - self.enrich) * library['U238'][0]) / uh3_molar * _Constants.barn
        # Add H1
        density_list['h1'] = (rho * _Constants.avagadro) / uh3_molar * 3 * _Constants.barn

        return density_list


class _Constants:
    # Constants
    avagadro = 6.022E23
    barn = 1E-24

    # Dictionaries
    # ['Abundance %','molar mass']
    isotope_abundance = json.load(open(DATA_PATH + 'isotope_abundance.json','r'))
    # ['molar mass','density']
    compound_density = json.load(open(DATA_PATH + 'compound_density.json','r'))


class _Tools:
    def natural_compound(element,isotopes):
        """ Calculating the Molar Mass of the Element from 
        the isotopes given and the abundance dictionary  """
        return 1/sum([_Constants.isotope_abundance[iso][0]/_Constants.isotope_abundance[iso][1] for iso in isotopes if element in iso]) 

