""" Getting number densities for different compounds """
def mm_ss440(mm=False):
    """ Information from webelements.com and matweb.com """
    # percentage, MM
    cr50 = [0.04345, 49.9460464]
    cr52 = [0.83789, 51.9405098]
    cr53 = [0.09501, 52.9406513]
    cr54 = [0.02365, 53.9388825]
    crPer = 0.18
    cr = 1/sum([ii[0]/ii[1] for ii in [cr50,cr52,cr53,cr54]])
    fe54 = [0.05845,53.9396127]
    fe56 = [0.92036,55.9349393] # 0.91754
    fe57 = [0.02119,56.9353958]
    fePer = 0.79
    fe = 1/sum([ii[0]/ii[1] for ii in [fe54,fe56,fe57]])
    si28 = [0.922297,27.9769271]
    si29 = [0.046832,28.9764949]
    si30 = [0.030872,29.9737707]
    siPer = 0.01
    si = 1/sum([ii[0]/ii[1] for ii in [si28,si29,si30]])
    mn = 54.9380471; mnPer = 0.01
    c = 12.0116; cPer = 0.01
    mm_ss440 = 1/(fePer/fe+crPer/cr+cPer/c+siPer/si+mnPer/mn)
    if mm:
        return mm_ss440
    NA = 6.022E23; rho = 7.85 # g/cc
    Nfe = [(ii[0]*fePer*rho*NA)/ii[1]*1e-24 for ii in [fe54,fe56,fe57]]
    Ncr = [(ii[0]*crPer*rho*NA)/ii[1]*1e-24 for ii in [cr50,cr52,cr53,cr54]]
    Nsi = [(ii[0]*siPer*rho*NA)/ii[1]*1e-24 for ii in [si28,si29,si30]]
    Nmn = [(mnPer*rho*NA)/mn*1e-24]
    Nc = [(cPer*rho*NA)/c*1e-24]
    number_density = list(Nfe+Ncr+Nsi+Nmn+Nc)
    return number_density

def xs_ss440(G,spec_temp):
    import numpy as np
    address_scatter = ['mydata/fe54/scatter_0{}.npy'.format(spec_temp),
                       'mydata/fe56/scatter_0{}.npy'.format(spec_temp),
                       'mydata/fe57/scatter_0{}.npy'.format(spec_temp),
                       'mydata/cr50/scatter_0{}.npy'.format(spec_temp),
                       'mydata/cr52/scatter_0{}.npy'.format(spec_temp),
                       'mydata/cr53/scatter_0{}.npy'.format(spec_temp),
                       'mydata/cr54/scatter_0{}.npy'.format(spec_temp),
                       'mydata/si28/scatter_0{}.npy'.format(spec_temp),
                       'mydata/si29/scatter_0{}.npy'.format(spec_temp),
                       'mydata/si30/scatter_0{}.npy'.format(spec_temp),
                       'mydata/mn55/scatter_0{}.npy'.format(spec_temp),
                       'mydata/cnat/scatter_0{}.npy'.format(spec_temp)]

    address_total = ['mydata/fe54/vecTotal.npy',
                     'mydata/fe56/vecTotal.npy',
                     'mydata/fe57/vecTotal.npy',
                     'mydata/cr50/vecTotal.npy',
                     'mydata/cr52/vecTotal.npy',
                     'mydata/cr53/vecTotal.npy',
                     'mydata/cr54/vecTotal.npy',
                     'mydata/si28/vecTotal.npy',
                     'mydata/si29/vecTotal.npy',
                     'mydata/si30/vecTotal.npy',
                     'mydata/mn55/vecTotal.npy',
                     'mydata/cnat/vecTotal.npy']
    number_density = mm_ss440()
    total_xs = np.zeros((G))
    scatter_xs = np.zeros((G,G))
    for ii in range(len(number_density)):
        total_xs += number_density[ii]*np.load(address_total[ii])[eval(spec_temp)]
        scatter_xs += number_density[ii]*np.load(address_scatter[ii])[0]
    return total_xs,scatter_xs


def micMac(xs,element):
    ''' Converts microscopc cross sections to macroscopic
    Arguments:
        xs: microscopic cross section to be converted
        element: string of element or list of element [molar mass,density]
            molar mass is g/mol, density is g/cm^3
    Returns:
        Macroscopic cross section
    '''
    if type(element) == str:
        import json
        library = json.load(open('discrete1/data/element_dictionary.json'))
        info = library[element]
    else:
        info = element.copy()
    return xs*1e-24*info[1]*6.022e23/info[0]

def cleaning_compound(compound):
    import re
    compound = re.findall('[A-Z][^A-Z]*',compound)
    counter = []
    for ii in range(len(compound)):
        if len(re.findall('[0-9][^0-9]*',compound[ii])) == 0:
            counter.append(1)
            compound[ii] = re.sub(r'[0-9][A-Za-z]+','',compound[ii])
        else:
            if '^' in compound[ii]:
                if '_' not in compound[ii]:
                    compound[ii] = compound[ii]+'_1'
                isotope = re.findall('\^(.*)\_',compound[ii])[0]
                counter.append(int(re.findall('\_(.*)',compound[ii])[0]))
                compound[ii] = re.sub(r'[^A-Za-z]+','',compound[ii])+''+isotope
            else:
                counter.append(int(''.join(re.findall('[0-9][^0-9]*',compound[ii]))))
                compound[ii] = re.sub(r'[0-9]+','',compound[ii])
    return compound,counter

def total_mass(compound,counter,enrich,library):
    molar_mass = 0
    for ii,jj in zip(compound,counter):
        if ii == 'U':
            molar_mass += (enrich*library['U235'][0]+(1-enrich)*library['U238'][0])*jj
        else:
            molar_mass += library[ii][0]*jj
    return molar_mass

def number_density(compound,counter,molar_mass,density,enrich,library):
    NA = 6.022E23 # Avagadros number
    density_list = []
    for ckk,kk in enumerate(compound):
        if kk == 'U':
            density_list.append(((enrich*density*NA)/library['U235'][0]*(enrich*library['U235'][0]+(1-enrich)*library['U238'][0])/molar_mass)*counter[ckk])
            density_list.append((((1-enrich)*density*NA)/library['U238'][0]*(enrich*library['U235'][0]+(1-enrich)*library['U238'][0])/molar_mass)*counter[ckk])
        else:
            density_list.append(((density*NA)/molar_mass)*counter[ckk])
    return density_list

def density_list(compound,density,enrich=0.0,library=None):
    import numpy as np
    if library is None:
        import json
        library = json.load(open('discrete1/data/compound_density.json'))

    # Formatting (compounds and count of each)
    compound,counter = cleaning_compound(compound)

    # Calculating molar mass
    molar_mass = total_mass(compound,counter,enrich,library)

    # number densities
    density_list = number_density(compound,counter,molar_mass,density,enrich,library)

    # Convert from barns
    return np.array(density_list)*1e-24

# counter = [1 if len(re.findall('[0-9][^0-9]*',compound[ii])) == 0 else int(''.join(re.findall('[0-9][^0-9]*',compound[ii]))) for ii in range(len(compound))]
# compound = [re.sub(r'[0-9]+','',ii) for ii in compound]
