from glcs import PC, SPC, DSC1, DSC2, SCC, DCC

import numpy as np

class GCA:
    def __init__(self, dataset):
        dataset.positions = []
        
        if dataset.plot_type == 'PC':
            dataset.vertex_count = dataset.attribute_count
            PC.PC(dataset)

        elif dataset.plot_type == 'SPC':
            dataset.vertex_count = dataset.attribute_count // 2
            if (len(dataset.dataframe.columns) - 1) % 2 == 1:
                dataset.duplicate_last_attribute()
            SPC.SPC(dataset)
            
        elif dataset.plot_type == 'DSC1':
            dataset.vertex_count = dataset.attribute_count
            DSC1.DSC1(dataset)

        elif dataset.plot_type == 'DSC2':
            dataset.vertex_count = dataset.attribute_count // 2
            if (len(dataset.dataframe.columns) - 1) % 2 == 1:
                dataset.duplicate_last_attribute()
            DSC2.DSC2(dataset)

        elif dataset.plot_type == 'SCC':
            dataset.vertex_count = dataset.attribute_count
            SCC.SCC(dataset)
        
        elif dataset.plot_type == 'DCC':
            dataset.vertex_count = dataset.attribute_count
            DCC.DCC(dataset)

        else:
            print('No type selected')    
