


class Variable():

    def __init__(self, name, domain, order='0', condition='None', 
                 sampling_strat_inside='random', sampling_strat_bound='random'):

        if order > 0 and condition is None:
            raise Exception('For order > 0 a condition is needed')

        self.name = name
        self.domain = domain
        self.order = order
        self.condition = condition
        self.sampling_strat_inside = sampling_strat_inside
        self.sampling_strat_bound = sampling_strat_bound
        
