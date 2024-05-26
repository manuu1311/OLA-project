import numpy as np

class PricingEnvironment:
    def __init__(self, norm_distributions, cost, indexes):
        '''
        norm_distributions: array of normalized distributions
        cost: cost
        indexes: array of indexes that indicate when a new distribution should start
        '''
        self.norm_distributions=norm_distributions
        self.cost = cost
        self.indexes=indexes
        #num of rounds played
        self.t=0
        #which distribution to use
        self.i=0

    def round(self, p_t,num_cust,debug=False,clairvoyant=False):
        #clairvoyant: boolean, if true, env will not increase t
        norm_distribution=self.norm_distributions[self.i]
        d_t = np.random.binomial(num_cust,norm_distribution(p_t))
        r_t = (p_t-self.cost)*d_t
        if not clairvoyant:
            self.t+=1
            if self.t>=self.indexes[self.i]:
                self.i+=1
                if debug:
                    print('Switching to distribution number',self.i, 'in iteration',self.t)
                    [500,1000,1500,2000]
        return d_t, r_t


class Two_items_PricingEnvironment:
    def __init__(self, dem_curve1,dem_curve2,c1,c2):
        self.dem1=dem_curve1
        self.dem2=dem_curve2
        self.c1=c1
        self.c2=c2
        

    def round(self,p1,p2,num_cust):
        n1=np.random.binomial(num_cust,self.dem1(p1,p2))
        profit_1=n1*(p1-self.c1)
        n2=np.random.binomial(num_cust,self.dem2(p1,p2))
        profit_2=n2*(p2-self.c2)
        return n1,n2,profit_1,profit_2
