import numpy as np
from scipy import stats

def evaluate_counts_error(
        Operator,
        counts, 
        ):
    sample_means = []
    samples = 
    pass

    def evalate(a)



def evaluate_error(
        self,
        numberOfSamples=256, # of times to repeat
        sample_size=1024, # number of counts in sample
        ci=0.90, #target CI#,
        f=None,
        replace=False,
        spin_alt=False
        ):
    print('Samples: {}'.format(numberOfSamples))
    print('Sample size: {}'.format(sample_size))
    count_list = []
    N = self.qs.Ns
    if sample_size>=N*8:
        sample_size=int(N/8)
    samplesSD = []
    sample_means = []
    counts_list = {}
    for pauli,counts in self.counts.items():
        count_list = []
        for k,v in counts.items():
            count_list = count_list+[k]*v
        counts_list[pauli]=count_list
    for t in range(numberOfSamples):
        t1 = dt()
        sample_mean  = f(
                self.getRandomRDMFromCounts(
                    counts_list,sample_size
                    )
                )
        if np.isnan(sample_mean):
            continue
        else:
            sample_means.append(sample_mean)
        t2 = dt()
        #print('Time: {}'.format(t2-t1))
    t = stats.t.ppf(ci,N)
    std_err = np.std(np.asarray(sample_means),axis=0) #standard error of mean
    ci = std_err*np.sqrt(sample_size/N)*t
    return ci

def getRandomRDMFromCounts(self,counts_list,length):
    random_counts = {}
    for pauli,clist in counts_list.items():
        random_counts[pauli]={}
        sample_list = np.random.choice(clist,length,replace=False)
        for j in sample_list:
            try:
                random_counts[pauli][j]+=1
            except KeyError:
                random_counts[pauli][j]=1
    #print('Build random list: {}'.format(t3-t5))
    del self.rdm
    self.counts = random_counts
    self.construct()
    #new = self._build_mod_2RDM(random_counts)
    #print('Build 2rdm: {}'.format(t4-t3))
    return self.rdm
