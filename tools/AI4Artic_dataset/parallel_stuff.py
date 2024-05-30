import multiprocessing
from functools import partial

def Parallel(function, iterable, *args):
    '''
    When the iterable is too long ~> 30000. Try to  distribute it 
    in more than one run. Otherwise it takes too long to setup the 
    parallel process.
    '''
    n_cores = multiprocessing.cpu_count() - 1
    print('Configuring CPU multiprocessing...')
    print('Number of cores: %d'%(n_cores))
    p = multiprocessing.Pool(n_cores)
    func = partial(function, *args)
    x = p.map(func, iterable)
    p.close()
    p.join()

    return x