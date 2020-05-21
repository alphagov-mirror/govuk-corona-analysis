from typing import Callable, Optional, Union
import multiprocessing as mp
import numpy as np
import pandas
import warnings

# Get the total number of processors on this machine
COUNT_CPU = mp.cpu_count()


def parallelise_pandas(data: Union[pandas.Series, pandas.DataFrame],
                       func: Callable[[Union[pandas.Series, pandas.DataFrame]], Union[pandas.Series, pandas.DataFrame]],
                       n_cores: Optional[int] = None) -> Union[pandas.Series, pandas.DataFrame]:
    """Parallelise a function onto a pandas Series or DataFrame.

    Taken from: https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1

    :param data: A pandas Series or DataFrame.
    :param func: A function to apply onto `data`. The function should accept a pandas Series/DataFrame, depending on
        `data`, as an argument, and return the same type. Note that `func` cannot be a lambda function.
    :param n_cores: Default: None. The number of processors to use during parallel processing. If None, or greater than
        the actual number of available processors, this will be set to the maximum number of processors.
    :return: A pandas Series or DataFrame, depending on the type of `data`, with `func` applied to it.

    """

    # If `n_cores` is None or larger than the available number of processors, set to the maximum number of processors
    if n_cores is None or n_cores > COUNT_CPU:
        if n_cores is not None and n_cores > COUNT_CPU:
            warnings.warn(f"`n_cores` is larger than available processors ({n_cores} > {COUNT_CPU}). Setting to "
                          f"{COUNT_CPU}", RuntimeWarning)
        n_cores = COUNT_CPU

    # Split `data` into `n_cores` chunks
    data_split = np.array_split(data, n_cores)

    # Create a multiprocessing.Pool object, and map `func` to each chunk in `data_split`, returning a pandas DataFrame
    with mp.Pool(n_cores) as pool:
        out_data = pandas.concat(pool.map(func, data_split))
    pool.join()

    # Return the transformed data
    return out_data
