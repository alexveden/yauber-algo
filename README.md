# yauber-algo
Yet Another Universal Backtesting Engine Release (YAUBER) - Algo Lib

## About
This package is a collection of standalone algorithms for financial time series analysis. 

Highlights:
- It heavily uses Numba for improving performance of the code. It's generally faster than comparable Pandas algorithms. 
- It's build based on strict principles,  see AlgoCodex section for more information.
- It's stable and well tested, this means that logic of algorithms won't change in the future and any algorithm in this package is 100% covered by unit tests.

## Algo Codex 
### Common principles of yauber-algo time-series algorithms 

1. All algorithms must work with a rolling window or should be linked to some event in the past. 
   Changing starting point of a time series must not change results of an algorithm in the future, compared to results of entire history calculation. Otherwise, SanityChecker.window_consistency test will fail.
    
2. All algorithms must be NaN friendly, the NaN means missing data and must not affect any algorithm results. 
   
   Algorithms will return NaN in the following cases:
   
   - When there is not enough data for calculation
   - When input data contains NaN, Inf or other incorrect inputs (depends on algorithm logic)
   - When algorithm cannot correctly calculate its logic or in some ambiguous cases
   - When input series has NaN at specific array index, an algorithm must return NaN at this index too. Except series referencing algorithms, like: referencing - ref() or value_when()

3. All algorithms must gracefully handle bad values like NaN/Inf to protect final results from distortion. 

4. All algorithms must work with arrays of following types: np.float, np.bool, np.int, in other cases they must throw an exception.

5. All algorithms must return results as an array of dtype=np.float, even in boolean operations. 

6. Even in boolean operations, all algorithms must use float 1.0/0.0/NaN dtype=np.float, to avoid ambiguous cases with NaN boolean operations. 

7. All algorithm must return series (Numpy array or Pandas Series) with the exactly the same length and index (for Pandas series).

8. All algorithms must be 100% covered by unit tests and pass all automatic tests of SanytyChecker class (including future reference and window consistency tests)

9. Some algorithms might explicitly refer to the future (e.g. ref(Arr, +1) - refers to future next bar). In these cases, an algorithm must warn a user about future reference, and raise an exception if the algorithm is referring to the future in a production environment. 
   These algorithms must raise YaUberAlgoWindowConsistencyError or YaUberAlgoFutureReferenceError or warn user depending on environment settings. 
   
10. In most cases, the first argument of the function must be the input array (-s), and the following should be 
    algorithm period and other   
    
11. Percents must be always returned in range [0;1]. e.g. 1.0 - means 100%, 0.33 - equal to 33%. The same for input parameters, for example q= parameter for quantile() function)

12. All algorithms that uses OHLC must check validity of Open/High/Low/Close price for each bar, i.e. H>=L, L <= C <= H, L <= O <= H

13. Logic of all algorithms must be stable, and do not change after release.  

### Installation
```
pip install git+https://github.com/alexveden/yauber-algo.git#egg=yauber_algo
```

### Usage

```python
import pandas as pd
import numpy as np
import yauber_algo.algo as a

# Getting list of all functions
# > help(a)

# Let's load financial OHLCV series from some datasource 
ohlc = pd.read_csv('<path to financial series>')

# Trading rule
long_rule = a.ma(ohlc['c'], 20) > ohlc['c']

# Historical Volatility Indicator
vola = np.sqrt(a.ema(a.roc_log(ohlc['c'], 1) ** 2, 20)) * np.sqrt(248)
```

### Copyright
MIT License

Copyright (c) 2018-2021 Aleksandr Vedeneev